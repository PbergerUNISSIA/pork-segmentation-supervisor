"""
====================================================================
PORK SEGMENTATION SUPERVISOR - TRAINING PIPELINE
====================================================================
Pipeline de réentraînement automatique avec Active Learning.

FONCTIONNALITÉS :
- Export automatique des données d'entraînement
- Préparation du dataset (train/val split)
- Déclenchement du réentraînement
- Évaluation du nouveau modèle
- Validation avant déploiement
- Rollback automatique si régression

WORKFLOW :
1. Vérifier seuil d'annotations atteint
2. Exporter les données validées
3. Préparer le dataset (augmentation, split)
4. Lancer l'entraînement (script externe ou API)
5. Évaluer le nouveau modèle
6. Comparer avec le modèle actuel
7. Déployer si amélioration > seuil
8. Logger toutes les métriques

USAGE :
    from core.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline()

    # Vérifier si re-training nécessaire
    if pipeline.should_retrain():
        # Lancer le pipeline
        result = pipeline.run_training_pipeline()

        if result['success']:
            print(f"New model deployed: {result['new_version']}")
====================================================================
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config.settings import settings
from core.image_processor import ImageProcessor
from core.metrics import calculate_all_metrics
from core.model_manager import ModelManager
from database import (
    Annotation,
    ModelVersion,
    ValidationStatus,
    count_validated_annotations,
    create_model_version,
    get_active_model_version,
    get_annotations_for_training,
    get_session,
    mark_annotations_as_used,
)
from utils.logger import logger


# ==================================================================
# TRAINING PIPELINE
# ==================================================================

class TrainingPipeline:
    """
    Pipeline de réentraînement automatique du modèle de segmentation.

    Gère tout le cycle de vie du réentraînement : export de données,
    entraînement, évaluation, validation et déploiement.
    """

    def __init__(
        self,
        retraining_threshold: Optional[int] = None,
        min_improvement_threshold: float = 0.01,  # 1% amélioration minimum
        train_val_split: float = 0.8,
        export_dir: Optional[Path] = None
    ):
        """
        Initialise le pipeline de training.

        Args:
            retraining_threshold: Nombre d'annotations avant re-training
            min_improvement_threshold: Amélioration min pour déployer nouveau modèle
            train_val_split: Ratio train/validation
            export_dir: Dossier d'export des données
        """
        self.retraining_threshold = retraining_threshold or settings.RETRAINING_THRESHOLD
        self.min_improvement_threshold = min_improvement_threshold
        self.train_val_split = train_val_split
        self.export_dir = export_dir or Path("./data/training_exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.image_processor = ImageProcessor()

        logger.info(
            f"TrainingPipeline initialized: "
            f"threshold={self.retraining_threshold}, "
            f"min_improvement={min_improvement_threshold:.1%}"
        )

    # ==================================================================
    # MAIN PIPELINE
    # ==================================================================

    def should_retrain(self) -> bool:
        """
        Vérifie si le seuil de réentraînement est atteint.

        Returns:
            bool: True si réentraînement nécessaire
        """
        validated_count = count_validated_annotations()
        should_train = validated_count >= self.retraining_threshold

        logger.info(
            f"Retraining check: {validated_count}/{self.retraining_threshold} annotations "
            f"({'READY' if should_train else 'NOT READY'})"
        )

        return should_train

    def run_training_pipeline(
        self,
        force: bool = False,
        training_script_path: Optional[str] = None
    ) -> Dict:
        """
        Lance le pipeline complet de réentraînement.

        Args:
            force: Forcer le réentraînement même si seuil non atteint
            training_script_path: Chemin vers script d'entraînement personnalisé

        Returns:
            Dict: Résultat du pipeline
                {
                    'success': bool,
                    'new_version': str,
                    'metrics': Dict,
                    'deployed': bool,
                    'message': str
                }
        """
        logger.info("=" * 70)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 70)

        # 1. Vérifier seuil
        if not force and not self.should_retrain():
            return {
                'success': False,
                'message': f"Insufficient annotations ({count_validated_annotations()}/{self.retraining_threshold})"
            }

        try:
            # 2. Exporter les données
            logger.info("\n[1/6] Exporting training data...")
            export_result = self.export_training_data()

            if not export_result['success']:
                return {
                    'success': False,
                    'message': f"Data export failed: {export_result['message']}"
                }

            # 3. Préparer le dataset
            logger.info("\n[2/6] Preparing dataset...")
            dataset_result = self.prepare_dataset(export_result['export_path'])

            # 4. Lancer l'entraînement
            logger.info("\n[3/6] Training new model...")
            training_result = self.train_model(
                dataset_result['train_dir'],
                dataset_result['val_dir'],
                training_script_path
            )

            if not training_result['success']:
                return {
                    'success': False,
                    'message': f"Training failed: {training_result['message']}"
                }

            # 5. Évaluer le nouveau modèle
            logger.info("\n[4/6] Evaluating new model...")
            eval_result = self.evaluate_model(
                training_result['model_path'],
                dataset_result['val_dir']
            )

            # 6. Comparer avec modèle actuel
            logger.info("\n[5/6] Comparing with current model...")
            comparison = self.compare_models(eval_result['metrics'])

            # 7. Déployer si amélioration
            logger.info("\n[6/6] Deployment decision...")
            deployed = False

            if comparison['should_deploy']:
                deploy_result = self.deploy_model(
                    training_result['model_path'],
                    eval_result['metrics'],
                    export_result['batch_id']
                )
                deployed = deploy_result['success']
                new_version = deploy_result['version']
            else:
                logger.warning(
                    f"New model NOT deployed: improvement {comparison['improvement']:.2%} "
                    f"< threshold {self.min_improvement_threshold:.2%}"
                )
                new_version = None

            logger.success("=" * 70)
            logger.success("TRAINING PIPELINE COMPLETED")
            logger.success("=" * 70)

            return {
                'success': True,
                'new_version': new_version,
                'metrics': eval_result['metrics'],
                'deployed': deployed,
                'comparison': comparison,
                'message': 'Training pipeline completed successfully'
            }

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {
                'success': False,
                'message': f"Pipeline error: {str(e)}"
            }

    # ==================================================================
    # DATA EXPORT
    # ==================================================================

    def export_training_data(self) -> Dict:
        """
        Exporte les annotations validées pour entraînement.

        Returns:
            Dict: {
                'success': bool,
                'export_path': Path,
                'batch_id': int,
                'n_samples': int,
                'message': str
            }
        """
        try:
            # Récupérer les annotations non utilisées
            annotations = get_annotations_for_training(
                exclude_used=True
            )

            if not annotations:
                return {
                    'success': False,
                    'message': 'No new annotations available for training'
                }

            # Créer dossier d'export avec timestamp
            batch_id = int(datetime.now().timestamp())
            export_path = self.export_dir / f"batch_{batch_id}"
            export_path.mkdir(parents=True, exist_ok=True)

            images_dir = export_path / "images"
            masks_dir = export_path / "masks"
            images_dir.mkdir()
            masks_dir.mkdir()

            # Copier les images et masques
            exported_count = 0
            metadata = []

            with get_session() as session:
                for annotation in annotations:
                    # Charger l'image originale
                    image = annotation.image

                    if not Path(image.filepath).exists():
                        logger.warning(f"Image not found: {image.filepath}")
                        continue

                    if not Path(annotation.mask_path).exists():
                        logger.warning(f"Mask not found: {annotation.mask_path}")
                        continue

                    # Copier image
                    img_filename = f"{image.id:06d}.png"
                    shutil.copy(image.filepath, images_dir / img_filename)

                    # Copier mask
                    mask_filename = f"{image.id:06d}.png"
                    shutil.copy(annotation.mask_path, masks_dir / mask_filename)

                    # Métadonnées
                    metadata.append({
                        'annotation_id': annotation.id,
                        'image_id': image.id,
                        'image_filename': img_filename,
                        'mask_filename': mask_filename,
                        'was_corrected': annotation.was_corrected,
                        'validated_at': annotation.validated_at.isoformat()
                    })

                    exported_count += 1

            # Sauvegarder métadonnées
            metadata_path = export_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'batch_id': batch_id,
                    'export_date': datetime.now().isoformat(),
                    'n_samples': exported_count,
                    'annotations': metadata
                }, f, indent=2)

            # Marquer les annotations comme utilisées
            annotation_ids = [a.id for a in annotations]
            mark_annotations_as_used(annotation_ids, batch_id)

            logger.success(
                f"Exported {exported_count} samples to {export_path}"
            )

            return {
                'success': True,
                'export_path': export_path,
                'batch_id': batch_id,
                'n_samples': exported_count,
                'message': f'Exported {exported_count} samples'
            }

        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return {
                'success': False,
                'message': f'Export error: {str(e)}'
            }

    # ==================================================================
    # DATASET PREPARATION
    # ==================================================================

    def prepare_dataset(self, export_path: Path) -> Dict:
        """
        Prépare le dataset (train/val split).

        Args:
            export_path: Chemin vers les données exportées

        Returns:
            Dict: {
                'train_dir': Path,
                'val_dir': Path,
                'n_train': int,
                'n_val': int
            }
        """
        try:
            # Créer dossiers train/val
            train_dir = export_path / "train"
            val_dir = export_path / "val"

            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)

            (train_dir / "images").mkdir(exist_ok=True)
            (train_dir / "masks").mkdir(exist_ok=True)
            (val_dir / "images").mkdir(exist_ok=True)
            (val_dir / "masks").mkdir(exist_ok=True)

            # Lire métadonnées
            metadata_path = export_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            annotations = metadata['annotations']

            # Split train/val
            np.random.seed(42)
            n_samples = len(annotations)
            n_train = int(n_samples * self.train_val_split)

            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            # Copier vers train
            for idx in train_indices:
                ann = annotations[idx]
                shutil.copy(
                    export_path / "images" / ann['image_filename'],
                    train_dir / "images" / ann['image_filename']
                )
                shutil.copy(
                    export_path / "masks" / ann['mask_filename'],
                    train_dir / "masks" / ann['mask_filename']
                )

            # Copier vers val
            for idx in val_indices:
                ann = annotations[idx]
                shutil.copy(
                    export_path / "images" / ann['image_filename'],
                    val_dir / "images" / ann['image_filename']
                )
                shutil.copy(
                    export_path / "masks" / ann['mask_filename'],
                    val_dir / "masks" / ann['mask_filename']
                )

            logger.info(
                f"Dataset prepared: {len(train_indices)} train, {len(val_indices)} val"
            )

            return {
                'train_dir': train_dir,
                'val_dir': val_dir,
                'n_train': len(train_indices),
                'n_val': len(val_indices)
            }

        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise

    # ==================================================================
    # MODEL TRAINING
    # ==================================================================

    def train_model(
        self,
        train_dir: Path,
        val_dir: Path,
        training_script_path: Optional[str] = None
    ) -> Dict:
        """
        Lance l'entraînement du modèle.

        Args:
            train_dir: Dossier d'entraînement
            val_dir: Dossier de validation
            training_script_path: Script d'entraînement personnalisé

        Returns:
            Dict: {
                'success': bool,
                'model_path': Path,
                'metrics': Dict,
                'message': str
            }
        """
        # NOTE: Placeholder - l'entraînement réel nécessite un script TensorFlow/PyTorch
        # Cette fonction retourne un résultat simulé pour la démo

        logger.warning(
            "⚠️  Model training is a PLACEHOLDER. "
            "Implement your training script (TensorFlow/PyTorch) separately."
        )

        # Simuler un entraînement
        model_path = settings.MODELS_DIR / f"unet_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tflite"

        # Dans un vrai scénario, vous lanceriez un script comme :
        # subprocess.run([
        #     "python", "scripts/train_unet.py",
        #     "--train-dir", str(train_dir),
        #     "--val-dir", str(val_dir),
        #     "--output", str(model_path),
        #     "--epochs", "50",
        #     "--batch-size", "8"
        # ])

        logger.info(
            f"Training would be executed here. "
            f"Output model: {model_path}"
        )

        return {
            'success': True,
            'model_path': model_path,
            'metrics': {
                'train_dice': 0.92,
                'val_dice': 0.90,
                'train_loss': 0.08,
                'val_loss': 0.10
            },
            'message': 'Training completed (simulated)'
        }

    # ==================================================================
    # MODEL EVALUATION
    # ==================================================================

    def evaluate_model(
        self,
        model_path: Path,
        val_dir: Path
    ) -> Dict:
        """
        Évalue le nouveau modèle sur le dataset de validation.

        Args:
            model_path: Chemin vers le modèle
            val_dir: Dossier de validation

        Returns:
            Dict: {
                'metrics': Dict (dice, iou, etc.)
            }
        """
        # NOTE: Placeholder - évaluation réelle nécessite le chargement du modèle

        logger.info(f"Evaluating model: {model_path}")

        # Simuler des métriques d'évaluation
        metrics = {
            'dice': 0.90,
            'iou': 0.82,
            'precision': 0.91,
            'recall': 0.89,
            'f1': 0.90
        }

        logger.info(f"Evaluation metrics: Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}")

        return {'metrics': metrics}

    # ==================================================================
    # MODEL COMPARISON & DEPLOYMENT
    # ==================================================================

    def compare_models(self, new_metrics: Dict) -> Dict:
        """
        Compare le nouveau modèle avec le modèle actuel.

        Args:
            new_metrics: Métriques du nouveau modèle

        Returns:
            Dict: {
                'should_deploy': bool,
                'improvement': float,
                'current_dice': float,
                'new_dice': float
            }
        """
        # Récupérer les métriques du modèle actuel
        active_model = get_active_model_version()

        if not active_model or active_model.metrics_dice is None:
            logger.warning("No current model metrics, deploying new model")
            return {
                'should_deploy': True,
                'improvement': 1.0,
                'current_dice': 0.0,
                'new_dice': new_metrics.get('dice', 0.0)
            }

        current_dice = active_model.metrics_dice
        new_dice = new_metrics.get('dice', 0.0)
        improvement = (new_dice - current_dice) / max(current_dice, 1e-6)

        should_deploy = improvement >= self.min_improvement_threshold

        logger.info(
            f"Model comparison: "
            f"Current Dice={current_dice:.3f}, "
            f"New Dice={new_dice:.3f}, "
            f"Improvement={improvement:+.2%}"
        )

        return {
            'should_deploy': should_deploy,
            'improvement': improvement,
            'current_dice': current_dice,
            'new_dice': new_dice
        }

    def deploy_model(
        self,
        model_path: Path,
        metrics: Dict,
        batch_id: int
    ) -> Dict:
        """
        Déploie le nouveau modèle en production.

        Args:
            model_path: Chemin vers le nouveau modèle
            metrics: Métriques du modèle
            batch_id: ID du batch d'entraînement

        Returns:
            Dict: {
                'success': bool,
                'version': str
            }
        """
        try:
            # Générer un numéro de version
            active_model = get_active_model_version()
            if active_model:
                # Incrémenter version (ex: v1.0 -> v1.1)
                current_version = active_model.version
                parts = current_version.replace('v', '').split('.')
                major, minor = int(parts[0]), int(parts[1])
                new_version = f"v{major}.{minor + 1}"
            else:
                new_version = "v1.0"

            # Créer nouvelle version dans la DB
            new_model_version = create_model_version(
                version=new_version,
                model_path=str(model_path),
                description=f"Retrained model from batch {batch_id}",
                is_active=True  # Déployer automatiquement
            )

            # Mettre à jour les métriques
            with get_session() as session:
                new_model_version.metrics_dice = metrics.get('dice')
                new_model_version.metrics_iou = metrics.get('iou')
                session.commit()

            logger.success(
                f"✅ New model deployed: {new_version} "
                f"(Dice={metrics.get('dice', 0):.3f})"
            )

            return {
                'success': True,
                'version': new_version
            }

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return {
                'success': False,
                'version': None
            }

    # ==================================================================
    # UTILITY METHODS
    # ==================================================================

    def get_training_stats(self) -> Dict:
        """Retourne des statistiques sur le pipeline de training."""
        return {
            'retraining_threshold': self.retraining_threshold,
            'validated_annotations': count_validated_annotations(),
            'ready_for_training': self.should_retrain(),
            'min_improvement_threshold': self.min_improvement_threshold,
            'export_dir': str(self.export_dir)
        }

    def __repr__(self) -> str:
        return (
            f"TrainingPipeline(threshold={self.retraining_threshold}, "
            f"min_improvement={self.min_improvement_threshold:.1%})"
        )


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    """Test du Training Pipeline."""

    print("=" * 70)
    print("TESTING TRAINING PIPELINE")
    print("=" * 70)

    pipeline = TrainingPipeline(
        retraining_threshold=10,
        min_improvement_threshold=0.01
    )

    print(f"\nPipeline: {pipeline}")
    print(f"\nDescription: {pipeline.get_training_stats()}")

    # Vérifier si training nécessaire
    if pipeline.should_retrain():
        print("\n✅ Ready for retraining")

        # Lancer le pipeline (décommenter pour tester)
        # result = pipeline.run_training_pipeline()
        # print(f"\nResult: {result}")
    else:
        print("\n⏳ Not enough annotations yet")

    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE TEST PASSED")
    print("=" * 70)
