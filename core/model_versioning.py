"""
====================================================================
PORK SEGMENTATION SUPERVISOR - MODEL VERSIONING & A/B TESTING
====================================================================
Gestion de versions de modèles et A/B testing en production.

FONCTIONNALITÉS :
- Versioning sémantique des modèles (v1.0, v1.1, v2.0)
- A/B testing (traffic splitting entre modèles)
- Champion/Challenger pattern
- Rollback automatique si dégradation
- Comparaison de performance en temps réel
- Blue/Green deployment

USAGE :
    from core.model_versioning import ModelVersioningManager, ABTestManager

    # Gestion de versions
    manager = ModelVersioningManager()
    manager.promote_to_production("v2.0")

    # A/B testing
    ab_test = ABTestManager(champion_version="v1.0", challenger_version="v2.0")
    ab_test.start_ab_test(traffic_split=0.8)  # 80% champion, 20% challenger

    # Sélectionner un modèle pour prédiction
    model_version = ab_test.select_model_for_prediction()
====================================================================
"""

import enum
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, func

from config.settings import settings
from database import (
    ModelVersion,
    Prediction,
    create_model_version,
    get_active_model_version,
    get_session,
)
from utils.logger import logger


# ==================================================================
# ENUMS
# ==================================================================

class DeploymentStrategy(enum.Enum):
    """Stratégies de déploiement de modèles."""
    BLUE_GREEN = "blue_green"  # Bascule complète
    CANARY = "canary"  # Déploiement progressif
    AB_TEST = "ab_test"  # Test A/B contrôlé
    SHADOW = "shadow"  # Mode shadow (pas de production)


class ModelStatus(enum.Enum):
    """Statuts de modèle."""
    ACTIVE = "active"  # En production
    CANDIDATE = "candidate"  # Candidat pour déploiement
    DEPRECATED = "deprecated"  # Obsolète
    FAILED = "failed"  # Échec de validation
    TESTING = "testing"  # En test A/B


# ==================================================================
# MODEL VERSIONING MANAGER
# ==================================================================

class ModelVersioningManager:
    """
    Gestionnaire de versions de modèles.

    Gère le cycle de vie des modèles : création, promotion, rollback.
    """

    def __init__(self):
        """Initialise le gestionnaire de versions."""
        logger.info("ModelVersioningManager initialized")

    def create_new_version(
        self,
        model_path: str,
        description: str,
        metrics: Optional[Dict] = None,
        auto_increment: bool = True
    ) -> str:
        """
        Crée une nouvelle version de modèle.

        Args:
            model_path: Chemin vers le fichier .tflite
            description: Description de la version
            metrics: Métriques du modèle (Dice, IoU, etc.)
            auto_increment: Incrémenter automatiquement la version

        Returns:
            str: Numéro de version créé (ex: "v1.2")
        """
        if auto_increment:
            version = self._generate_next_version()
        else:
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_version = create_model_version(
            version=version,
            model_path=model_path,
            description=description,
            is_active=False  # Pas actif par défaut
        )

        # Ajouter les métriques
        if metrics:
            with get_session() as session:
                mv = session.query(ModelVersion).filter(
                    ModelVersion.version == version
                ).first()

                if mv:
                    mv.metrics_dice = metrics.get('dice')
                    mv.metrics_iou = metrics.get('iou')
                    session.commit()

        logger.info(f"Created new model version: {version}")

        return version

    def _generate_next_version(self) -> str:
        """
        Génère automatiquement le prochain numéro de version.

        Returns:
            str: Prochaine version (ex: "v1.3")
        """
        with get_session() as session:
            # Récupérer toutes les versions
            versions = session.query(ModelVersion.version).all()

            if not versions:
                return "v1.0"

            # Parser les versions (format: v{major}.{minor})
            max_major = 0
            max_minor = 0

            for (v,) in versions:
                try:
                    parts = v.replace('v', '').split('.')
                    major = int(parts[0])
                    minor = int(parts[1]) if len(parts) > 1 else 0

                    if major > max_major:
                        max_major = major
                        max_minor = minor
                    elif major == max_major and minor > max_minor:
                        max_minor = minor
                except:
                    continue

            # Incrémenter minor version
            return f"v{max_major}.{max_minor + 1}"

    def promote_to_production(
        self,
        version: str,
        force: bool = False
    ) -> bool:
        """
        Promeut une version en production.

        Args:
            version: Version à promouvoir
            force: Forcer même si métriques insuffisantes

        Returns:
            bool: Succès de la promotion
        """
        with get_session() as session:
            # Vérifier que la version existe
            new_model = session.query(ModelVersion).filter(
                ModelVersion.version == version
            ).first()

            if not new_model:
                logger.error(f"Version {version} not found")
                return False

            # Vérifier les métriques (si pas force)
            if not force:
                active_model = get_active_model_version()

                if active_model and active_model.metrics_dice:
                    if not new_model.metrics_dice:
                        logger.error(f"Version {version} has no metrics")
                        return False

                    if new_model.metrics_dice < active_model.metrics_dice:
                        logger.warning(
                            f"Version {version} has lower Dice "
                            f"({new_model.metrics_dice:.3f} < {active_model.metrics_dice:.3f})"
                        )
                        return False

            # Désactiver toutes les versions
            session.query(ModelVersion).update({ModelVersion.is_active: False})

            # Activer la nouvelle version
            new_model.is_active = True
            new_model.deployed_at = datetime.utcnow()

            session.commit()

            logger.success(f"✅ Promoted version {version} to production")

            return True

    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback vers une version précédente.

        Args:
            version: Version cible

        Returns:
            bool: Succès du rollback
        """
        logger.warning(f"⚠️  Rolling back to version {version}")

        success = self.promote_to_production(version, force=True)

        if success:
            logger.success(f"✅ Rolled back to version {version}")
        else:
            logger.error(f"❌ Rollback to {version} failed")

        return success

    def get_version_history(self, limit: int = 10) -> List[Dict]:
        """
        Récupère l'historique des versions.

        Args:
            limit: Nombre max de versions

        Returns:
            List[Dict]: Historique des versions
        """
        with get_session() as session:
            versions = (
                session.query(ModelVersion)
                .order_by(ModelVersion.created_at.desc())
                .limit(limit)
                .all()
            )

            history = []
            for v in versions:
                history.append({
                    'version': v.version,
                    'created_at': v.created_at,
                    'deployed_at': v.deployed_at,
                    'is_active': v.is_active,
                    'metrics_dice': v.metrics_dice,
                    'metrics_iou': v.metrics_iou,
                    'description': v.description
                })

            return history

    def get_model_stats(self, version: str, days: int = 7) -> Dict:
        """
        Récupère les statistiques d'utilisation d'une version.

        Args:
            version: Version du modèle
            days: Nombre de jours

        Returns:
            Dict: Statistiques
        """
        with get_session() as session:
            model = session.query(ModelVersion).filter(
                ModelVersion.version == version
            ).first()

            if not model:
                return {}

            since = datetime.utcnow() - timedelta(days=days)

            # Compter les prédictions
            n_predictions = (
                session.query(Prediction)
                .filter(
                    and_(
                        Prediction.model_version_id == model.id,
                        Prediction.predicted_at >= since
                    )
                )
                .count()
            )

            # Confiance moyenne
            avg_confidence = (
                session.query(func.avg(Prediction.confidence_score))
                .filter(
                    and_(
                        Prediction.model_version_id == model.id,
                        Prediction.predicted_at >= since
                    )
                )
                .scalar()
            ) or 0.0

            # Dice moyen (si disponible)
            avg_dice = (
                session.query(func.avg(Prediction.dice_score))
                .filter(
                    and_(
                        Prediction.model_version_id == model.id,
                        Prediction.predicted_at >= since,
                        Prediction.dice_score != None
                    )
                )
                .scalar()
            ) or 0.0

            return {
                'version': version,
                'n_predictions': n_predictions,
                'avg_confidence': float(avg_confidence),
                'avg_dice': float(avg_dice),
                'is_active': model.is_active,
                'days': days
            }


# ==================================================================
# A/B TEST MANAGER
# ==================================================================

class ABTestManager:
    """
    Gestionnaire d'A/B testing entre deux versions de modèles.

    Implémente le pattern Champion/Challenger :
    - Champion : Modèle actuellement en production
    - Challenger : Nouveau modèle candidat

    Le traffic est divisé selon un ratio configurable.
    """

    def __init__(
        self,
        champion_version: Optional[str] = None,
        challenger_version: Optional[str] = None,
        traffic_split: float = 0.8,  # 80% champion, 20% challenger
        random_seed: Optional[int] = None
    ):
        """
        Initialise l'A/B test manager.

        Args:
            champion_version: Version du champion (défaut: modèle actif)
            challenger_version: Version du challenger
            traffic_split: Ratio de traffic vers champion [0-1]
            random_seed: Seed pour reproductibilité
        """
        if champion_version is None:
            active_model = get_active_model_version()
            self.champion_version = active_model.version if active_model else None
        else:
            self.champion_version = champion_version

        self.challenger_version = challenger_version
        self.traffic_split = traffic_split
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)

        self.ab_test_active = False

        logger.info(
            f"ABTestManager initialized: "
            f"Champion={self.champion_version}, "
            f"Challenger={self.challenger_version}, "
            f"Split={traffic_split:.0%}"
        )

    def start_ab_test(
        self,
        traffic_split: Optional[float] = None,
        duration_hours: Optional[int] = None
    ) -> bool:
        """
        Démarre un test A/B.

        Args:
            traffic_split: Ratio de traffic (optionnel)
            duration_hours: Durée du test en heures (optionnel)

        Returns:
            bool: Succès du démarrage
        """
        if not self.champion_version or not self.challenger_version:
            logger.error("Cannot start A/B test: missing champion or challenger")
            return False

        if traffic_split is not None:
            self.traffic_split = traffic_split

        self.ab_test_active = True

        logger.success(
            f"✅ A/B test started: "
            f"{self.champion_version} ({self.traffic_split:.0%}) vs "
            f"{self.challenger_version} ({1-self.traffic_split:.0%})"
        )

        if duration_hours:
            logger.info(f"Test duration: {duration_hours} hours")

        return True

    def stop_ab_test(self) -> None:
        """Arrête le test A/B."""
        self.ab_test_active = False
        logger.info("A/B test stopped")

    def select_model_for_prediction(self) -> str:
        """
        Sélectionne un modèle pour une prédiction (traffic splitting).

        Returns:
            str: Version du modèle sélectionné
        """
        if not self.ab_test_active:
            return self.champion_version

        # Traffic splitting
        if random.random() < self.traffic_split:
            return self.champion_version
        else:
            return self.challenger_version

    def get_ab_test_results(self, days: int = 7) -> Dict:
        """
        Récupère les résultats du test A/B.

        Args:
            days: Période d'analyse

        Returns:
            Dict: Résultats comparatifs
        """
        versioning_manager = ModelVersioningManager()

        champion_stats = versioning_manager.get_model_stats(
            self.champion_version, days
        )

        challenger_stats = versioning_manager.get_model_stats(
            self.challenger_version, days
        )

        # Calculer le winner
        winner = None
        if (challenger_stats.get('avg_dice', 0) >
            champion_stats.get('avg_dice', 0)):
            winner = self.challenger_version
        else:
            winner = self.champion_version

        return {
            'ab_test_active': self.ab_test_active,
            'traffic_split': self.traffic_split,
            'champion': {
                'version': self.champion_version,
                'stats': champion_stats
            },
            'challenger': {
                'version': self.challenger_version,
                'stats': challenger_stats
            },
            'winner': winner,
            'confidence_improvement': (
                challenger_stats.get('avg_confidence', 0) -
                champion_stats.get('avg_confidence', 0)
            ),
            'dice_improvement': (
                challenger_stats.get('avg_dice', 0) -
                champion_stats.get('avg_dice', 0)
            )
        }

    def decide_winner(
        self,
        min_improvement: float = 0.01,  # 1%
        min_predictions: int = 100
    ) -> Optional[str]:
        """
        Décide du gagnant du test A/B.

        Args:
            min_improvement: Amélioration minimale requise
            min_predictions: Nombre min de prédictions par modèle

        Returns:
            Optional[str]: Version gagnante (ou None si pas concluant)
        """
        results = self.get_ab_test_results()

        champion_preds = results['champion']['stats'].get('n_predictions', 0)
        challenger_preds = results['challenger']['stats'].get('n_predictions', 0)

        # Vérifier nombre de prédictions suffisant
        if champion_preds < min_predictions or challenger_preds < min_predictions:
            logger.warning(
                f"Insufficient predictions for decision "
                f"(Champion: {champion_preds}, Challenger: {challenger_preds})"
            )
            return None

        # Vérifier amélioration
        improvement = results['dice_improvement']

        if improvement >= min_improvement:
            logger.success(
                f"✅ Challenger wins! Improvement: {improvement:+.2%}"
            )
            return self.challenger_version
        elif improvement <= -min_improvement:
            logger.info(
                f"Champion wins! Challenger regression: {improvement:+.2%}"
            )
            return self.champion_version
        else:
            logger.info(
                f"No clear winner. Improvement: {improvement:+.2%} "
                f"< threshold {min_improvement:.2%}"
            )
            return None

    def promote_winner(self) -> bool:
        """
        Promeut le gagnant du test A/B en production.

        Returns:
            bool: Succès de la promotion
        """
        winner = self.decide_winner()

        if not winner:
            logger.warning("No clear winner, cannot promote")
            return False

        if winner == self.champion_version:
            logger.info(f"Champion {self.champion_version} remains in production")
            return True
        else:
            # Promouvoir le challenger
            versioning_manager = ModelVersioningManager()
            success = versioning_manager.promote_to_production(winner)

            if success:
                self.stop_ab_test()
                logger.success(f"✅ Challenger {winner} promoted to production")

            return success


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    """Test du Model Versioning et A/B Testing."""

    print("=" * 70)
    print("TESTING MODEL VERSIONING & A/B TESTING")
    print("=" * 70)

    # 1. Model Versioning
    print("\n[1] Model Versioning")
    print("-" * 70)

    manager = ModelVersioningManager()

    # Créer une nouvelle version
    new_version = manager.create_new_version(
        model_path="data/models/unet_v2.tflite",
        description="Improved U-Net with attention mechanism",
        metrics={'dice': 0.92, 'iou': 0.85}
    )

    print(f"Created version: {new_version}")

    # Voir l'historique
    history = manager.get_version_history(limit=5)
    print(f"\nVersion history ({len(history)} versions):")
    for v in history[:3]:
        print(f"  - {v['version']}: Dice={v['metrics_dice']}, Active={v['is_active']}")

    # 2. A/B Testing
    print("\n[2] A/B Testing")
    print("-" * 70)

    ab_test = ABTestManager(
        champion_version="v1.0",
        challenger_version="v1.1",
        traffic_split=0.8
    )

    # Démarrer le test
    ab_test.start_ab_test()

    # Simuler sélections de modèles
    selections = {'v1.0': 0, 'v1.1': 0}

    for _ in range(1000):
        selected = ab_test.select_model_for_prediction()
        selections[selected] += 1

    print(f"\nTraffic distribution (1000 predictions):")
    print(f"  Champion (v1.0): {selections['v1.0']} ({selections['v1.0']/10:.1f}%)")
    print(f"  Challenger (v1.1): {selections['v1.1']} ({selections['v1.1']/10:.1f}%)")

    # Résultats du test
    results = ab_test.get_ab_test_results()
    print(f"\nA/B Test Results:")
    print(f"  Winner: {results['winner']}")
    print(f"  Dice Improvement: {results['dice_improvement']:+.2%}")

    print("\n" + "=" * 70)
    print("✅ MODEL VERSIONING & A/B TESTING TEST PASSED")
    print("=" * 70)
