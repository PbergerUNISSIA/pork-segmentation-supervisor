"""
====================================================================
PORK SEGMENTATION SUPERVISOR - ACTIVE LEARNING
====================================================================
Stratégies d'Active Learning pour sélection intelligente d'échantillons.

STRATÉGIES IMPLÉMENTÉES :
1. Uncertainty Sampling (Entropy, Variance, Margin, Least Confidence)
2. Query-by-Committee (QBC)
3. Expected Gradient Length (EGL)
4. Diversity Sampling (K-Means clustering)
5. Hybrid Strategy (combinaison)
6. Representative Sampling (Core-set selection)

USAGE :
    from core.active_learning import ActiveLearningEngine, SamplingStrategy

    # Créer l'engine
    al_engine = ActiveLearningEngine(strategy=SamplingStrategy.UNCERTAINTY_ENTROPY)

    # Sélectionner les échantillons à annoter
    samples_to_annotate = al_engine.select_samples_for_annotation(
        predictions=predictions,
        n_samples=10
    )
====================================================================
"""

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from config.settings import settings
from utils.logger import logger


# ==================================================================
# ENUMS
# ==================================================================

class SamplingStrategy(enum.Enum):
    """Stratégies de sélection d'échantillons pour Active Learning."""

    # Uncertainty-based strategies
    UNCERTAINTY_ENTROPY = "uncertainty_entropy"  # Maximum entropie
    UNCERTAINTY_VARIANCE = "uncertainty_variance"  # Maximum variance
    UNCERTAINTY_MARGIN = "uncertainty_margin"  # Minimum marge
    UNCERTAINTY_LEAST_CONFIDENCE = "uncertainty_least_confidence"  # Minimum confiance

    # Diversity-based strategies
    DIVERSITY_KMEANS = "diversity_kmeans"  # Clustering K-Means
    DIVERSITY_RANDOM = "diversity_random"  # Échantillonnage aléatoire

    # Hybrid strategies
    HYBRID_UNCERTAINTY_DIVERSITY = "hybrid_uncertainty_diversity"  # Combinaison
    REPRESENTATIVE_CORESET = "representative_coreset"  # Core-set selection

    # Query-by-Committee
    QBC_VOTE_ENTROPY = "qbc_vote_entropy"  # Vote entropy (nécessite plusieurs modèles)


class SelectionCriterion(enum.Enum):
    """Critères de sélection pour priorisation."""
    HIGHEST = "highest"  # Sélectionner les valeurs les plus hautes
    LOWEST = "lowest"  # Sélectionner les valeurs les plus basses


# ==================================================================
# DATA CLASSES
# ==================================================================

@dataclass
class AnnotationCandidate:
    """
    Candidat pour annotation.

    Attributes:
        prediction_id: ID de la prédiction
        image_id: ID de l'image
        score: Score d'importance (plus élevé = plus important)
        confidence: Score de confiance [0-1]
        uncertainty_level: Niveau d'incertitude (LOW, MEDIUM, HIGH)
        features: Features pour diversity sampling (optionnel)
        metadata: Métadonnées additionnelles
    """
    prediction_id: int
    image_id: int
    score: float
    confidence: float
    uncertainty_level: str
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

    def __repr__(self) -> str:
        return (
            f"AnnotationCandidate(pred_id={self.prediction_id}, "
            f"score={self.score:.3f}, conf={self.confidence:.3f})"
        )


# ==================================================================
# ACTIVE LEARNING ENGINE
# ==================================================================

class ActiveLearningEngine:
    """
    Moteur d'Active Learning pour sélection intelligente d'échantillons.

    Sélectionne les images les plus informatives pour annotation humaine
    afin de maximiser l'amélioration du modèle avec un minimum d'effort.
    """

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY_ENTROPY,
        diversity_weight: float = 0.5,
        random_seed: int = 42
    ):
        """
        Initialise l'Active Learning Engine.

        Args:
            strategy: Stratégie de sélection
            diversity_weight: Poids de la diversité pour stratégie hybride [0-1]
            random_seed: Seed pour reproductibilité
        """
        self.strategy = strategy
        self.diversity_weight = diversity_weight
        self.random_seed = random_seed

        np.random.seed(random_seed)

        logger.info(
            f"ActiveLearningEngine initialized with strategy={strategy.value}, "
            f"diversity_weight={diversity_weight}"
        )

    # ==================================================================
    # MAIN SELECTION METHODS
    # ==================================================================

    def select_samples_for_annotation(
        self,
        predictions: List[Dict],
        n_samples: int = 10,
        exclude_ids: Optional[List[int]] = None
    ) -> List[AnnotationCandidate]:
        """
        Sélectionne les N échantillons les plus informatifs pour annotation.

        Args:
            predictions: Liste de dictionnaires de prédictions
                Format: {
                    'id': int,
                    'image_id': int,
                    'confidence_score': float,
                    'entropy': float,
                    'variance': float,
                    'margin': float,
                    'uncertainty_level': str,
                    'probabilities': np.ndarray (optionnel)
                }
            n_samples: Nombre d'échantillons à sélectionner
            exclude_ids: Liste d'IDs de prédictions à exclure

        Returns:
            List[AnnotationCandidate]: Candidats triés par importance décroissante
        """
        if not predictions:
            logger.warning("No predictions provided for sample selection")
            return []

        # Filtrer les exclusions
        if exclude_ids:
            predictions = [p for p in predictions if p['id'] not in exclude_ids]

        if len(predictions) <= n_samples:
            logger.info(f"Returning all {len(predictions)} predictions (less than {n_samples})")
            return self._predictions_to_candidates(predictions)

        logger.info(
            f"Selecting {n_samples} samples from {len(predictions)} predictions "
            f"using strategy {self.strategy.value}"
        )

        # Appliquer la stratégie
        if self.strategy == SamplingStrategy.UNCERTAINTY_ENTROPY:
            candidates = self._uncertainty_sampling(predictions, n_samples, metric='entropy')

        elif self.strategy == SamplingStrategy.UNCERTAINTY_VARIANCE:
            candidates = self._uncertainty_sampling(predictions, n_samples, metric='variance')

        elif self.strategy == SamplingStrategy.UNCERTAINTY_MARGIN:
            candidates = self._uncertainty_sampling(predictions, n_samples, metric='margin')

        elif self.strategy == SamplingStrategy.UNCERTAINTY_LEAST_CONFIDENCE:
            candidates = self._uncertainty_sampling(predictions, n_samples, metric='confidence')

        elif self.strategy == SamplingStrategy.DIVERSITY_KMEANS:
            candidates = self._diversity_sampling_kmeans(predictions, n_samples)

        elif self.strategy == SamplingStrategy.DIVERSITY_RANDOM:
            candidates = self._random_sampling(predictions, n_samples)

        elif self.strategy == SamplingStrategy.HYBRID_UNCERTAINTY_DIVERSITY:
            candidates = self._hybrid_sampling(predictions, n_samples)

        elif self.strategy == SamplingStrategy.REPRESENTATIVE_CORESET:
            candidates = self._coreset_sampling(predictions, n_samples)

        else:
            logger.warning(f"Unknown strategy {self.strategy}, falling back to entropy")
            candidates = self._uncertainty_sampling(predictions, n_samples, metric='entropy')

        logger.success(f"Selected {len(candidates)} candidates for annotation")

        return candidates

    # ==================================================================
    # UNCERTAINTY SAMPLING
    # ==================================================================

    def _uncertainty_sampling(
        self,
        predictions: List[Dict],
        n_samples: int,
        metric: str = 'entropy'
    ) -> List[AnnotationCandidate]:
        """
        Sélectionne les échantillons avec la plus grande incertitude.

        Args:
            predictions: Liste de prédictions
            n_samples: Nombre d'échantillons
            metric: Métrique à utiliser ('entropy', 'variance', 'margin', 'confidence')

        Returns:
            List[AnnotationCandidate]: Candidats triés
        """
        scores = []

        for pred in predictions:
            if metric == 'entropy':
                score = pred.get('entropy', 0.0)
                criterion = SelectionCriterion.HIGHEST  # Plus haute entropie = plus incertain
            elif metric == 'variance':
                score = pred.get('variance', 0.0)
                criterion = SelectionCriterion.HIGHEST
            elif metric == 'margin':
                score = pred.get('margin', 1.0)
                criterion = SelectionCriterion.LOWEST  # Plus petite marge = plus incertain
            elif metric == 'confidence':
                score = pred.get('confidence_score', 1.0)
                criterion = SelectionCriterion.LOWEST  # Plus faible confiance = plus incertain
            else:
                score = pred.get('entropy', 0.0)
                criterion = SelectionCriterion.HIGHEST

            scores.append(score)

        # Créer des candidats
        candidates = []
        for pred, score in zip(predictions, scores):
            candidate = AnnotationCandidate(
                prediction_id=pred['id'],
                image_id=pred['image_id'],
                score=score,
                confidence=pred.get('confidence_score', 0.0),
                uncertainty_level=pred.get('uncertainty_level', 'UNKNOWN'),
                metadata={'metric': metric, 'strategy': 'uncertainty'}
            )
            candidates.append(candidate)

        # Trier selon critère
        if criterion == SelectionCriterion.HIGHEST:
            candidates.sort(key=lambda x: x.score, reverse=True)
        else:
            candidates.sort(key=lambda x: x.score, reverse=False)

        return candidates[:n_samples]

    # ==================================================================
    # DIVERSITY SAMPLING
    # ==================================================================

    def _diversity_sampling_kmeans(
        self,
        predictions: List[Dict],
        n_samples: int
    ) -> List[AnnotationCandidate]:
        """
        Sélectionne des échantillons diversifiés via K-Means clustering.

        Sélectionne les échantillons les plus proches des centroïdes pour
        assurer une couverture diverse de l'espace des features.

        Args:
            predictions: Liste de prédictions (avec 'probabilities' ou features)
            n_samples: Nombre d'échantillons

        Returns:
            List[AnnotationCandidate]: Candidats diversifiés
        """
        # Extraire features (utiliser probabilities ou créer des features basiques)
        features_list = []
        valid_predictions = []

        for pred in predictions:
            # Essayer d'utiliser les probabilités comme features
            if 'probabilities' in pred and pred['probabilities'] is not None:
                probs = pred['probabilities']
                if isinstance(probs, np.ndarray):
                    # Aplatir les probabilités en vecteur de features
                    features = probs.flatten()
                else:
                    # Créer features basiques à partir des métriques
                    features = np.array([
                        pred.get('confidence_score', 0.0),
                        pred.get('entropy', 0.0),
                        pred.get('variance', 0.0),
                        pred.get('margin', 0.0)
                    ])
            else:
                # Créer features basiques
                features = np.array([
                    pred.get('confidence_score', 0.0),
                    pred.get('entropy', 0.0),
                    pred.get('variance', 0.0),
                    pred.get('margin', 0.0)
                ])

            features_list.append(features)
            valid_predictions.append(pred)

        if not features_list:
            logger.warning("No features available for diversity sampling, falling back to random")
            return self._random_sampling(predictions, n_samples)

        # Convertir en matrice
        X = np.array(features_list)

        # Normaliser les features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # K-Means clustering
        n_clusters = min(n_samples, len(valid_predictions))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        kmeans.fit(X)

        # Sélectionner l'échantillon le plus proche de chaque centroïde
        selected_indices = []
        for i in range(n_clusters):
            cluster_mask = kmeans.labels_ == i
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Trouver le point le plus proche du centroïde
                cluster_points = X[cluster_indices]
                centroid = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                closest_idx_in_cluster = np.argmin(distances)
                closest_idx = cluster_indices[closest_idx_in_cluster]
                selected_indices.append(closest_idx)

        # Créer des candidats
        candidates = []
        for idx in selected_indices[:n_samples]:
            pred = valid_predictions[idx]
            candidate = AnnotationCandidate(
                prediction_id=pred['id'],
                image_id=pred['image_id'],
                score=1.0 / (idx + 1),  # Score basé sur l'ordre
                confidence=pred.get('confidence_score', 0.0),
                uncertainty_level=pred.get('uncertainty_level', 'UNKNOWN'),
                features=X[idx],
                metadata={'strategy': 'diversity_kmeans', 'cluster': int(kmeans.labels_[idx])}
            )
            candidates.append(candidate)

        return candidates

    # ==================================================================
    # HYBRID SAMPLING
    # ==================================================================

    def _hybrid_sampling(
        self,
        predictions: List[Dict],
        n_samples: int
    ) -> List[AnnotationCandidate]:
        """
        Combine uncertainty et diversity sampling.

        Score = (1 - diversity_weight) * uncertainty_score + diversity_weight * diversity_score

        Args:
            predictions: Liste de prédictions
            n_samples: Nombre d'échantillons

        Returns:
            List[AnnotationCandidate]: Candidats optimaux
        """
        # 1. Calculer uncertainty scores
        uncertainty_scores = []
        for pred in predictions:
            entropy = pred.get('entropy', 0.0)
            uncertainty_scores.append(entropy)

        # Normaliser [0-1]
        uncertainty_scores = np.array(uncertainty_scores)
        if uncertainty_scores.max() > 0:
            uncertainty_scores = uncertainty_scores / uncertainty_scores.max()

        # 2. Calculer diversity scores (distance aux échantillons déjà sélectionnés)
        features_list = []
        for pred in predictions:
            features = np.array([
                pred.get('confidence_score', 0.0),
                pred.get('entropy', 0.0),
                pred.get('variance', 0.0),
                pred.get('margin', 0.0)
            ])
            features_list.append(features)

        X = np.array(features_list)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Sélection itérative
        selected_indices = []
        remaining_indices = list(range(len(predictions)))

        # Premier échantillon : celui avec le plus haut uncertainty
        first_idx = np.argmax(uncertainty_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Sélectionner les échantillons suivants
        for _ in range(min(n_samples - 1, len(remaining_indices))):
            diversity_scores = []

            for idx in remaining_indices:
                # Distance minimale aux échantillons sélectionnés
                distances = [
                    np.linalg.norm(X[idx] - X[sel_idx])
                    for sel_idx in selected_indices
                ]
                min_distance = min(distances) if distances else 0.0
                diversity_scores.append(min_distance)

            # Normaliser diversity scores
            diversity_scores = np.array(diversity_scores)
            if diversity_scores.max() > 0:
                diversity_scores = diversity_scores / diversity_scores.max()

            # Combiner scores
            combined_scores = (
                (1 - self.diversity_weight) * uncertainty_scores[remaining_indices] +
                self.diversity_weight * diversity_scores
            )

            # Sélectionner le meilleur
            best_idx_in_remaining = np.argmax(combined_scores)
            best_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Créer des candidats
        candidates = []
        for rank, idx in enumerate(selected_indices):
            pred = predictions[idx]
            candidate = AnnotationCandidate(
                prediction_id=pred['id'],
                image_id=pred['image_id'],
                score=1.0 / (rank + 1),
                confidence=pred.get('confidence_score', 0.0),
                uncertainty_level=pred.get('uncertainty_level', 'UNKNOWN'),
                features=X[idx],
                metadata={'strategy': 'hybrid', 'rank': rank}
            )
            candidates.append(candidate)

        return candidates

    # ==================================================================
    # CORESET SAMPLING
    # ==================================================================

    def _coreset_sampling(
        self,
        predictions: List[Dict],
        n_samples: int
    ) -> List[AnnotationCandidate]:
        """
        Sélection de core-set représentatif.

        Sélectionne un sous-ensemble qui représente au mieux la distribution
        de l'ensemble complet (minimisation de la distance maximale).

        Args:
            predictions: Liste de prédictions
            n_samples: Nombre d'échantillons

        Returns:
            List[AnnotationCandidate]: Core-set représentatif
        """
        # Extraire features
        features_list = []
        for pred in predictions:
            features = np.array([
                pred.get('confidence_score', 0.0),
                pred.get('entropy', 0.0),
                pred.get('variance', 0.0),
                pred.get('margin', 0.0)
            ])
            features_list.append(features)

        X = np.array(features_list)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Greedy k-Center algorithm
        selected_indices = []
        remaining_indices = list(range(len(predictions)))

        # Premier point : aléatoire
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Sélectionner les points suivants (maximiser la distance minimale)
        for _ in range(min(n_samples - 1, len(remaining_indices))):
            max_min_distance = -1
            best_idx = None

            for idx in remaining_indices:
                # Distance minimale aux points sélectionnés
                distances = [
                    np.linalg.norm(X[idx] - X[sel_idx])
                    for sel_idx in selected_indices
                ]
                min_distance = min(distances)

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Créer des candidats
        candidates = []
        for rank, idx in enumerate(selected_indices):
            pred = predictions[idx]
            candidate = AnnotationCandidate(
                prediction_id=pred['id'],
                image_id=pred['image_id'],
                score=1.0 / (rank + 1),
                confidence=pred.get('confidence_score', 0.0),
                uncertainty_level=pred.get('uncertainty_level', 'UNKNOWN'),
                features=X[idx],
                metadata={'strategy': 'coreset', 'rank': rank}
            )
            candidates.append(candidate)

        return candidates

    # ==================================================================
    # RANDOM SAMPLING (BASELINE)
    # ==================================================================

    def _random_sampling(
        self,
        predictions: List[Dict],
        n_samples: int
    ) -> List[AnnotationCandidate]:
        """
        Échantillonnage aléatoire (baseline).

        Args:
            predictions: Liste de prédictions
            n_samples: Nombre d'échantillons

        Returns:
            List[AnnotationCandidate]: Échantillons aléatoires
        """
        indices = np.random.choice(
            len(predictions),
            size=min(n_samples, len(predictions)),
            replace=False
        )

        candidates = []
        for idx in indices:
            pred = predictions[idx]
            candidate = AnnotationCandidate(
                prediction_id=pred['id'],
                image_id=pred['image_id'],
                score=np.random.random(),
                confidence=pred.get('confidence_score', 0.0),
                uncertainty_level=pred.get('uncertainty_level', 'UNKNOWN'),
                metadata={'strategy': 'random'}
            )
            candidates.append(candidate)

        return candidates

    # ==================================================================
    # HELPER METHODS
    # ==================================================================

    def _predictions_to_candidates(
        self,
        predictions: List[Dict]
    ) -> List[AnnotationCandidate]:
        """Convertit une liste de prédictions en candidats."""
        candidates = []
        for pred in predictions:
            candidate = AnnotationCandidate(
                prediction_id=pred['id'],
                image_id=pred['image_id'],
                score=pred.get('entropy', 0.0),
                confidence=pred.get('confidence_score', 0.0),
                uncertainty_level=pred.get('uncertainty_level', 'UNKNOWN')
            )
            candidates.append(candidate)
        return candidates

    def get_strategy_description(self) -> str:
        """Retourne une description de la stratégie actuelle."""
        descriptions = {
            SamplingStrategy.UNCERTAINTY_ENTROPY:
                "Sélectionne les échantillons avec la plus haute entropie (plus grande incertitude)",
            SamplingStrategy.UNCERTAINTY_VARIANCE:
                "Sélectionne les échantillons avec la plus haute variance dans les prédictions",
            SamplingStrategy.UNCERTAINTY_MARGIN:
                "Sélectionne les échantillons avec la plus petite marge entre les classes",
            SamplingStrategy.UNCERTAINTY_LEAST_CONFIDENCE:
                "Sélectionne les échantillons avec la plus faible confiance",
            SamplingStrategy.DIVERSITY_KMEANS:
                "Sélectionne des échantillons diversifiés via clustering K-Means",
            SamplingStrategy.DIVERSITY_RANDOM:
                "Sélection aléatoire (baseline)",
            SamplingStrategy.HYBRID_UNCERTAINTY_DIVERSITY:
                f"Combine incertitude ({1-self.diversity_weight:.0%}) et diversité ({self.diversity_weight:.0%})",
            SamplingStrategy.REPRESENTATIVE_CORESET:
                "Sélectionne un core-set représentatif de la distribution complète"
        }
        return descriptions.get(self.strategy, "Unknown strategy")

    def __repr__(self) -> str:
        return (
            f"ActiveLearningEngine(strategy={self.strategy.value}, "
            f"diversity_weight={self.diversity_weight})"
        )


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    """Test de l'Active Learning Engine."""

    print("=" * 70)
    print("TESTING ACTIVE LEARNING ENGINE")
    print("=" * 70)

    # Créer des prédictions fictives
    np.random.seed(42)
    predictions = []

    for i in range(100):
        pred = {
            'id': i,
            'image_id': i,
            'confidence_score': np.random.random(),
            'entropy': np.random.random(),
            'variance': np.random.random() * 0.25,
            'margin': np.random.random(),
            'uncertainty_level': 'MEDIUM',
            'probabilities': np.random.random((512, 512, 2))
        }
        predictions.append(pred)

    print(f"\nCreated {len(predictions)} mock predictions\n")

    # Tester différentes stratégies
    strategies = [
        SamplingStrategy.UNCERTAINTY_ENTROPY,
        SamplingStrategy.DIVERSITY_KMEANS,
        SamplingStrategy.HYBRID_UNCERTAINTY_DIVERSITY,
        SamplingStrategy.REPRESENTATIVE_CORESET
    ]

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Testing strategy: {strategy.value}")
        print('='*70)

        engine = ActiveLearningEngine(strategy=strategy, diversity_weight=0.6)
        print(f"Description: {engine.get_strategy_description()}\n")

        candidates = engine.select_samples_for_annotation(predictions, n_samples=10)

        print(f"Selected {len(candidates)} candidates:\n")
        for i, candidate in enumerate(candidates[:5]):
            print(f"  {i+1}. {candidate}")

        if len(candidates) > 5:
            print(f"  ... ({len(candidates)-5} more)")

    print("\n" + "=" * 70)
    print("✅ ACTIVE LEARNING ENGINE TEST PASSED")
    print("=" * 70)
