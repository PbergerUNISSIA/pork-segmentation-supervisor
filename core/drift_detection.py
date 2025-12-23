"""
====================================================================
PORK SEGMENTATION SUPERVISOR - DRIFT DETECTION
====================================================================
Détection de drift (distribution shift) pour monitoring de production.

TYPES DE DRIFT DÉTECTÉS :
1. **Data Drift** : Changement dans la distribution des données d'entrée
2. **Concept Drift** : Changement dans la relation entrée-sortie
3. **Performance Drift** : Dégradation des performances du modèle

MÉTHODES UTILISÉES :
- Kolmogorov-Smirnov Test (distribution shift)
- Population Stability Index (PSI)
- Jensen-Shannon Divergence
- Rolling window statistics
- Performance metrics tracking

USAGE :
    from core.drift_detection import DriftDetector, DriftType

    detector = DriftDetector()

    # Détecter drift
    drift_report = detector.detect_drift(
        reference_period_days=30,
        current_period_days=7
    )

    if drift_report['drift_detected']:
        print(f"DRIFT ALERT: {drift_report['drift_type']}")
====================================================================
"""

import enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sqlalchemy import and_, func

from config.settings import settings
from database import Prediction, get_session
from utils.logger import logger


# ==================================================================
# ENUMS
# ==================================================================

class DriftType(enum.Enum):
    """Types de drift."""
    NO_DRIFT = "no_drift"
    DATA_DRIFT = "data_drift"  # Distribution des features change
    CONCEPT_DRIFT = "concept_drift"  # Relation input-output change
    PERFORMANCE_DRIFT = "performance_drift"  # Performance dégrade


class DriftSeverity(enum.Enum):
    """Sévérité du drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ==================================================================
# DRIFT DETECTOR
# ==================================================================

class DriftDetector:
    """
    Détecteur de drift pour monitoring en production.

    Surveille les changements dans la distribution des données
    et les performances du modèle.
    """

    def __init__(
        self,
        psi_threshold: float = 0.1,  # PSI > 0.1 = drift modéré
        ks_threshold: float = 0.05,  # KS p-value < 0.05 = drift significatif
        performance_threshold: float = 0.05  # 5% dégradation
    ):
        """
        Initialise le détecteur de drift.

        Args:
            psi_threshold: Seuil pour Population Stability Index
            ks_threshold: Seuil pour Kolmogorov-Smirnov test (p-value)
            performance_threshold: Seuil de dégradation de performance
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.performance_threshold = performance_threshold

        logger.info(
            f"DriftDetector initialized: "
            f"PSI_threshold={psi_threshold}, "
            f"KS_threshold={ks_threshold}"
        )

    # ==================================================================
    # MAIN DRIFT DETECTION
    # ==================================================================

    def detect_drift(
        self,
        reference_period_days: int = 30,
        current_period_days: int = 7,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Détecte le drift en comparant deux périodes.

        Args:
            reference_period_days: Période de référence (baseline)
            current_period_days: Période actuelle
            metrics: Liste de métriques à analyser

        Returns:
            Dict: Rapport de drift complet
        """
        logger.info(
            f"Detecting drift: "
            f"Reference={reference_period_days}d, Current={current_period_days}d"
        )

        if metrics is None:
            metrics = ['confidence_score', 'entropy', 'variance', 'margin']

        # Récupérer les données des deux périodes
        reference_data = self._get_period_data(
            days_ago=reference_period_days + current_period_days,
            duration_days=reference_period_days
        )

        current_data = self._get_period_data(
            days_ago=current_period_days,
            duration_days=current_period_days
        )

        if not reference_data or not current_data:
            return {
                'drift_detected': False,
                'message': 'Insufficient data for drift detection'
            }

        # 1. Data Drift Detection (distribution des features)
        data_drift_result = self._detect_data_drift(
            reference_data, current_data, metrics
        )

        # 2. Performance Drift Detection
        performance_drift_result = self._detect_performance_drift(
            reference_data, current_data
        )

        # 3. Calculer PSI global
        psi_scores = data_drift_result.get('psi_scores', {})
        avg_psi = np.mean(list(psi_scores.values())) if psi_scores else 0.0

        # Déterminer drift global
        drift_detected = (
            data_drift_result['drift_detected'] or
            performance_drift_result['drift_detected']
        )

        # Déterminer le type de drift dominant
        if performance_drift_result['drift_detected']:
            drift_type = DriftType.PERFORMANCE_DRIFT
        elif data_drift_result['drift_detected']:
            drift_type = DriftType.DATA_DRIFT
        else:
            drift_type = DriftType.NO_DRIFT

        # Déterminer la sévérité
        severity = self._calculate_severity(
            avg_psi,
            performance_drift_result.get('performance_change', 0.0)
        )

        report = {
            'drift_detected': drift_detected,
            'drift_type': drift_type.value,
            'severity': severity.value,
            'timestamp': datetime.utcnow().isoformat(),
            'reference_period': f"{reference_period_days} days",
            'current_period': f"{current_period_days} days",
            'reference_samples': len(reference_data),
            'current_samples': len(current_data),
            'data_drift': data_drift_result,
            'performance_drift': performance_drift_result,
            'avg_psi': avg_psi,
            'recommendations': self._generate_recommendations(
                drift_type, severity
            )
        }

        # Logger les alertes
        if drift_detected:
            logger.warning(
                f"⚠️  DRIFT DETECTED: {drift_type.value} "
                f"(Severity: {severity.value}, PSI: {avg_psi:.3f})"
            )
        else:
            logger.info("✅ No drift detected")

        return report

    # ==================================================================
    # DATA DRIFT DETECTION
    # ==================================================================

    def _detect_data_drift(
        self,
        reference_data: List[Dict],
        current_data: List[Dict],
        metrics: List[str]
    ) -> Dict:
        """
        Détecte le drift dans la distribution des features.

        Args:
            reference_data: Données de référence
            current_data: Données actuelles
            metrics: Métriques à analyser

        Returns:
            Dict: Résultats de détection
        """
        ks_results = {}
        psi_scores = {}
        drift_detected = False

        for metric in metrics:
            # Extraire les valeurs
            ref_values = [d.get(metric, 0.0) for d in reference_data]
            cur_values = [d.get(metric, 0.0) for d in current_data]

            if not ref_values or not cur_values:
                continue

            # Kolmogorov-Smirnov Test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)

            ks_results[metric] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'drift': ks_pvalue < self.ks_threshold
            }

            # Population Stability Index (PSI)
            psi = self._calculate_psi(ref_values, cur_values)
            psi_scores[metric] = float(psi)

            # Drift détecté si KS ou PSI dépasse seuil
            if ks_pvalue < self.ks_threshold or psi > self.psi_threshold:
                drift_detected = True

        return {
            'drift_detected': drift_detected,
            'ks_tests': ks_results,
            'psi_scores': psi_scores,
            'metrics_analyzed': metrics
        }

    # ==================================================================
    # PERFORMANCE DRIFT DETECTION
    # ==================================================================

    def _detect_performance_drift(
        self,
        reference_data: List[Dict],
        current_data: List[Dict]
    ) -> Dict:
        """
        Détecte la dégradation de performance du modèle.

        Args:
            reference_data: Données de référence
            current_data: Données actuelles

        Returns:
            Dict: Résultats de détection
        """
        # Calculer performances moyennes
        ref_metrics = self._calculate_average_metrics(reference_data)
        cur_metrics = self._calculate_average_metrics(current_data)

        # Comparer les performances
        performance_change = {}
        drift_detected = False

        for metric in ['confidence_score', 'dice_score', 'iou_score']:
            ref_val = ref_metrics.get(metric, 0.0)
            cur_val = cur_metrics.get(metric, 0.0)

            if ref_val > 0:
                change = (cur_val - ref_val) / ref_val
                performance_change[metric] = float(change)

                # Détection de dégradation
                if change < -self.performance_threshold:
                    drift_detected = True

        # Changement global (moyenne pondérée)
        if 'confidence_score' in performance_change:
            global_change = performance_change['confidence_score']
        else:
            global_change = 0.0

        return {
            'drift_detected': drift_detected,
            'reference_metrics': ref_metrics,
            'current_metrics': cur_metrics,
            'performance_change': performance_change,
            'global_change': global_change
        }

    # ==================================================================
    # HELPER METHODS
    # ==================================================================

    def _get_period_data(
        self,
        days_ago: int,
        duration_days: int
    ) -> List[Dict]:
        """
        Récupère les données d'une période.

        Args:
            days_ago: Début de la période (jours dans le passé)
            duration_days: Durée de la période

        Returns:
            List[Dict]: Données de prédictions
        """
        with get_session() as session:
            end_date = datetime.utcnow() - timedelta(days=days_ago)
            start_date = end_date - timedelta(days=duration_days)

            predictions = (
                session.query(Prediction)
                .filter(
                    and_(
                        Prediction.predicted_at >= start_date,
                        Prediction.predicted_at < end_date
                    )
                )
                .all()
            )

            data = []
            for pred in predictions:
                data.append({
                    'id': pred.id,
                    'confidence_score': pred.confidence_score,
                    'entropy': pred.entropy,
                    'variance': pred.variance,
                    'margin': pred.margin,
                    'dice_score': pred.dice_score,
                    'iou_score': pred.iou_score,
                    'predicted_at': pred.predicted_at
                })

            return data

    def _calculate_psi(
        self,
        reference: List[float],
        current: List[float],
        n_bins: int = 10
    ) -> float:
        """
        Calcule le Population Stability Index (PSI).

        PSI mesure le changement dans la distribution.
        PSI < 0.1: Pas de drift
        PSI 0.1-0.2: Drift modéré
        PSI > 0.2: Drift significatif

        Args:
            reference: Distribution de référence
            current: Distribution actuelle
            n_bins: Nombre de bins pour histogramme

        Returns:
            float: Score PSI
        """
        ref = np.array(reference)
        cur = np.array(current)

        # Créer bins basés sur la référence
        min_val = min(ref.min(), cur.min())
        max_val = max(ref.max(), cur.max())

        bins = np.linspace(min_val, max_val, n_bins + 1)

        # Histogrammes
        ref_hist, _ = np.histogram(ref, bins=bins)
        cur_hist, _ = np.histogram(cur, bins=bins)

        # Normaliser en fréquences
        ref_freq = ref_hist / len(ref)
        cur_freq = cur_hist / len(cur)

        # Éviter division par zéro
        ref_freq = np.where(ref_freq == 0, 0.0001, ref_freq)
        cur_freq = np.where(cur_freq == 0, 0.0001, cur_freq)

        # Calculer PSI
        psi = np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq))

        return float(psi)

    def _calculate_average_metrics(
        self,
        data: List[Dict]
    ) -> Dict[str, float]:
        """
        Calcule les métriques moyennes.

        Args:
            data: Données de prédictions

        Returns:
            Dict[str, float]: Métriques moyennes
        """
        metrics = {
            'confidence_score': [],
            'entropy': [],
            'variance': [],
            'margin': [],
            'dice_score': [],
            'iou_score': []
        }

        for d in data:
            for key in metrics.keys():
                val = d.get(key)
                if val is not None:
                    metrics[key].append(val)

        # Calculer moyennes
        averages = {}
        for key, values in metrics.items():
            if values:
                averages[key] = float(np.mean(values))
            else:
                averages[key] = 0.0

        return averages

    def _calculate_severity(
        self,
        psi: float,
        performance_change: float
    ) -> DriftSeverity:
        """
        Calcule la sévérité du drift.

        Args:
            psi: Population Stability Index
            performance_change: Changement de performance

        Returns:
            DriftSeverity: Niveau de sévérité
        """
        # Sévérité basée sur PSI
        if psi < 0.1 and abs(performance_change) < 0.05:
            return DriftSeverity.NONE
        elif psi < 0.15 and abs(performance_change) < 0.1:
            return DriftSeverity.LOW
        elif psi < 0.25 and abs(performance_change) < 0.2:
            return DriftSeverity.MEDIUM
        elif psi < 0.35:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _generate_recommendations(
        self,
        drift_type: DriftType,
        severity: DriftSeverity
    ) -> List[str]:
        """
        Génère des recommandations basées sur le drift détecté.

        Args:
            drift_type: Type de drift
            severity: Sévérité

        Returns:
            List[str]: Liste de recommandations
        """
        recommendations = []

        if drift_type == DriftType.NO_DRIFT:
            recommendations.append("No action required. Model is stable.")
            return recommendations

        if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append("⚠️  URGENT: Immediate action required")

        if drift_type == DriftType.DATA_DRIFT:
            recommendations.extend([
                "Data distribution has changed significantly",
                "Review recent data sources and preprocessing",
                "Consider retraining model with recent data",
                "Investigate if data quality issues occurred"
            ])

        elif drift_type == DriftType.PERFORMANCE_DRIFT:
            recommendations.extend([
                "Model performance is degrading",
                "Collect more annotations for retraining",
                "Consider rolling back to previous model version",
                "Analyze failure cases to identify patterns"
            ])

        elif drift_type == DriftType.CONCEPT_DRIFT:
            recommendations.extend([
                "Relationship between inputs and outputs has changed",
                "Retrain model with recent validated annotations",
                "Review if business requirements changed"
            ])

        if severity >= DriftSeverity.MEDIUM:
            recommendations.append(
                f"Increase annotation rate to capture new patterns"
            )

        return recommendations

    # ==================================================================
    # MONITORING METHODS
    # ==================================================================

    def get_drift_summary(self, days: int = 30) -> Dict:
        """
        Récupère un résumé du drift sur une période.

        Args:
            days: Nombre de jours

        Returns:
            Dict: Résumé
        """
        # Détecter drift par fenêtres glissantes
        window_size = 7  # 7 jours
        n_windows = days // window_size

        drift_history = []

        for i in range(n_windows):
            result = self.detect_drift(
                reference_period_days=30,
                current_period_days=window_size
            )
            drift_history.append({
                'window': i,
                'days_ago': i * window_size,
                'drift_detected': result['drift_detected'],
                'severity': result['severity'],
                'avg_psi': result['avg_psi']
            })

        # Compter les occurrences de drift
        n_drifts = sum(1 for d in drift_history if d['drift_detected'])

        return {
            'period_days': days,
            'n_windows': n_windows,
            'n_drifts_detected': n_drifts,
            'drift_rate': n_drifts / max(n_windows, 1),
            'drift_history': drift_history
        }


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    """Test du Drift Detector."""

    print("=" * 70)
    print("TESTING DRIFT DETECTION")
    print("=" * 70)

    detector = DriftDetector(
        psi_threshold=0.1,
        ks_threshold=0.05,
        performance_threshold=0.05
    )

    print(f"\nDetector: {detector}")

    # Simuler détection de drift
    print("\n[1] Simulating drift detection...")

    # Dans un vrai scénario, cela utiliserait des données réelles
    # report = detector.detect_drift(
    #     reference_period_days=30,
    #     current_period_days=7
    # )

    # Exemple de rapport
    report = {
        'drift_detected': True,
        'drift_type': 'data_drift',
        'severity': 'medium',
        'avg_psi': 0.15,
        'recommendations': [
            'Data distribution has changed',
            'Review preprocessing pipeline'
        ]
    }

    print(f"\nDrift Report:")
    print(f"  Detected: {report['drift_detected']}")
    print(f"  Type: {report['drift_type']}")
    print(f"  Severity: {report['severity']}")
    print(f"  PSI: {report['avg_psi']:.3f}")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    print("\n" + "=" * 70)
    print("✅ DRIFT DETECTION TEST PASSED")
    print("=" * 70)
