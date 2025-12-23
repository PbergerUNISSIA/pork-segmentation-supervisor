"""
====================================================================
TEST ACTIVE LEARNING SYSTEM
====================================================================
Script de test pour vérifier le bon fonctionnement du système
d'Active Learning complet.
====================================================================
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("TESTING ACTIVE LEARNING SYSTEM")
print("=" * 70)

# ==================================================================
# TEST 1: IMPORTS
# ==================================================================

print("\n[TEST 1] Vérification des imports...")

try:
    from core.active_learning import (
        ActiveLearningEngine,
        AnnotationCandidate,
        SamplingStrategy,
    )
    print("✅ Active Learning module imported")
except Exception as e:
    print(f"❌ Active Learning import failed: {e}")
    sys.exit(1)

try:
    from core.training_pipeline import TrainingPipeline
    print("✅ Training Pipeline module imported")
except Exception as e:
    print(f"❌ Training Pipeline import failed: {e}")
    sys.exit(1)

try:
    from core.model_versioning import (
        ModelVersioningManager,
        ABTestManager,
    )
    print("✅ Model Versioning module imported")
except Exception as e:
    print(f"❌ Model Versioning import failed: {e}")
    sys.exit(1)

try:
    from core.drift_detection import DriftDetector, DriftType
    print("✅ Drift Detection module imported")
except Exception as e:
    print(f"❌ Drift Detection import failed: {e}")
    sys.exit(1)

# ==================================================================
# TEST 2: ACTIVE LEARNING ENGINE
# ==================================================================

print("\n[TEST 2] Test Active Learning Engine...")

try:
    import numpy as np

    # Créer l'engine
    al_engine = ActiveLearningEngine(
        strategy=SamplingStrategy.UNCERTAINTY_ENTROPY,
        diversity_weight=0.5
    )

    # Créer des prédictions fictives
    predictions = []
    for i in range(50):
        predictions.append({
            'id': i,
            'image_id': i,
            'confidence_score': np.random.random(),
            'entropy': np.random.random(),
            'variance': np.random.random() * 0.25,
            'margin': np.random.random(),
            'uncertainty_level': 'MEDIUM'
        })

    # Sélectionner échantillons
    candidates = al_engine.select_samples_for_annotation(
        predictions=predictions,
        n_samples=10
    )

    assert len(candidates) == 10, "Devrait retourner 10 candidats"
    assert all(isinstance(c, AnnotationCandidate) for c in candidates), \
        "Tous les candidats doivent être des AnnotationCandidate"

    print(f"✅ Active Learning Engine fonctionne: {len(candidates)} candidats sélectionnés")

    # Tester différentes stratégies
    strategies_to_test = [
        SamplingStrategy.UNCERTAINTY_VARIANCE,
        SamplingStrategy.HYBRID_UNCERTAINTY_DIVERSITY,
        SamplingStrategy.DIVERSITY_KMEANS
    ]

    for strategy in strategies_to_test:
        engine = ActiveLearningEngine(strategy=strategy)
        candidates = engine.select_samples_for_annotation(predictions, n_samples=5)
        assert len(candidates) == 5
        print(f"  ✅ Stratégie {strategy.value}: OK")

except Exception as e:
    print(f"❌ Active Learning Engine test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================================================================
# TEST 3: TRAINING PIPELINE
# ==================================================================

print("\n[TEST 3] Test Training Pipeline...")

try:
    pipeline = TrainingPipeline(
        retraining_threshold=10,
        min_improvement_threshold=0.01
    )

    # Vérifier les stats
    stats = pipeline.get_training_stats()
    assert 'retraining_threshold' in stats
    assert stats['retraining_threshold'] == 10

    print(f"✅ Training Pipeline initialisé: threshold={stats['retraining_threshold']}")

except Exception as e:
    print(f"❌ Training Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================================================================
# TEST 4: MODEL VERSIONING
# ==================================================================

print("\n[TEST 4] Test Model Versioning...")

try:
    versioning_mgr = ModelVersioningManager()

    # Test de génération de version
    next_version = versioning_mgr._generate_next_version()
    assert next_version.startswith('v')

    print(f"✅ Model Versioning Manager initialisé")
    print(f"  Prochaine version: {next_version}")

except Exception as e:
    print(f"❌ Model Versioning test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================================================================
# TEST 5: A/B TESTING
# ==================================================================

print("\n[TEST 5] Test A/B Testing...")

try:
    ab_test = ABTestManager(
        champion_version="v1.0",
        challenger_version="v1.1",
        traffic_split=0.8,
        random_seed=42
    )

    # Simuler sélections
    selections = {'v1.0': 0, 'v1.1': 0}
    ab_test.ab_test_active = True

    for _ in range(1000):
        selected = ab_test.select_model_for_prediction()
        selections[selected] += 1

    # Vérifier distribution (80/20 avec tolérance)
    ratio_champion = selections['v1.0'] / 1000
    assert 0.75 < ratio_champion < 0.85, \
        f"Ratio champion devrait être ~0.8, got {ratio_champion}"

    print(f"✅ A/B Testing fonctionne:")
    print(f"  Champion: {selections['v1.0']/10:.1f}%")
    print(f"  Challenger: {selections['v1.1']/10:.1f}%")

except Exception as e:
    print(f"❌ A/B Testing test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================================================================
# TEST 6: DRIFT DETECTION
# ==================================================================

print("\n[TEST 6] Test Drift Detection...")

try:
    detector = DriftDetector(
        psi_threshold=0.1,
        ks_threshold=0.05
    )

    # Test du calcul PSI
    reference = list(np.random.normal(0.5, 0.1, 100))
    current = list(np.random.normal(0.5, 0.1, 100))  # Pas de drift
    current_drift = list(np.random.normal(0.7, 0.1, 100))  # Drift

    psi_no_drift = detector._calculate_psi(reference, current)
    psi_with_drift = detector._calculate_psi(reference, current_drift)

    assert psi_with_drift > psi_no_drift, \
        "PSI devrait être plus élevé quand il y a du drift"

    print(f"✅ Drift Detection fonctionne:")
    print(f"  PSI (pas de drift): {psi_no_drift:.3f}")
    print(f"  PSI (avec drift): {psi_with_drift:.3f}")

except Exception as e:
    print(f"❌ Drift Detection test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================================================================
# RÉSUMÉ
# ==================================================================

print("\n" + "=" * 70)
print("✅ TOUS LES TESTS SONT PASSÉS")
print("=" * 70)

print("\n📊 Résumé:")
print("  ✅ Active Learning Engine : OK")
print("  ✅ Training Pipeline : OK")
print("  ✅ Model Versioning : OK")
print("  ✅ A/B Testing : OK")
print("  ✅ Drift Detection : OK")

print("\n🎉 Système d'Active Learning prêt pour la production!")

print("\n📖 Prochaines étapes:")
print("  1. Lire ACTIVE_LEARNING_GUIDE.md pour utilisation détaillée")
print("  2. Configurer les seuils dans config/settings.py")
print("  3. Lancer l'application: streamlit run app.py")
print("  4. Commencer à annoter des images")
print("  5. Le système déclenchera automatiquement le re-training")

print("\n" + "=" * 70)
