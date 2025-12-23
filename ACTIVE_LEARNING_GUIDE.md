# 🎯 Guide d'Active Learning avec Human-in-the-Loop

**Système d'amélioration continue pour segmentation de carcasses de porc**

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Architecture](#-architecture)
- [Composants](#-composants)
- [Workflow Complet](#-workflow-complet)
- [Utilisation](#-utilisation)
- [Stratégies d'Active Learning](#-stratégies-dactive-learning)
- [Pipeline de Réentraînement](#-pipeline-de-réentraînement)
- [A/B Testing](#-ab-testing)
- [Détection de Drift](#-détection-de-drift)
- [Monitoring](#-monitoring)
- [Best Practices](#-best-practices)

---

## 🎯 Vue d'ensemble

Ce système implémente un **cycle d'amélioration continue** pour un modèle de segmentation en production :

```
┌─────────────────────────────────────────────────────────────────┐
│                   CYCLE D'ACTIVE LEARNING                       │
└─────────────────────────────────────────────────────────────────┘

1. 📸 INFÉRENCE
   ↓ Prédiction sur nouvelles images

2. 📊 SCORING
   ↓ Calcul d'incertitude (entropie, variance, margin)

3. 🎯 SÉLECTION INTELLIGENTE
   ↓ Active Learning : sélectionner les images les plus informatives

4. 👤 ANNOTATION HUMAINE
   ↓ Opérateur valide/corrige les prédictions

5. 💾 STOCKAGE
   ↓ Ground truth sauvegardé en base de données

6. 🔄 RÉENTRAÎNEMENT
   ↓ Après N annotations, re-training automatique

7. ✅ VALIDATION
   ↓ Évaluation du nouveau modèle vs modèle actuel

8. 🚀 DÉPLOIEMENT
   ↓ A/B testing puis promotion si amélioration

9. 📈 MONITORING
   ↓ Détection de drift, alertes, métriques

10. ↺ RETOUR À L'ÉTAPE 1
```

---

## 🏗️ Architecture

### Modules Créés

```
core/
├── active_learning.py         # Stratégies de sélection d'échantillons
├── training_pipeline.py       # Pipeline de réentraînement automatique
├── model_versioning.py        # Gestion de versions & A/B testing
└── drift_detection.py         # Détection de drift & monitoring
```

### Intégration avec l'Architecture Existante

```
┌──────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                          │
│         Upload → Prédiction → Validation → Monitoring       │
└──────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌─────────────┐    ┌──────────────┐   ┌──────────────┐
  │   Model     │    │    Active    │   │   Database   │
  │  Manager    │◄───┤   Learning   │◄──┤   (SQLite)   │
  │  (TFLite)   │    │    Engine    │   │              │
  └─────────────┘    └──────────────┘   └──────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
  ┌─────────────┐    ┌──────────────┐   ┌──────────────┐
  │   Model     │    │   Training   │   │    Drift     │
  │ Versioning  │◄───┤   Pipeline   │◄──┤  Detection   │
  │ & A/B Test  │    │              │   │              │
  └─────────────┘    └──────────────┘   └──────────────┘
```

---

## 🔧 Composants

### 1. Active Learning Engine

**Stratégies de sélection d'échantillons**

- ✅ **Uncertainty Sampling** : Entropie, variance, margin, least confidence
- ✅ **Diversity Sampling** : K-Means clustering, random
- ✅ **Hybrid Strategy** : Combinaison incertitude + diversité
- ✅ **Core-set Selection** : Échantillonnage représentatif

### 2. Training Pipeline

**Pipeline de réentraînement automatique**

- ✅ Export automatique des annotations validées
- ✅ Préparation du dataset (train/val split)
- ✅ Déclenchement du training après N annotations
- ✅ Évaluation du nouveau modèle
- ✅ Validation avant déploiement
- ✅ Rollback automatique si régression

### 3. Model Versioning & A/B Testing

**Gestion de versions de modèles**

- ✅ Versioning sémantique (v1.0, v1.1, v2.0)
- ✅ Champion/Challenger pattern
- ✅ A/B testing avec traffic splitting
- ✅ Comparaison de performance en temps réel
- ✅ Promotion/Rollback automatique

### 4. Drift Detection

**Monitoring de production**

- ✅ Data Drift (distribution shift)
- ✅ Performance Drift (dégradation)
- ✅ Concept Drift (relation input-output)
- ✅ Kolmogorov-Smirnov Test
- ✅ Population Stability Index (PSI)
- ✅ Alertes automatiques

---

## 🔄 Workflow Complet

### Exemple d'Utilisation End-to-End

```python
"""
Exemple complet d'utilisation du système d'Active Learning.
"""

from core import (
    ActiveLearningEngine,
    SamplingStrategy,
    TrainingPipeline,
    ModelVersioningManager,
    ABTestManager,
    DriftDetector
)
from database import get_predictions_by_uncertainty, UncertaintyLevel

# ==================================================================
# 1. SÉLECTION D'ÉCHANTILLONS POUR ANNOTATION
# ==================================================================

# Initialiser l'Active Learning Engine
al_engine = ActiveLearningEngine(
    strategy=SamplingStrategy.HYBRID_UNCERTAINTY_DIVERSITY,
    diversity_weight=0.6  # 60% diversité, 40% incertitude
)

# Récupérer les prédictions récentes
predictions = [
    {
        'id': 1,
        'image_id': 1,
        'confidence_score': 0.65,
        'entropy': 0.35,
        'variance': 0.12,
        'margin': 0.30,
        'uncertainty_level': 'MEDIUM'
    },
    # ... plus de prédictions
]

# Sélectionner les 10 échantillons les plus informatifs
candidates = al_engine.select_samples_for_annotation(
    predictions=predictions,
    n_samples=10
)

print(f"📋 {len(candidates)} images sélectionnées pour annotation:")
for i, candidate in enumerate(candidates[:5], 1):
    print(f"  {i}. Image {candidate.image_id} - Score: {candidate.score:.3f}")

# ==================================================================
# 2. VÉRIFIER SI RÉENTRAÎNEMENT NÉCESSAIRE
# ==================================================================

pipeline = TrainingPipeline(
    retraining_threshold=50,  # Re-training après 50 annotations
    min_improvement_threshold=0.01  # 1% amélioration minimum
)

if pipeline.should_retrain():
    print("\n✅ Seuil de réentraînement atteint!")

    # Lancer le pipeline de training
    result = pipeline.run_training_pipeline()

    if result['success']:
        print(f"🎉 Nouveau modèle créé: {result['new_version']}")
        print(f"📊 Dice Score: {result['metrics']['dice']:.3f}")

        if result['deployed']:
            print("🚀 Modèle déployé en production!")
    else:
        print(f"❌ Erreur: {result['message']}")
else:
    print("⏳ Pas encore assez d'annotations pour réentraînement")

# ==================================================================
# 3. A/B TESTING D'UN NOUVEAU MODÈLE
# ==================================================================

# Démarrer un test A/B entre champion et challenger
ab_test = ABTestManager(
    champion_version="v1.0",
    challenger_version="v1.1",
    traffic_split=0.8  # 80% champion, 20% challenger
)

ab_test.start_ab_test(duration_hours=168)  # 7 jours

print("\n🧪 Test A/B démarré:")
print(f"  Champion: {ab_test.champion_version} (80% traffic)")
print(f"  Challenger: {ab_test.challenger_version} (20% traffic)")

# Simuler sélection de modèle pour prédiction
for i in range(5):
    selected_model = ab_test.select_model_for_prediction()
    print(f"  Prédiction {i+1}: Modèle {selected_model}")

# Après quelques jours, analyser les résultats
results = ab_test.get_ab_test_results(days=7)

print(f"\n📊 Résultats du test A/B:")
print(f"  Winner: {results['winner']}")
print(f"  Amélioration Dice: {results['dice_improvement']:+.2%}")
print(f"  Amélioration Confiance: {results['confidence_improvement']:+.2%}")

# Décider du gagnant et promouvoir
winner = ab_test.decide_winner(min_improvement=0.01)

if winner:
    ab_test.promote_winner()
    print(f"✅ {winner} promu en production!")

# ==================================================================
# 4. DÉTECTION DE DRIFT
# ==================================================================

detector = DriftDetector(
    psi_threshold=0.1,
    ks_threshold=0.05,
    performance_threshold=0.05
)

# Détecter drift (comparer 30 derniers jours vs 7 derniers jours)
drift_report = detector.detect_drift(
    reference_period_days=30,
    current_period_days=7
)

if drift_report['drift_detected']:
    print(f"\n⚠️  DRIFT DÉTECTÉ:")
    print(f"  Type: {drift_report['drift_type']}")
    print(f"  Sévérité: {drift_report['severity']}")
    print(f"  PSI: {drift_report['avg_psi']:.3f}")
    print(f"\n  Recommandations:")
    for rec in drift_report['recommendations']:
        print(f"    - {rec}")
else:
    print("\n✅ Pas de drift détecté - Modèle stable")

# ==================================================================
# 5. MONITORING CONTINU
# ==================================================================

# Statistiques de versioning
versioning_mgr = ModelVersioningManager()
history = versioning_mgr.get_version_history(limit=5)

print(f"\n📜 Historique des versions:")
for v in history:
    status = "🟢 ACTIF" if v['is_active'] else "⚪"
    print(f"  {status} {v['version']}: Dice={v['metrics_dice']:.3f}")

# Stats d'une version spécifique
stats = versioning_mgr.get_model_stats("v1.0", days=7)
print(f"\n📊 Stats v1.0 (7 derniers jours):")
print(f"  Prédictions: {stats['n_predictions']}")
print(f"  Confiance moy: {stats['avg_confidence']:.2%}")
print(f"  Dice moyen: {stats['avg_dice']:.3f}")

print("\n" + "="*70)
print("✅ WORKFLOW D'ACTIVE LEARNING COMPLET")
print("="*70)
```

---

## 🎯 Stratégies d'Active Learning

### Comparaison des Stratégies

| Stratégie | Objectif | Cas d'usage | Complexité |
|-----------|----------|-------------|------------|
| **Uncertainty Entropy** | Maximiser incertitude | Début du projet, peu de données | ⭐ |
| **Uncertainty Variance** | Zones de forte variance | Images complexes | ⭐ |
| **Uncertainty Margin** | Cas limites (frontières) | Affiner les frontières | ⭐⭐ |
| **Diversity K-Means** | Couverture de l'espace | Dataset déséquilibré | ⭐⭐⭐ |
| **Hybrid** | Équilibrer incertitude + diversité | Production (recommandé) | ⭐⭐⭐ |
| **Core-set** | Représentation optimale | Budget d'annotation limité | ⭐⭐⭐⭐ |

### Recommandations

- **Phase 1 (0-100 annotations)** : `UNCERTAINTY_ENTROPY` - Maximiser l'apprentissage initial
- **Phase 2 (100-500 annotations)** : `HYBRID_UNCERTAINTY_DIVERSITY` - Équilibrer
- **Phase 3 (500+ annotations)** : `REPRESENTATIVE_CORESET` - Optimiser la couverture

---

## 🔄 Pipeline de Réentraînement

### Configuration

```python
pipeline = TrainingPipeline(
    retraining_threshold=50,        # Nombre d'annotations
    min_improvement_threshold=0.01, # 1% amélioration minimum
    train_val_split=0.8,            # 80% train, 20% val
    export_dir=Path("./data/training_exports")
)
```

### Déclenchement Manuel

```python
result = pipeline.run_training_pipeline(force=True)
```

### Déclenchement Automatique

```python
# Dans un cron job ou scheduler
if pipeline.should_retrain():
    result = pipeline.run_training_pipeline()

    if result['success'] and result['deployed']:
        send_notification(f"New model {result['new_version']} deployed!")
```

### Intégration avec Script d'Entraînement Personnalisé

Le pipeline peut appeler votre script TensorFlow/PyTorch personnalisé :

```python
# Créer scripts/train_unet.py
# Le pipeline l'appellera automatiquement

result = pipeline.run_training_pipeline(
    training_script_path="scripts/train_unet.py"
)
```

---

## 🧪 A/B Testing

### Démarrer un Test A/B

```python
ab_test = ABTestManager(
    champion_version="v1.0",
    challenger_version="v1.1",
    traffic_split=0.9  # 90% champion, 10% challenger
)

ab_test.start_ab_test(duration_hours=168)  # 7 jours
```

### Canary Deployment (Déploiement Progressif)

```python
# Semaine 1: 5% traffic
ab_test.start_ab_test(traffic_split=0.95)

# Semaine 2: 20% traffic (si OK)
ab_test.traffic_split = 0.80

# Semaine 3: 50% traffic
ab_test.traffic_split = 0.50

# Semaine 4: 100% si gagnant
winner = ab_test.decide_winner()
if winner == "v1.1":
    ab_test.promote_winner()
```

### Rollback Automatique

```python
results = ab_test.get_ab_test_results()

# Si régression détectée
if results['dice_improvement'] < -0.05:  # -5%
    print("⚠️  Régression détectée, rollback automatique")

    versioning_mgr = ModelVersioningManager()
    versioning_mgr.rollback_to_version("v1.0")
```

---

## 📊 Détection de Drift

### Types de Drift

#### 1. Data Drift (Distribution Shift)

**Exemple** : Les images deviennent plus sombres/claires

```python
detector = DriftDetector(psi_threshold=0.1)

drift_report = detector.detect_drift(
    reference_period_days=30,
    current_period_days=7,
    metrics=['confidence_score', 'entropy']
)

if drift_report['drift_detected']:
    print(f"PSI: {drift_report['avg_psi']:.3f}")
    # PSI > 0.1 = drift modéré
    # PSI > 0.2 = drift significatif
```

#### 2. Performance Drift

**Exemple** : Le Dice Score chute progressivement

```python
performance = drift_report['performance_drift']

if performance['drift_detected']:
    print(f"Dégradation: {performance['global_change']:+.2%}")
    # Action: Augmenter le taux d'annotation
```

### Monitoring Continu

```python
# Résumé sur 30 jours
summary = detector.get_drift_summary(days=30)

print(f"Drift détecté dans {summary['drift_rate']:.0%} des périodes")

# Alertes par email/Slack si drift critique
if summary['drift_rate'] > 0.3:
    send_alert("⚠️ Drift détecté dans 30% des périodes récentes")
```

---

## 📈 Monitoring

### Dashboard Streamlit

Ajoutez une page de monitoring dans `app.py` :

```python
def page_active_learning_monitoring():
    st.title("🎯 Active Learning Monitoring")

    # 1. Statistiques d'annotation
    al_engine = ActiveLearningEngine(
        strategy=SamplingStrategy.HYBRID_UNCERTAINTY_DIVERSITY
    )

    st.metric("Stratégie Active", al_engine.strategy.value)
    st.text(al_engine.get_strategy_description())

    # 2. Pipeline de training
    pipeline = TrainingPipeline()
    stats = pipeline.get_training_stats()

    col1, col2 = st.columns(2)
    col1.metric("Annotations Validées", stats['validated_annotations'])
    col2.metric("Seuil Re-training", stats['retraining_threshold'])

    progress = stats['validated_annotations'] / stats['retraining_threshold']
    st.progress(min(progress, 1.0))

    # 3. Drift Detection
    detector = DriftDetector()
    drift_report = detector.detect_drift(
        reference_period_days=30,
        current_period_days=7
    )

    if drift_report['drift_detected']:
        st.error(f"⚠️ Drift détecté: {drift_report['drift_type']}")
    else:
        st.success("✅ Pas de drift détecté")

    # 4. Versions de modèles
    versioning_mgr = ModelVersioningManager()
    history = versioning_mgr.get_version_history(limit=10)

    st.dataframe(pd.DataFrame(history))
```

---

## ✅ Best Practices

### 1. Annotation Quality

- ✅ Valider au minimum **50-100 images** avant premier re-training
- ✅ Viser **>80% de confiance** pour validation automatique
- ✅ Marquer les images **corrigées** séparément (flag `was_corrected`)
- ✅ Plusieurs annotateurs pour consensus sur cas difficiles

### 2. Active Learning Strategy

- ✅ Commencer avec **Uncertainty Entropy** (simple, efficace)
- ✅ Passer à **Hybrid** après 100+ annotations
- ✅ Ajuster `diversity_weight` selon besoins :
  - `0.3-0.4` : Privilégier incertitude (début)
  - `0.5-0.6` : Équilibré (production)
  - `0.7-0.8` : Privilégier diversité (dataset déséquilibré)

### 3. Re-training

- ✅ **Seuil minimal** : 50 annotations (dépend de la complexité)
- ✅ **Amélioration minimale** : 1% (0.01) pour déployer nouveau modèle
- ✅ **Validation stricte** : Tester sur holdout set avant déploiement
- ✅ **Backup** : Sauvegarder toutes les versions de modèles

### 4. A/B Testing

- ✅ **Durée minimale** : 7 jours pour avoir des statistiques significatives
- ✅ **Traffic minimal** : Au moins 100 prédictions par modèle
- ✅ **Canary deployment** : Commencer avec 5-10% de traffic challenger
- ✅ **Rollback automatique** : Si régression > 5%

### 5. Drift Detection

- ✅ **Fréquence** : Vérifier drift toutes les semaines
- ✅ **PSI Seuils** :
  - < 0.1 : Stable
  - 0.1-0.2 : Drift modéré (surveiller)
  - \> 0.2 : Drift significatif (action requise)
- ✅ **Alertes** : Configurer notifications pour drift HIGH/CRITICAL
- ✅ **Action** : Si drift détecté, augmenter taux d'annotation

### 6. Production

- ✅ **Logging** : Logger toutes les prédictions avec métriques
- ✅ **Métriques** : Tracer Dice, IoU, Confiance au fil du temps
- ✅ **Alertes** : Configurer PagerDuty/Slack pour erreurs critiques
- ✅ **Backup** : Sauvegarder DB régulièrement
- ✅ **Audit** : Tracker qui a validé quoi et quand

---

## 🚀 Prochaines Étapes

### Évolution vers Architecture Microservices

Ce MVP peut évoluer vers l'architecture complète décrite dans votre diagramme initial :

1. **Conteneurisation Docker**
   ```dockerfile
   # Créer Dockerfile pour chaque service
   - inference-service (FastAPI + TFLite)
   - annotation-service (FastAPI)
   - training-service (Celery workers)
   - monitoring-service (Prometheus + Grafana)
   ```

2. **API REST** (FastAPI)
   ```python
   # api/main.py
   @app.post("/predict")
   @app.post("/annotate")
   @app.post("/retrain")
   @app.get("/drift-report")
   ```

3. **Orchestration Kubernetes**
   ```yaml
   # k8s/deployment.yaml
   - Pods auto-scalables
   - Load balancing
   - Health checks
   ```

4. **Monitoring Production**
   ```
   - Prometheus (métriques)
   - Grafana (dashboards)
   - ELK Stack (logs)
   - Jaeger (tracing)
   ```

---

## 📞 Support

Pour toute question :
- 📧 Email : [support@unissia.fr]
- 📚 Documentation : Voir `README_SUPERVISOR.md`
- 🐛 Issues : GitHub Issues

---

**Développé avec ❤️ pour l'amélioration continue de la qualité en abattoir**
