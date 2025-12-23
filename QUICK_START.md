# 🚀 Guide de Démarrage Rapide - Active Learning

**Tout est prêt ! Voici comment lancer le système en 3 minutes.**

---

## ✅ **Étape 1 : Installation** (2 minutes)

```bash
# 1. Activer environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 2. Installer dépendances
pip install -r requirements.txt

# 3. Initialiser la base de données
python init_db.py
```

---

## ✅ **Étape 2 : Lancer l'Application** (30 secondes)

```bash
streamlit run app.py
```

➡️ L'application s'ouvre sur **http://localhost:8501**

---

## ✅ **Étape 3 : Utiliser le Système** (immédiat)

### 📸 **Page "🏠 Home"** : Upload & Prédiction
1. Uploader une image de carcasse
2. Cliquer sur "🔮 Lancer la Prédiction"
3. Voir le masque prédit avec confiance

### 📋 **Page "📋 Validation"** : Valider les prédictions
1. Voir les images en attente
2. Cliquer sur "✅ Valider" pour accepter
3. Les annotations sont sauvegardées

### 🎯 **Page "🎯 Active Learning"** : Système intelligent
1. **Sélection d'échantillons** :
   - Choisir stratégie (Hybrid recommandé)
   - Cliquer "🎯 Sélectionner Échantillons"
   - Voir les images les plus informatives

2. **Pipeline de Réentraînement** :
   - Voir la progress bar (ex: 10/50 annotations)
   - Quand le seuil est atteint : "🚀 Lancer Re-training"
   - Le système s'entraîne automatiquement !

3. **Détection de Drift** :
   - Cliquer "🔍 Détecter Drift"
   - Voir si le modèle dérive

4. **Versions de Modèles** :
   - Historique des versions
   - Modèle actif marqué 🟢

---

## 🎯 **Workflow Complet** (résumé)

```
1. Upload 50 images
   ↓
2. Prédire avec le modèle actuel
   ↓
3. Valider les prédictions (50 annotations)
   ↓
4. Aller sur page "🎯 Active Learning"
   ↓
5. Cliquer "🚀 Lancer Re-training"
   ↓
6. Nouveau modèle créé et déployé automatiquement !
   ↓
7. Répéter avec nouvelles images
```

---

## 📊 **Ce qui est déjà fait**

✅ Active Learning Engine (6 stratégies)
✅ Training Pipeline automatique
✅ Model Versioning & A/B Testing
✅ Drift Detection
✅ Interface Streamlit complète
✅ Script d'entraînement U-Net adapté
✅ Base de données SQLite
✅ Documentation complète

---

## ⚙️ **Configuration** (optionnel)

Éditer `.env` pour ajuster :

```env
# Seuil de réentraînement (nombre d'annotations)
RETRAINING_THRESHOLD=50

# Seuil de confiance pour validation auto
MODEL_CONFIDENCE_THRESHOLD=0.75

# Stratégie d'Active Learning (par défaut)
UNCERTAINTY_METHOD=entropy
```

---

## 🐛 **Dépannage**

### ❌ "ModuleNotFoundError: No module named 'cv2'"
```bash
pip install opencv-python
```

### ❌ "ModuleNotFoundError: No module named 'pydantic'"
```bash
pip install -r requirements.txt  # Installer toutes les dépendances
```

### ❌ "boundary_loss.py non trouvé"
➡️ **C'est normal !** Le script d'entraînement fonctionnera avec loss standard si `boundary_loss.py` n'est pas présent. Pour utiliser Boundary Loss :
```bash
# Copier votre fichier boundary_loss.py à la racine
cp /path/to/boundary_loss.py .
```

---

## 📚 **Documentation Détaillée**

- **Guide complet** : `ACTIVE_LEARNING_GUIDE.md` (450 lignes)
- **Implémentation technique** : `ACTIVE_LEARNING_IMPLEMENTATION.md`
- **Installation** : `INSTALLATION.md`
- **Architecture** : `README_SUPERVISOR.md`

---

## 🎉 **C'est tout !**

Le système est **100% opérationnel**. Il suffit de :

1. Lancer `streamlit run app.py`
2. Uploader des images
3. Valider les prédictions
4. Le système s'améliore automatiquement !

**Questions ?** Consultez `ACTIVE_LEARNING_GUIDE.md` pour plus de détails.

---

**Bon Active Learning ! 🚀🐷**
