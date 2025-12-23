# 🚀 Quick Start Guide - Pork Segmentation Supervisor

Guide de démarrage rapide pour lancer l'application en **5 minutes**.

---

## ⚡ Démarrage Express

### 1. Installer les dépendances

```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer packages
pip install -r requirements.txt
```

### 2. Initialiser la base de données

```bash
# Créer les tables + données de test
python init_db.py --seed
```

**Sortie attendue** :
```
============================================================
PORK SEGMENTATION SUPERVISOR - DATABASE INITIALIZATION
============================================================
Database URL: sqlite:///./data/supervisor.db
Environment: development
============================================================

Creating database tables...
✅ Database tables created successfully

Seeding test data...
✅ Test data seeded successfully

============================================================
DATABASE STATISTICS
============================================================
Total Images:              5
Total Predictions:         5
Total Annotations:         3
...
```

### 3. Placer votre modèle TFLite

```bash
# Copier votre modèle dans data/models/
cp /path/to/your/model.tflite data/models/unet_int8.tflite
```

**OU** pour tester sans modèle, l'app fonctionnera en mode "demo" (upload uniquement).

### 4. Lancer l'application

```bash
streamlit run app.py
```

**L'interface s'ouvre automatiquement sur** `http://localhost:8501` 🎉

---

## 🎯 Utilisation Rapide

### Page 🏠 Home - Upload & Prédiction

1. **Uploader une image** de carcasse (jpg, png, bmp)
2. **Cliquer sur "🔮 Lancer la Prédiction"**
3. **Voir le résultat** :
   - Masque prédit (overlay rouge)
   - Score de confiance
   - Temps d'inférence
   - Niveau d'incertitude
4. **Valider ou Rejeter** la prédiction

### Page 📋 Validation

- Voir toutes les **images en attente de validation**
- **Valider** rapidement les prédictions correctes
- **Rejeter** les images invalides

### Page 📊 Dashboard

- **KPIs** : Nombre d'images, prédictions, annotations
- **Performance** : Confiance moyenne, Dice score
- **Progress Re-training** : Suivi du seuil (ex: 35/50 annotations)

### Page 📜 Historique

- **Tableau** de toutes les images traitées
- **Filtres** par statut de validation
- **Statistiques** individuelles

---

## 🔧 Configuration

Toute la configuration se fait via **`.env`** :

```bash
# Éditer les variables
nano .env  # ou code .env
```

**Variables clés** :

```env
# Modèle
MODEL_PATH="./data/models/unet_int8.tflite"
MODEL_CONFIDENCE_THRESHOLD=0.75

# Active Learning
RETRAINING_THRESHOLD=50
UNCERTAINTY_METHOD="entropy"

# Logging
LOG_LEVEL="DEBUG"
```

Après modification, **redémarrer Streamlit** (Ctrl+C puis relancer).

---

## 📊 Workflow Complet

```
1. UPLOAD
   └─> Image uploadée et sauvegardée

2. PRÉDICTION
   └─> Modèle génère masque
   └─> Calcul confiance (entropie)

3. DÉCISION
   ├─> Confiance HAUTE (≥0.75)
   │   └─> Validation automatique possible
   └─> Confiance FAIBLE (<0.75)
       └─> Validation humaine requise

4. VALIDATION
   └─> Opérateur valide/corrige
   └─> Sauvegarde annotation en DB

5. RE-TRAINING (après N annotations)
   └─> Récupération annotations validées
   └─> Re-entraînement du modèle
   └─> Déploiement nouvelle version
```

---

## 🧪 Vérifier l'Installation

```bash
# Tester la configuration
python config/settings.py

# Tester le logging
python utils/logger.py

# Tester la base de données
python init_db.py --stats

# Vérifier toute l'installation
python check_setup.py
```

---

## ❓ Problèmes Fréquents

### "Model not found"
**Solution** : Placer un fichier `.tflite` dans `data/models/` et mettre à jour `MODEL_PATH` dans `.env`

### "No module named 'tensorflow'"
**Solution** :
```bash
pip install -r requirements.txt
```

### "Port 8501 already in use"
**Solution** : Changer le port dans `.streamlit/config.toml` ou `.env`

### "Database locked"
**Solution** : Fermer les autres connexions SQLite, ou redémarrer l'app

---

## 📚 Documentation Complète

- **README complet** : `README_SUPERVISOR.md`
- **Installation détaillée** : `INSTALLATION.md`
- **Architecture** : `README_SUPERVISOR.md#architecture`

---

## 🎉 Vous êtes prêt !

L'application tourne sur **http://localhost:8501**

**Prochaines étapes** :
1. ✅ Uploader vos premières images de carcasses
2. ✅ Valider les prédictions
3. ✅ Atteindre le seuil de re-training (50 annotations)
4. ✅ Re-entraîner le modèle avec les nouvelles annotations

---

**Bon usage ! 🐷**
