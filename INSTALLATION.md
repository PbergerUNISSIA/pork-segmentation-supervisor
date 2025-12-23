# 📦 Guide d'Installation - Pork Segmentation Supervisor

Guide complet pour installer et configurer le système de supervision ML.

---

## 📋 Prérequis

### Système

- **OS** : Windows 10/11, Linux, ou macOS
- **Python** : 3.10 ou supérieur
- **RAM** : Minimum 4 GB (8 GB recommandé)
- **Espace disque** : 2 GB minimum

### Logiciels

- **Git** : Pour cloner le repository
- **pip** : Gestionnaire de packages Python
- **Virtualenv** : Pour isoler l'environnement (recommandé)

---

## 🚀 Installation Rapide

### 1. Cloner le repository

```bash
git clone <repository-url>
cd objective-blackwell
```

### 2. Créer un environnement virtuel

**Windows** :
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS** :
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

Cela installera automatiquement :
- TensorFlow 2.17.0
- Streamlit 1.39.0
- SQLAlchemy 2.0.36
- Pydantic 2.9.2
- Loguru 0.7.3
- OpenCV, NumPy, Pandas, etc.

### 4. Vérifier l'installation

```bash
python check_setup.py
```

Ce script vérifie :
- ✅ Imports des dépendances
- ✅ Configuration Pydantic
- ✅ Système de logging
- ✅ Structure des dossiers
- ✅ Variables d'environnement

### 5. Configurer l'environnement

Le fichier `.env` est déjà créé avec des valeurs par défaut. Vous pouvez le modifier selon vos besoins :

```bash
# Éditer avec votre éditeur préféré
nano .env
# ou
code .env
```

### 6. Placer votre modèle TFLite

Copiez votre modèle U-Net TFLite dans le dossier `data/models/` :

```bash
cp /path/to/your/unet_int8.tflite data/models/
```

**Important** : Vérifiez que le chemin dans `.env` correspond :
```
MODEL_PATH="./data/models/unet_int8.tflite"
```

---

## 🔧 Configuration Détaillée

### Variables d'environnement (.env)

#### Modèle ML

```env
MODEL_PATH="./data/models/unet_int8.tflite"
MODEL_VERSION="v1.0"
MODEL_INPUT_SIZE=512
MODEL_CONFIDENCE_THRESHOLD=0.75
```

#### Active Learning

```env
RETRAINING_THRESHOLD=50          # Re-training après 50 images
UNCERTAINTY_METHOD="entropy"      # entropy | variance | margin
HIGH_UNCERTAINTY_THRESHOLD=0.3    # Seuil d'incertitude
```

#### Base de données

```env
DATABASE_URL="sqlite:///./data/supervisor.db"
```

Pour PostgreSQL (production) :
```env
DATABASE_URL="postgresql://user:password@localhost:5432/pork_supervisor"
```

#### Logging

```env
LOG_LEVEL="DEBUG"               # DEBUG | INFO | WARNING | ERROR
LOG_FILE="./logs/supervisor.log"
LOG_ROTATION="10 MB"            # Rotation tous les 10 MB
LOG_RETENTION="30 days"         # Garder 30 jours
```

#### Streamlit

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS="localhost"
STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B"
```

---

## 🐳 Installation avec Docker (Optionnel)

**À venir dans une prochaine version**

```bash
docker build -t pork-supervisor .
docker run -p 8501:8501 pork-supervisor
```

---

## 🧪 Tests Post-Installation

### 1. Tester la configuration

```bash
python config/settings.py
```

**Sortie attendue** :
```
============================================================
PORK SEGMENTATION SUPERVISOR - CONFIGURATION
============================================================
App Name: Pork Segmentation Supervisor
Version: 1.0.0-MVP
Environment: development
Model Path: data\models\unet_int8.tflite
Model Input Size: 512
Retraining Threshold: 50
...
```

### 2. Tester le logging

```bash
python utils/logger.py
```

**Sortie attendue** : Logs colorés dans le terminal avec différents niveaux.

### 3. Tester l'application Streamlit

```bash
streamlit run app.py
```

**Note** : Le fichier `app.py` sera créé dans les prochaines étapes.

---

## ⚠️ Résolution de Problèmes

### Erreur : "ModuleNotFoundError: No module named 'tensorflow'"

**Solution** :
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Erreur : "FileNotFoundError: [Errno 2] No such file or directory: '.env'"

**Solution** : Créer le fichier `.env` :
```bash
touch .env
# Puis copier le contenu depuis .env.example
```

### Erreur : "Permission denied" lors de l'installation

**Solution Linux/macOS** :
```bash
sudo chown -R $USER:$USER venv/
```

**Solution Windows** : Exécuter le terminal en mode Administrateur.

### TensorFlow ne fonctionne pas sur Apple Silicon (M1/M2)

**Solution** :
```bash
# Désinstaller TensorFlow standard
pip uninstall tensorflow

# Installer tensorflow-metal pour M1/M2
pip install tensorflow-macos
pip install tensorflow-metal
```

### Streamlit ne démarre pas

**Solution** :
```bash
# Vérifier que le port 8501 n'est pas utilisé
netstat -ano | findstr :8501  # Windows
lsof -i :8501                  # Linux/macOS

# Changer le port dans .env
STREAMLIT_SERVER_PORT=8502
```

---

## 📊 Vérification de l'Installation

Si le script `check_setup.py` affiche :

```
✅ All checks passed! Setup is complete.
```

Vous êtes prêt à utiliser l'application ! 🎉

Sinon, vérifiez les étapes marquées ❌ et consultez la section Résolution de Problèmes.

---

## 📚 Prochaines Étapes

Après l'installation :

1. ✅ **Lire le README** : `README_SUPERVISOR.md`
2. ✅ **Créer les modules core** : `model_manager.py`, `metrics.py`, etc.
3. ✅ **Créer la base de données** : Schéma SQLAlchemy
4. ✅ **Développer l'interface Streamlit** : `app.py`
5. ✅ **Tester avec des images réelles** : Upload et inférence

---

## 💡 Conseils

- **Environnement virtuel** : Toujours activer le venv avant de travailler
- **Git** : Committez régulièrement vos modifications
- **Logs** : Consultez `logs/supervisor.log` en cas de problème
- **Documentation** : Commentez votre code pour faciliter la maintenance

---

## 📞 Support

En cas de problème :

1. Vérifier les logs : `logs/supervisor.log`
2. Exécuter `python check_setup.py`
3. Consulter la documentation : `README_SUPERVISOR.md`
4. Ouvrir une issue sur GitHub (si applicable)

---

**Installation réussie ? Passez au développement ! 🚀**
