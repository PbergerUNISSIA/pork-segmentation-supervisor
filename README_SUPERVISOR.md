# 🐷 Pork Segmentation Supervisor

**Système de supervision ML avec Active Learning pour segmentation de carcasses de porc (mesure TMP)**

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Technologies](#-technologies)
- [Structure du projet](#-structure-du-projet)

---

## 🎯 Vue d'ensemble

Ce projet est un **système de supervision intelligent** pour le déploiement en production d'un modèle U-Net de segmentation d'images de carcasses de porc.

**Objectif** : Mesurer automatiquement le TMP (Taux de Muscle de la Porcine) à partir de l'extraction des segments G et V sur les images Meater 4.

**Innovation** : Intégration d'**Active Learning** pour améliorer continuellement le modèle en production grâce aux corrections des opérateurs.

---

## ✨ Fonctionnalités

### MVP (Version 1.0)

- ✅ **Interface web Streamlit** pour upload et annotation d'images
- ✅ **Inférence TensorFlow Lite INT8** (modèle optimisé, Dice ~0.88)
- ✅ **Calcul de confiance** (entropie, variance) pour détection d'incertitude
- ✅ **Validation humaine** avec correction interactive des masques
- ✅ **Base de données SQLite** pour stockage annotations et métriques
- ✅ **Active Learning** : sélection automatique des images à labelliser
- ✅ **Re-training automatique** après N images validées
- ✅ **Logging avancé** avec Loguru (rotation, rétention)
- ✅ **Configuration typée** avec Pydantic

### Roadmap future

- 🔄 **Drift detection** (distribution shift, feature drift)
- 📊 **Dashboard métriques** (Dice, IoU, temps inférence)
- 🔔 **Alertes** (performance dégradée, anomalies)
- 🐳 **Dockerisation** pour déploiement
- 🌐 **API REST** pour intégration externe

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                         │
│  Upload images → Prédiction → Validation → Re-training     │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐      ┌────────────┐    ┌─────────────┐
  │  Model   │      │  Active    │    │  Database   │
  │ Manager  │      │  Learning  │    │  (SQLite)   │
  │ (TFLite) │      │  Engine    │    │             │
  └──────────┘      └────────────┘    └─────────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                    ┌──────────────┐
                    │   Logging    │
                    │   (Loguru)   │
                    └──────────────┘
```

**Workflow Active Learning** :

1. **Upload** : Opérateur upload une image de carcasse
2. **Inférence** : Modèle prédit le masque de segmentation
3. **Confiance** : Calcul d'entropie/variance pour évaluer l'incertitude
4. **Validation** :
   - Si confiance **haute** (>0.75) : Validation automatique
   - Si confiance **basse** : Demande validation humaine
5. **Correction** : Opérateur corrige le masque si nécessaire
6. **Stockage** : Annotation validée sauvegardée en DB
7. **Re-training** : Après N images (ex: 50), re-entraînement automatique
8. **Déploiement** : Nouveau modèle versioned et déployé

---

## 📦 Installation

### Prérequis

- Python 3.10+
- pip
- Git

### Étapes

1. **Cloner le repository** :

```bash
git clone <repo-url>
cd pork-segmentation-supervisor
```

2. **Créer un environnement virtuel** :

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances** :

```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement** :

```bash
cp .env.example .env  # Si fourni
# Ou éditer .env manuellement
```

5. **Placer le modèle TFLite** :

```bash
# Copier votre modèle dans data/models/
cp path/to/unet_int8.tflite data/models/
```

---

## ⚙️ Configuration

Toute la configuration se fait via le fichier **`.env`**.

### Variables principales

| Variable                      | Description                              | Valeur par défaut       |
| ----------------------------- | ---------------------------------------- | ----------------------- |
| `MODEL_PATH`                  | Chemin vers le modèle TFLite             | `./data/models/...`     |
| `MODEL_INPUT_SIZE`            | Taille d'entrée (512x512)                | `512`                   |
| `MODEL_CONFIDENCE_THRESHOLD`  | Seuil confiance validation auto          | `0.75`                  |
| `RETRAINING_THRESHOLD`        | Nb images avant re-training              | `50`                    |
| `UNCERTAINTY_METHOD`          | Méthode calcul incertitude               | `entropy`               |
| `LOG_LEVEL`                   | Niveau de logging                        | `DEBUG`                 |
| `DATABASE_URL`                | URL base de données                      | `sqlite:///./data/...`  |

Voir `.env` pour la configuration complète.

---

## 🚀 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'interface s'ouvre sur `http://localhost:8501`.

### Tester la configuration

```bash
# Tester le système de config
python config/settings.py

# Tester le système de logging
python utils/logger.py
```

### Workflow utilisateur

1. **Upload** : Glisser-déposer une image de carcasse
2. **Prédiction** : Le modèle génère automatiquement le masque
3. **Visualisation** : Overlay du masque sur l'image originale
4. **Validation** :
   - ✅ **Accepter** : Si prédiction correcte
   - ✏️ **Corriger** : Modifier le masque avec outils dessin
   - ❌ **Rejeter** : Si image invalide
5. **Métriques** : Voir Dice, IoU, confiance, temps inférence
6. **Historique** : Consulter les images traitées et métriques

---

## 🛠️ Technologies

| Catégorie              | Technologie           | Usage                                    |
| ---------------------- | --------------------- | ---------------------------------------- |
| **ML Framework**       | TensorFlow Lite       | Inférence optimisée INT8                 |
| **Web UI**             | Streamlit             | Interface web interactive                |
| **Database**           | SQLAlchemy + SQLite   | ORM et stockage                          |
| **Configuration**      | Pydantic              | Validation typée des settings            |
| **Logging**            | Loguru                | Logs avancés (rotation, couleurs)        |
| **Image Processing**   | OpenCV + Pillow       | Preprocessing et visualisation           |
| **Data Science**       | NumPy, scikit-learn   | Métriques (Dice, IoU), calculs           |
| **Visualization**      | Matplotlib, Plotly    | Graphiques et heatmaps                   |

---

## 📁 Structure du projet

```
pork-segmentation-supervisor/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration Pydantic
│
├── core/
│   ├── __init__.py
│   ├── model_manager.py         # Gestion modèle TFLite
│   ├── metrics.py               # Calcul confiance, Dice, IoU
│   └── image_processor.py       # Preprocessing images
│
├── database/
│   ├── __init__.py
│   ├── models.py                # ORM SQLAlchemy (tables)
│   └── crud.py                  # Opérations CRUD
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Logging Loguru
│   └── visualization.py         # Overlay, heatmaps
│
├── data/
│   ├── models/                  # Modèles TFLite (.tflite)
│   ├── annotations/             # Masques validés (.png)
│   ├── uploads/                 # Images uploadées
│   └── supervisor.db            # Base SQLite
│
├── logs/
│   └── supervisor.log           # Fichiers de logs
│
├── app.py                       # Application Streamlit principale
├── requirements.txt             # Dépendances Python
├── .env                         # Variables d'environnement
├── .gitignore                   # Fichiers à ignorer (Git)
└── README.md                    # Documentation (ce fichier)
```

---

## 📊 Métriques suivies

- **Dice Coefficient** : Mesure de similarité entre masque prédit et GT
- **IoU (Intersection over Union)** : Overlap entre prédiction et GT
- **Confiance** : Entropie/variance des prédictions
- **Temps d'inférence** : Performance du modèle
- **Distribution des prédictions** : Drift detection

---

## 🤝 Contribution

Ce projet est développé dans le cadre d'un partenariat **Unissia / Polytech Nantes**.

---

## 📝 Licence

**Propriétaire** : Unissia / Polytech Nantes

---

## 📞 Contact

Pour toute question ou support :
- **Email** : [votre-email]
- **Projet parent** : [AIA - Analyse d'Images en Abattoir](../README.md)

---

**Développé avec ❤️ pour l'amélioration continue de la qualité en abattoir**
