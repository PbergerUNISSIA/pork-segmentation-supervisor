# Configuration GitHub pour Pork Segmentation Supervisor

## Étape 1 : Créer un dépôt sur GitHub

1. Allez sur https://github.com
2. Cliquez sur le bouton **"New repository"** (ou le "+" en haut à droite)
3. Remplissez les informations :
   - **Repository name** : `pork-segmentation-supervisor` (ou votre choix)
   - **Description** : "Active Learning system for pork carcass segmentation with TFLite and Streamlit"
   - **Visibility** : Public ou Private (selon vos préférences)
   - **NE PAS** cocher "Add a README" (vous en avez déjà un)
   - **NE PAS** ajouter de .gitignore (vous en avez déjà un)
4. Cliquez sur **"Create repository"**

## Étape 2 : Lier votre dépôt local à GitHub

Une fois le dépôt créé sur GitHub, GitHub vous donnera des commandes. Utilisez celles-ci :

### Option A : Si le dépôt est vide (recommandé)

```bash
# Ajouter l'URL remote (remplacez <username> par votre nom d'utilisateur GitHub)
git remote add origin https://github.com/<username>/pork-segmentation-supervisor.git

# Renommer la branche principale en 'main' (convention moderne)
git branch -M main

# Pousser le code vers GitHub
git push -u origin main
```

### Option B : Commande directe si vous avez l'URL

```bash
# Exemple avec un nom d'utilisateur fictif
git remote add origin https://github.com/votre-username/pork-segmentation-supervisor.git
git branch -M main
git push -u origin main
```

## Étape 3 : Vérifier l'envoi

Après le push, allez sur la page de votre dépôt GitHub :
https://github.com/<username>/pork-segmentation-supervisor

Vous devriez voir tous vos fichiers !

## Étape 4 : Configuration de l'identité Git (optionnel - global)

Si vous voulez configurer votre identité pour TOUS vos projets Git :

```bash
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

**Actuellement configuré pour ce projet uniquement :**
- Nom : Paul
- Email : paul@pigsel-aia.local

## Commandes Git Utiles

### Voir l'état du dépôt
```bash
git status
```

### Voir l'historique des commits
```bash
git log --oneline
```

### Voir les remotes configurés
```bash
git remote -v
```

### Ajouter des modifications
```bash
git add .
git commit -m "Description des changements"
git push
```

## Structure Actuelle du Commit

**Commit initial créé :** 69ab4f2
**Fichiers versionnés :** 28 fichiers
**Lignes de code :** 5067 insertions

**Fichiers exclus (dans .gitignore) :**
- img/ (images de test - trop volumineuses)
- src/ (scripts expérimentaux)
- venv/ (environnement virtuel)
- .env (secrets)
- data/supervisor.db (base de données)
- logs/ (fichiers de log)
- .claude/ (fichiers temporaires)

## Recommandations

1. **Créez un fichier .env.example** pour montrer la structure de configuration :
   ```bash
   cp .env .env.example
   # Puis éditez .env.example pour retirer les vraies valeurs
   git add .env.example
   git commit -m "Add .env.example template"
   git push
   ```

2. **Ajoutez un badge dans le README** pour montrer le statut du projet

3. **Créez des releases** quand vous atteignez des jalons importants

## Problèmes Courants

### Erreur "Permission denied (publickey)"
Vous devez configurer une clé SSH ou utiliser HTTPS avec authentification.

**Solution HTTPS (plus simple) :**
GitHub vous demandera votre username et un Personal Access Token lors du push.

**Créer un Personal Access Token :**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Sélectionnez "repo" comme scope
4. Utilisez ce token comme mot de passe lors du push

### Erreur "fatal: remote origin already exists"
```bash
git remote remove origin
# Puis réessayez la commande git remote add
```

## Support

Pour plus d'aide : https://docs.github.com/en/get-started
