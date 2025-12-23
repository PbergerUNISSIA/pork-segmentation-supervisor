"""
Script de restauration des fichiers manquants du projet.
Exécutez: python restore_files.py
"""
import os
from pathlib import Path

print("🔧 Restauration des fichiers du projet...")
print("=" * 60)

# Créer les dossiers s'ils n'existent pas
folders = [
    "config", "core", "utils", "database", ".streamlit",
    "data/models", "data/uploads", "data/annotations", "logs"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossier: {folder}")

print("\n" + "=" * 60)
print("✅ Structure des dossiers restaurée !")
print("\n⚠️  Les fichiers Python doivent être restaurés manuellement.")
print("   Utilisez la conversation précédente pour récupérer:")
print("   - config/settings.py")
print("   - core/model_manager.py, metrics.py, image_processor.py")
print("   - app.py")
print("   - etc.")
