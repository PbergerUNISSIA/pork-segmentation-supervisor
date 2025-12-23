"""
====================================================================
PORK SEGMENTATION SUPERVISOR - SETUP VERIFICATION SCRIPT
====================================================================
Vérifie que l'installation et la configuration sont correctes.

USAGE:
    python check_setup.py
====================================================================
"""

import sys
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_python_version():
    """Vérifie la version de Python."""
    print("🐍 Python Version Check...")
    major, minor = sys.version_info[:2]
    print(f"   Version: {major}.{minor}")

    if major == 3 and minor >= 9:
        print("   ✅ Python version OK (>= 3.9)")
        return True
    else:
        print(f"   ❌ Python version too old. Need >= 3.9, got {major}.{minor}")
        return False


def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    print("\n📦 Dependencies Check...")

    required_packages = [
        ("streamlit", "streamlit"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("loguru", "loguru"),
        ("tflite_runtime", "tflite-runtime OR tensorflow"),
    ]

    all_ok = True
    for module_name, package_name in required_packages:
        try:
            if module_name == "tflite_runtime":
                # Try both tflite_runtime and tensorflow
                try:
                    __import__("tflite_runtime")
                    print(f"   ✅ {package_name} (tflite_runtime)")
                except ImportError:
                    __import__("tensorflow.lite")
                    print(f"   ✅ {package_name} (tensorflow)")
            else:
                __import__(module_name)
                print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} NOT INSTALLED")
            all_ok = False

    return all_ok


def check_directories():
    """Vérifie que tous les dossiers existent."""
    print("\n📁 Directories Check...")

    required_dirs = [
        "config",
        "core",
        "database",
        "utils",
        "data",
        "data/models",
        "data/uploads",
        "data/annotations",
        "logs",
        ".streamlit",
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ NOT FOUND")
            all_ok = False

    return all_ok


def check_files():
    """Vérifie que tous les fichiers critiques existent."""
    print("\n📄 Critical Files Check...")

    required_files = [
        "app.py",
        "config/settings.py",
        "config/__init__.py",
        "core/model_manager.py",
        "core/metrics.py",
        "core/image_processor.py",
        "core/__init__.py",
        "database/models.py",
        "database/crud.py",
        "database/__init__.py",
        "utils/logger.py",
        "utils/__init__.py",
        ".env",
        "requirements.txt",
    ]

    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} NOT FOUND")
            all_ok = False

    return all_ok


def check_model():
    """Vérifie que le modèle TFLite existe."""
    print("\n🤖 Model Check...")

    from config.settings import settings

    model_path = settings.MODEL_PATH

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Model found: {model_path}")
        print(f"   📊 Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"   ⚠️  Model NOT found: {model_path}")
        print(f"   💡 Place your .tflite model in: {model_path}")
        print(f"      The system will work in demo mode without it.")
        return False


def check_database():
    """Vérifie que la base de données peut être initialisée."""
    print("\n🗄️  Database Check...")

    try:
        from database.models import init_db, get_session, Image
        from config.settings import settings

        # Initialiser la DB
        init_db()
        print(f"   ✅ Database initialized: {settings.DATABASE_URL}")

        # Tester une requête
        with get_session() as session:
            count = session.query(Image).count()
            print(f"   ✅ Database query OK (Images: {count})")

        return True

    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return False


def check_imports():
    """Vérifie que tous les modules du projet peuvent être importés."""
    print("\n🔧 Module Imports Check...")

    modules_to_test = [
        ("config.settings", "settings"),
        ("core", "ModelManager"),
        ("core", "ImageProcessor"),
        ("core", "calculate_dice"),
        ("database", "Image"),
        ("database", "Prediction"),
        ("database.crud", "create_image"),
        ("utils.logger", "logger"),
    ]

    all_ok = True
    for module_name, attr_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            getattr(module, attr_name)
            print(f"   ✅ {module_name}.{attr_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{attr_name}: {e}")
            all_ok = False

    return all_ok


def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("🐷 PORK SEGMENTATION SUPERVISOR - SETUP VERIFICATION")
    print("=" * 70)

    results = {}

    # Exécuter tous les checks
    results["Python Version"] = check_python_version()
    results["Dependencies"] = check_dependencies()
    results["Directories"] = check_directories()
    results["Files"] = check_files()
    results["Imports"] = check_imports()
    results["Database"] = check_database()
    results["Model"] = check_model()

    # Résumé
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name:20s} : {status}")

    all_passed = all(results.values())

    # Note: Model is optional, so we don't fail if it's missing
    critical_checks = {k: v for k, v in results.items() if k != "Model"}
    critical_passed = all(critical_checks.values())

    print("\n" + "=" * 70)

    if critical_passed:
        print("✅ ALL CRITICAL CHECKS PASSED")
        print("\n💡 Next steps:")
        print("   1. Place your TFLite model in: data/models/unet_int8.tflite")
        print("   2. Run: streamlit run app.py")
        print("   3. Open: http://localhost:8501")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("\n💡 Fix the errors above and run this script again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
