"""
====================================================================
PORK SEGMENTATION SUPERVISOR - DATABASE INITIALIZATION
====================================================================
Script d'initialisation de la base de données.

USAGE :
    python init_db.py                    # Créer les tables
    python init_db.py --seed             # Créer + données de test
    python init_db.py --reset            # DANGER: Supprimer et recréer
    python init_db.py --stats            # Afficher statistiques

FONCTIONNALITÉS :
- Création des tables SQLAlchemy
- Seed data pour tests
- Reset complet (développement uniquement)
- Affichage des statistiques

====================================================================
"""

import argparse
from pathlib import Path

from config.settings import settings
from database import (
    UncertaintyLevel,
    ValidationStatus,
    create_annotation,
    create_image,
    create_model_version,
    create_prediction,
    drop_all_tables,
    get_dashboard_stats,
    init_db,
)
from utils.logger import logger


def create_tables() -> None:
    """
    Crée toutes les tables de la base de données.
    """
    logger.info("Creating database tables...")
    init_db()
    logger.success("✅ Database tables created successfully")


def seed_data() -> None:
    """
    Ajoute des données de test dans la base de données.

    ATTENTION : Uniquement pour développement/tests.
    """
    logger.info("Seeding test data...")

    # Créer version de modèle initiale
    logger.info("Creating initial model version...")
    model_v1 = create_model_version(
        version=settings.MODEL_VERSION,
        model_path=str(settings.MODEL_PATH),
        description="Initial U-Net INT8 quantized model",
        is_active=True
    )
    logger.info(f"Created model version: {model_v1.version}")

    # Créer quelques images de test
    logger.info("Creating test images...")
    test_images = []

    for i in range(1, 6):
        image = create_image(
            filename=f"test_carcass_{i:03d}.jpg",
            filepath=f"data/uploads/test_carcass_{i:03d}.jpg",
            width=settings.MODEL_INPUT_SIZE,
            height=settings.MODEL_INPUT_SIZE,
            file_size=102400,
            uploaded_by="test_operator"
        )
        test_images.append(image)
        logger.debug(f"Created image: {image.filename}")

    # Créer des prédictions
    logger.info("Creating test predictions...")
    for i, image in enumerate(test_images):
        # Alterner entre haute et faible confiance
        if i % 2 == 0:
            confidence = 0.92
            uncertainty = UncertaintyLevel.LOW
            entropy = 0.08
        else:
            confidence = 0.55
            uncertainty = UncertaintyLevel.HIGH
            entropy = 0.45

        prediction = create_prediction(
            image_id=image.id,
            model_version_id=model_v1.id,
            mask_path=f"data/predictions/mask_{image.id:03d}.png",
            confidence_score=confidence,
            uncertainty_level=uncertainty,
            entropy=entropy,
            variance=0.02,
            margin=0.84,
            inference_time_ms=45.0 + i * 2
        )
        logger.debug(f"Created prediction for image {image.id}: confidence={confidence:.2f}")

    # Créer quelques annotations
    logger.info("Creating test annotations...")
    for i in range(3):  # Annoter seulement les 3 premières images
        annotation = create_annotation(
            image_id=test_images[i].id,
            mask_path=f"data/annotations/mask_{test_images[i].id:03d}.png",
            validated_by="test_operator",
            validation_status=ValidationStatus.VALIDATED,
            was_corrected=(i % 2 == 1),  # Alterner corrections
            validation_time_seconds=10.0 + i * 5
        )
        logger.debug(f"Created annotation for image {test_images[i].id}")

    logger.success("✅ Test data seeded successfully")


def reset_database() -> None:
    """
    DANGER: Supprime toutes les données et recrée les tables.

    Utilisé uniquement en développement.
    """
    if settings.is_production():
        logger.error("❌ Cannot reset database in production environment!")
        return

    logger.warning("⚠️  RESETTING DATABASE - All data will be lost!")

    # Supprimer toutes les tables
    drop_all_tables()
    logger.warning("Tables dropped")

    # Recréer les tables
    create_tables()
    logger.success("✅ Database reset complete")


def show_stats() -> None:
    """
    Affiche les statistiques de la base de données.
    """
    logger.info("Fetching database statistics...")

    stats = get_dashboard_stats()

    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total Images:              {stats['total_images']}")
    print(f"Total Predictions:         {stats['total_predictions']}")
    print(f"Total Annotations:         {stats['total_annotations']}")
    print(f"Validated Annotations:     {stats['validated_annotations']}")
    print(f"Pending Validation:        {stats['pending_validation']}")
    print(f"Active Model:              {stats['active_model']}")
    print(f"Avg Confidence (7d):       {stats['avg_confidence_7d']:.3f}")
    print(f"Avg Dice Score (7d):       {stats['avg_dice_7d']:.3f}")
    print("=" * 60)

    # Check si proche du threshold de re-training
    validated = stats['validated_annotations']
    threshold = settings.RETRAINING_THRESHOLD

    if validated >= threshold:
        print(f"\n🔔 READY FOR RETRAINING!")
        print(f"   {validated} annotations validated (threshold: {threshold})")
    else:
        remaining = threshold - validated
        print(f"\n📊 Progress to retraining: {validated}/{threshold}")
        print(f"   {remaining} annotations remaining")

    print()


def main():
    """
    Point d'entrée principal du script.
    """
    parser = argparse.ArgumentParser(
        description="Initialize Pork Segmentation Supervisor database"
    )

    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed database with test data"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="DANGER: Reset database (delete all data)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )

    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("PORK SEGMENTATION SUPERVISOR - DATABASE INITIALIZATION")
    print("=" * 60)
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print("=" * 60)

    # Exécuter les actions
    if args.reset:
        # Reset complet
        confirm = input("\n⚠️  Are you sure you want to RESET the database? (yes/no): ")
        if confirm.lower() == "yes":
            reset_database()

            # Proposer de seed après reset
            seed_after = input("\nSeed with test data? (yes/no): ")
            if seed_after.lower() == "yes":
                seed_data()
        else:
            logger.info("Reset cancelled")

    elif args.stats:
        # Afficher stats uniquement
        show_stats()

    else:
        # Création normale
        create_tables()

        if args.seed:
            seed_data()

        # Afficher stats finales
        print()
        show_stats()

    logger.success("✅ Database initialization complete!")


if __name__ == "__main__":
    main()
