"""
====================================================================
PORK SEGMENTATION SUPERVISOR - DATABASE MODELS
====================================================================
Schéma de base de données SQLAlchemy pour le système de supervision.

TABLES :
- Image : Stockage des images uploadées
- Annotation : Masques validés par opérateur
- Prediction : Prédictions du modèle avec métriques
- ModelVersion : Versioning des modèles
- Metric : Métriques de performance globales

RELATIONS :
- Image 1→N Prediction (une image peut avoir plusieurs prédictions)
- Image 1→1 Annotation (une image a au plus une annotation validée)
- Prediction N→1 ModelVersion (plusieurs prédictions par version)

USAGE :
    from database.models import Image, Annotation, Prediction
    from database.models import init_db, get_session

    # Initialiser la DB
    init_db()

    # Créer une session
    with get_session() as session:
        image = Image(filename="test.jpg", width=512, height=512)
        session.add(image)
        session.commit()

====================================================================
"""

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

from config.settings import settings
from utils.logger import logger

# ==================================================================
# BASE DECLARATIVE
# ==================================================================

Base = declarative_base()


# ==================================================================
# ENUMS
# ==================================================================

class ValidationStatus(enum.Enum):
    """Statut de validation d'une annotation."""
    PENDING = "pending"          # En attente de validation
    VALIDATED = "validated"      # Validé par opérateur
    REJECTED = "rejected"        # Rejeté (image invalide)
    AUTO_APPROVED = "auto_approved"  # Approuvé automatiquement (haute confiance)


class UncertaintyLevel(enum.Enum):
    """Niveau d'incertitude d'une prédiction."""
    LOW = "low"          # Confiance haute (>0.75)
    MEDIUM = "medium"    # Confiance moyenne (0.5-0.75)
    HIGH = "high"        # Confiance faible (<0.5)


# ==================================================================
# MODELS
# ==================================================================

class ModelVersion(Base):
    """
    Table de versioning des modèles ML.

    Permet de tracker les différentes versions de modèle déployées
    et d'associer les prédictions à leur version.

    Attributes:
        id: ID unique
        version: Nom de version (ex: "v1.0", "v1.1")
        model_path: Chemin vers le fichier .tflite
        created_at: Date de création
        deployed_at: Date de déploiement
        is_active: Modèle actuellement en production
        description: Description des changements
        metrics_dice: Performance Dice moyenne
        metrics_iou: Performance IoU moyenne
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    model_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deployed_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    description = Column(Text, nullable=True)

    # Métriques de performance du modèle
    metrics_dice = Column(Float, nullable=True)
    metrics_iou = Column(Float, nullable=True)

    # Relations
    predictions = relationship("Prediction", back_populates="model_version")

    def __repr__(self) -> str:
        return f"<ModelVersion(version={self.version}, active={self.is_active})>"


class Image(Base):
    """
    Table des images uploadées.

    Stocke les métadonnées de chaque image de carcasse uploadée
    dans le système.

    Attributes:
        id: ID unique
        filename: Nom du fichier original
        filepath: Chemin de stockage relatif
        width: Largeur en pixels
        height: Hauteur en pixels
        file_size: Taille du fichier en bytes
        uploaded_at: Date d'upload
        uploaded_by: Utilisateur qui a uploadé (optionnel)
        checksum: Hash MD5 pour déduplication
    """
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False, unique=True)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    file_size = Column(Integer, nullable=True)  # bytes
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    uploaded_by = Column(String(100), nullable=True)  # Nom opérateur
    checksum = Column(String(32), nullable=True, index=True)  # MD5 hash

    # Relations
    predictions = relationship("Prediction", back_populates="image", cascade="all, delete-orphan")
    annotation = relationship("Annotation", back_populates="image", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Image(id={self.id}, filename={self.filename})>"


class Prediction(Base):
    """
    Table des prédictions du modèle.

    Stocke chaque prédiction effectuée par le modèle avec ses métriques
    de confiance. Une image peut avoir plusieurs prédictions (historique).

    Attributes:
        id: ID unique
        image_id: FK vers Image
        model_version_id: FK vers ModelVersion
        mask_path: Chemin vers le masque prédit (.png)
        predicted_at: Date de prédiction
        inference_time_ms: Temps d'inférence en millisecondes

        # Métriques de confiance (Active Learning)
        confidence_score: Score global de confiance [0-1]
        entropy: Entropie des prédictions
        variance: Variance des prédictions
        margin: Marge entre top-1 et top-2
        uncertainty_level: LOW | MEDIUM | HIGH

        # Métriques de segmentation (si GT disponible)
        dice_score: Coefficient de Dice
        iou_score: Intersection over Union
        precision: Precision
        recall: Recall
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False, index=True)
    mask_path = Column(String(500), nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    inference_time_ms = Column(Float, nullable=True)

    # Métriques de confiance
    confidence_score = Column(Float, nullable=False)
    entropy = Column(Float, nullable=True)
    variance = Column(Float, nullable=True)
    margin = Column(Float, nullable=True)
    uncertainty_level = Column(Enum(UncertaintyLevel), nullable=False)

    # Métriques de segmentation (calculées après validation)
    dice_score = Column(Float, nullable=True)
    iou_score = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)

    # Relations
    image = relationship("Image", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, confidence={self.confidence_score:.2f})>"


class Annotation(Base):
    """
    Table des annotations validées.

    Stocke les masques validés/corrigés par les opérateurs humains.
    Ces annotations servent de ground truth pour re-training.

    Attributes:
        id: ID unique
        image_id: FK vers Image (one-to-one)
        mask_path: Chemin vers le masque annoté (.png)
        validated_at: Date de validation
        validated_by: Opérateur ayant validé
        validation_status: PENDING | VALIDATED | REJECTED | AUTO_APPROVED
        validation_time_seconds: Temps de validation en secondes

        # Flags de correction
        was_corrected: True si l'opérateur a modifié la prédiction
        correction_notes: Notes de l'opérateur (optionnel)

        # Pour Active Learning
        used_for_training: True si utilisé pour re-training
        training_batch_id: ID du batch de training
    """
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False, unique=True, index=True)
    mask_path = Column(String(500), nullable=False)
    validated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    validated_by = Column(String(100), nullable=True)
    validation_status = Column(Enum(ValidationStatus), nullable=False, default=ValidationStatus.PENDING)
    validation_time_seconds = Column(Float, nullable=True)

    # Correction
    was_corrected = Column(Boolean, default=False, nullable=False)
    correction_notes = Column(Text, nullable=True)

    # Active Learning
    used_for_training = Column(Boolean, default=False, nullable=False)
    training_batch_id = Column(Integer, nullable=True)

    # Relations
    image = relationship("Image", back_populates="annotation")

    def __repr__(self) -> str:
        return f"<Annotation(id={self.id}, status={self.validation_status.value})>"


class Metric(Base):
    """
    Table des métriques globales du système.

    Stocke des statistiques agrégées pour monitoring et dashboards.

    Attributes:
        id: ID unique
        metric_name: Nom de la métrique (ex: "avg_dice_daily", "total_images")
        metric_value: Valeur de la métrique
        metric_type: Type de métrique (count, avg, min, max)
        category: Catégorie (performance, usage, active_learning)
        recorded_at: Date d'enregistrement
        extra_data: Données additionnelles (JSON string)
    """
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # count, avg, min, max
    category = Column(String(50), nullable=False, index=True)  # performance, usage, active_learning
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    extra_data = Column(Text, nullable=True)  # JSON string

    def __repr__(self) -> str:
        return f"<Metric(name={self.metric_name}, value={self.metric_value})>"


# ==================================================================
# DATABASE ENGINE & SESSION
# ==================================================================

# Engine global (créé une seule fois)
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,  # Set to True pour debug SQL
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# SessionMaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ==================================================================
# HELPER FUNCTIONS
# ==================================================================

def init_db() -> None:
    """
    Initialise la base de données.

    Crée toutes les tables définies dans Base.metadata.
    Idempotent : ne recrée pas les tables si elles existent déjà.

    Usage:
        from database.models import init_db
        init_db()
    """
    logger.info("Initializing database...")

    # Créer toutes les tables
    Base.metadata.create_all(bind=engine)

    logger.success(f"Database initialized at {settings.DATABASE_URL}")

    # Log des tables créées
    tables = Base.metadata.tables.keys()
    logger.info(f"Tables: {list(tables)}")


def get_session() -> Session:
    """
    Retourne une session de base de données.

    Utilise un context manager pour auto-close.

    Usage:
        with get_session() as session:
            image = session.query(Image).first()
            print(image.filename)

    Returns:
        Session: Session SQLAlchemy
    """
    session = SessionLocal()
    try:
        return session
    finally:
        pass  # Le context manager fermera la session


def drop_all_tables() -> None:
    """
    DANGER: Supprime toutes les tables de la base de données.

    Utilisé uniquement pour tests ou reset complet.
    """
    logger.warning("⚠️  Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    """
    Test de création de la base de données.
    """
    print("=" * 60)
    print("TESTING DATABASE MODELS")
    print("=" * 60)

    # Initialiser la DB
    print("\n1. Initializing database...")
    init_db()

    # Créer une session
    print("\n2. Creating test data...")
    with get_session() as session:
        # Créer une version de modèle
        model_v1 = ModelVersion(
            version="v1.0",
            model_path="data/models/unet_int8_v1.tflite",
            is_active=True,
            description="Initial U-Net model",
            metrics_dice=0.88,
            metrics_iou=0.79
        )
        session.add(model_v1)
        session.commit()
        print(f"Created: {model_v1}")

        # Créer une image
        image = Image(
            filename="carcass_001.jpg",
            filepath="data/uploads/carcass_001.jpg",
            width=512,
            height=512,
            file_size=102400,
            uploaded_by="operator_1"
        )
        session.add(image)
        session.commit()
        print(f"Created: {image}")

        # Créer une prédiction
        prediction = Prediction(
            image_id=image.id,
            model_version_id=model_v1.id,
            mask_path="data/predictions/mask_001.png",
            confidence_score=0.92,
            entropy=0.08,
            variance=0.02,
            margin=0.84,
            uncertainty_level=UncertaintyLevel.LOW,
            inference_time_ms=45.3
        )
        session.add(prediction)
        session.commit()
        print(f"Created: {prediction}")

        # Créer une annotation
        annotation = Annotation(
            image_id=image.id,
            mask_path="data/annotations/mask_001.png",
            validated_by="operator_1",
            validation_status=ValidationStatus.VALIDATED,
            was_corrected=False,
            validation_time_seconds=12.5
        )
        session.add(annotation)
        session.commit()
        print(f"Created: {annotation}")

    # Requêtes de test
    print("\n3. Testing queries...")
    with get_session() as session:
        # Compter les images
        image_count = session.query(Image).count()
        print(f"Total images: {image_count}")

        # Récupérer toutes les prédictions avec haute confiance
        high_conf_preds = session.query(Prediction).filter(
            Prediction.confidence_score > 0.9
        ).all()
        print(f"High confidence predictions: {len(high_conf_preds)}")

        # Récupérer annotations validées
        validated = session.query(Annotation).filter(
            Annotation.validation_status == ValidationStatus.VALIDATED
        ).count()
        print(f"Validated annotations: {validated}")

        # Récupérer une image avec ses relations
        img = session.query(Image).first()
        if img:
            print(f"\nImage: {img.filename}")
            print(f"  Predictions: {len(img.predictions)}")
            print(f"  Annotation: {img.annotation}")

    print("\n" + "=" * 60)
    print("✅ DATABASE MODELS TEST PASSED")
    print("=" * 60)
    print("\n💡 Database file created at:", settings.DATABASE_URL)
