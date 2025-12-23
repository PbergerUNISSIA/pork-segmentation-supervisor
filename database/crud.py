"""
====================================================================
PORK SEGMENTATION SUPERVISOR - CRUD OPERATIONS
====================================================================
Opérations CRUD (Create, Read, Update, Delete) pour la base de données.

FONCTIONNALITÉS :
- Create : Créer images, prédictions, annotations
- Read : Récupérer avec filtres, pagination
- Update : Mettre à jour statuts, métriques
- Delete : Supprimer (rarement utilisé)
- Analytics : Statistiques et métriques agrégées

USAGE :
    from database.crud import (
        create_image,
        create_prediction,
        get_images_pending_validation,
        count_validated_annotations
    )

    # Créer une image
    image = create_image(
        filename="carcass.jpg",
        filepath="data/uploads/carcass.jpg",
        width=512,
        height=512
    )

    # Récupérer images en attente
    pending = get_images_pending_validation(limit=10)

====================================================================
"""

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from database.models import (
    Annotation,
    Image,
    Metric,
    ModelVersion,
    Prediction,
    UncertaintyLevel,
    ValidationStatus,
    get_session,
)
from utils.logger import logger


# ==================================================================
# IMAGE OPERATIONS
# ==================================================================

def create_image(
    filename: str,
    filepath: str,
    width: int,
    height: int,
    file_size: Optional[int] = None,
    uploaded_by: Optional[str] = None,
    checksum: Optional[str] = None,
    session: Optional[Session] = None
) -> Image:
    """
    Crée une nouvelle image dans la base de données.

    Args:
        filename: Nom du fichier
        filepath: Chemin de stockage
        width: Largeur en pixels
        height: Hauteur en pixels
        file_size: Taille en bytes
        uploaded_by: Nom de l'opérateur
        checksum: Hash MD5 pour déduplication
        session: Session DB (si None, en crée une nouvelle)

    Returns:
        Image: Instance créée
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        image = Image(
            filename=filename,
            filepath=filepath,
            width=width,
            height=height,
            file_size=file_size,
            uploaded_by=uploaded_by,
            checksum=checksum
        )

        session.add(image)
        session.commit()
        session.refresh(image)

        logger.info(f"Created image: {image.filename} (ID={image.id})")
        return image

    finally:
        if should_close:
            session.close()


def get_image_by_id(image_id: int, session: Optional[Session] = None) -> Optional[Image]:
    """
    Récupère une image par son ID.

    Args:
        image_id: ID de l'image
        session: Session DB

    Returns:
        Optional[Image]: Image ou None si introuvable
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        return session.query(Image).filter(Image.id == image_id).first()
    finally:
        if should_close:
            session.close()


def get_image_by_checksum(checksum: str, session: Optional[Session] = None) -> Optional[Image]:
    """
    Récupère une image par son checksum (détection duplicatas).

    Args:
        checksum: Hash MD5
        session: Session DB

    Returns:
        Optional[Image]: Image ou None
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        return session.query(Image).filter(Image.checksum == checksum).first()
    finally:
        if should_close:
            session.close()


def get_recent_images(limit: int = 20, session: Optional[Session] = None) -> List[Image]:
    """
    Récupère les images les plus récentes.

    Args:
        limit: Nombre max d'images
        session: Session DB

    Returns:
        List[Image]: Liste d'images
    """
    from sqlalchemy.orm import joinedload

    should_close = session is None
    if session is None:
        session = get_session()

    try:
        images = (
            session.query(Image)
            .options(joinedload(Image.predictions))
            .options(joinedload(Image.annotation))
            .order_by(desc(Image.uploaded_at))
            .limit(limit)
            .all()
        )

        # Forcer le chargement des relations avant de fermer la session
        for img in images:
            _ = img.predictions  # Accès pour forcer le chargement
            _ = img.annotation   # Accès pour forcer le chargement

        return images
    finally:
        if should_close:
            session.close()


# ==================================================================
# PREDICTION OPERATIONS
# ==================================================================

def create_prediction(
    image_id: int,
    model_version_id: int,
    mask_path: str,
    confidence_score: float,
    uncertainty_level: UncertaintyLevel,
    entropy: Optional[float] = None,
    variance: Optional[float] = None,
    margin: Optional[float] = None,
    inference_time_ms: Optional[float] = None,
    session: Optional[Session] = None
) -> Prediction:
    """
    Crée une nouvelle prédiction.

    Args:
        image_id: ID de l'image
        model_version_id: ID de la version du modèle
        mask_path: Chemin vers le masque prédit
        confidence_score: Score de confiance [0-1]
        uncertainty_level: Niveau d'incertitude (LOW, MEDIUM, HIGH)
        entropy: Entropie
        variance: Variance
        margin: Marge
        inference_time_ms: Temps d'inférence
        session: Session DB

    Returns:
        Prediction: Instance créée
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        prediction = Prediction(
            image_id=image_id,
            model_version_id=model_version_id,
            mask_path=mask_path,
            confidence_score=confidence_score,
            uncertainty_level=uncertainty_level,
            entropy=entropy,
            variance=variance,
            margin=margin,
            inference_time_ms=inference_time_ms
        )

        session.add(prediction)
        session.commit()
        session.refresh(prediction)

        logger.info(
            f"Created prediction for image {image_id}: "
            f"confidence={confidence_score:.2f}, "
            f"uncertainty={uncertainty_level.value}"
        )
        return prediction

    finally:
        if should_close:
            session.close()


def update_prediction_metrics(
    prediction_id: int,
    dice_score: Optional[float] = None,
    iou_score: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    session: Optional[Session] = None
) -> Optional[Prediction]:
    """
    Met à jour les métriques de segmentation d'une prédiction.

    Utilisé après validation pour calculer Dice/IoU avec le GT.

    Args:
        prediction_id: ID de la prédiction
        dice_score: Coefficient de Dice
        iou_score: IoU
        precision: Precision
        recall: Recall
        session: Session DB

    Returns:
        Optional[Prediction]: Prédiction mise à jour
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        prediction = session.query(Prediction).filter(Prediction.id == prediction_id).first()

        if prediction:
            if dice_score is not None:
                prediction.dice_score = dice_score
            if iou_score is not None:
                prediction.iou_score = iou_score
            if precision is not None:
                prediction.precision = precision
            if recall is not None:
                prediction.recall = recall

            session.commit()
            session.refresh(prediction)

            logger.info(f"Updated metrics for prediction {prediction_id}")

        return prediction

    finally:
        if should_close:
            session.close()


def get_predictions_by_uncertainty(
    uncertainty_level: UncertaintyLevel,
    limit: int = 50,
    session: Optional[Session] = None
) -> List[Prediction]:
    """
    Récupère les prédictions par niveau d'incertitude.

    Utile pour Active Learning : sélectionner les images incertaines.

    Args:
        uncertainty_level: LOW, MEDIUM, HIGH
        limit: Nombre max de prédictions
        session: Session DB

    Returns:
        List[Prediction]: Liste de prédictions
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        return (
            session.query(Prediction)
            .filter(Prediction.uncertainty_level == uncertainty_level)
            .order_by(desc(Prediction.predicted_at))
            .limit(limit)
            .all()
        )
    finally:
        if should_close:
            session.close()


# ==================================================================
# ANNOTATION OPERATIONS
# ==================================================================

def create_annotation(
    image_id: int,
    mask_path: str,
    validated_by: Optional[str] = None,
    validation_status: ValidationStatus = ValidationStatus.VALIDATED,
    was_corrected: bool = False,
    correction_notes: Optional[str] = None,
    validation_time_seconds: Optional[float] = None,
    session: Optional[Session] = None
) -> Annotation:
    """
    Crée une nouvelle annotation validée.

    Args:
        image_id: ID de l'image
        mask_path: Chemin vers le masque annoté
        validated_by: Nom de l'opérateur
        validation_status: Statut de validation
        was_corrected: True si masque corrigé
        correction_notes: Notes de correction
        validation_time_seconds: Temps de validation
        session: Session DB

    Returns:
        Annotation: Instance créée
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        annotation = Annotation(
            image_id=image_id,
            mask_path=mask_path,
            validated_by=validated_by,
            validation_status=validation_status,
            was_corrected=was_corrected,
            correction_notes=correction_notes,
            validation_time_seconds=validation_time_seconds
        )

        session.add(annotation)
        session.commit()
        session.refresh(annotation)

        logger.info(
            f"Created annotation for image {image_id}: "
            f"status={validation_status.value}, corrected={was_corrected}"
        )
        return annotation

    finally:
        if should_close:
            session.close()


def update_annotation_status(
    annotation_id: int,
    validation_status: ValidationStatus,
    session: Optional[Session] = None
) -> Optional[Annotation]:
    """
    Met à jour le statut de validation d'une annotation.

    Args:
        annotation_id: ID de l'annotation
        validation_status: Nouveau statut
        session: Session DB

    Returns:
        Optional[Annotation]: Annotation mise à jour
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        annotation = session.query(Annotation).filter(Annotation.id == annotation_id).first()

        if annotation:
            annotation.validation_status = validation_status
            session.commit()
            session.refresh(annotation)

            logger.info(f"Updated annotation {annotation_id} status to {validation_status.value}")

        return annotation

    finally:
        if should_close:
            session.close()


def get_images_pending_validation(limit: int = 10, session: Optional[Session] = None) -> List[Image]:
    """
    Récupère les images en attente de validation.

    Images avec prédiction MAIS sans annotation validée.

    Args:
        limit: Nombre max d'images
        session: Session DB

    Returns:
        List[Image]: Images en attente
    """
    from sqlalchemy.orm import joinedload

    should_close = session is None
    if session is None:
        session = get_session()

    try:
        # Images avec au moins une prédiction
        # ET (pas d'annotation OU annotation PENDING)
        images = (
            session.query(Image)
            .options(joinedload(Image.predictions))
            .options(joinedload(Image.annotation))
            .join(Prediction)
            .outerjoin(Annotation)
            .filter(
                and_(
                    Prediction.image_id == Image.id,
                    (
                        (Annotation.id == None) |
                        (Annotation.validation_status == ValidationStatus.PENDING)
                    )
                )
            )
            .order_by(desc(Prediction.predicted_at))
            .limit(limit)
            .all()
        )

        # Forcer le chargement des relations
        for img in images:
            _ = img.predictions
            _ = img.annotation

        return images
    finally:
        if should_close:
            session.close()


def count_validated_annotations(session: Optional[Session] = None) -> int:
    """
    Compte le nombre d'annotations validées.

    Utile pour déclencher re-training (ex: après 50 annotations).

    Args:
        session: Session DB

    Returns:
        int: Nombre d'annotations validées
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        return (
            session.query(Annotation)
            .filter(Annotation.validation_status == ValidationStatus.VALIDATED)
            .count()
        )
    finally:
        if should_close:
            session.close()


def get_annotations_for_training(
    limit: Optional[int] = None,
    exclude_used: bool = True,
    session: Optional[Session] = None
) -> List[Annotation]:
    """
    Récupère les annotations pour re-training.

    Args:
        limit: Nombre max (None = toutes)
        exclude_used: Si True, exclut celles déjà utilisées
        session: Session DB

    Returns:
        List[Annotation]: Annotations prêtes pour training
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        query = (
            session.query(Annotation)
            .filter(Annotation.validation_status == ValidationStatus.VALIDATED)
        )

        if exclude_used:
            query = query.filter(Annotation.used_for_training == False)

        query = query.order_by(Annotation.validated_at)

        if limit:
            query = query.limit(limit)

        return query.all()

    finally:
        if should_close:
            session.close()


def mark_annotations_as_used(
    annotation_ids: List[int],
    training_batch_id: int,
    session: Optional[Session] = None
) -> int:
    """
    Marque des annotations comme utilisées pour training.

    Args:
        annotation_ids: Liste des IDs d'annotations
        training_batch_id: ID du batch de training
        session: Session DB

    Returns:
        int: Nombre d'annotations marquées
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        count = (
            session.query(Annotation)
            .filter(Annotation.id.in_(annotation_ids))
            .update(
                {
                    Annotation.used_for_training: True,
                    Annotation.training_batch_id: training_batch_id
                },
                synchronize_session=False
            )
        )

        session.commit()

        logger.info(f"Marked {count} annotations as used for training (batch {training_batch_id})")
        return count

    finally:
        if should_close:
            session.close()


# ==================================================================
# MODEL VERSION OPERATIONS
# ==================================================================

def create_model_version(
    version: str,
    model_path: str,
    description: Optional[str] = None,
    is_active: bool = False,
    session: Optional[Session] = None
) -> ModelVersion:
    """
    Crée une nouvelle version de modèle.

    Args:
        version: Nom de version (ex: "v1.1")
        model_path: Chemin vers le .tflite
        description: Description des changements
        is_active: Si True, devient le modèle actif
        session: Session DB

    Returns:
        ModelVersion: Instance créée
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        # Si is_active=True, désactiver les autres versions
        if is_active:
            session.query(ModelVersion).update({ModelVersion.is_active: False})

        model_version = ModelVersion(
            version=version,
            model_path=model_path,
            description=description,
            is_active=is_active,
            deployed_at=datetime.utcnow() if is_active else None
        )

        session.add(model_version)
        session.commit()
        session.refresh(model_version)

        logger.info(f"Created model version: {version} (active={is_active})")
        return model_version

    finally:
        if should_close:
            session.close()


def get_active_model_version(session: Optional[Session] = None) -> Optional[ModelVersion]:
    """
    Récupère la version de modèle actuellement active.

    Args:
        session: Session DB

    Returns:
        Optional[ModelVersion]: Version active ou None
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        return session.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    finally:
        if should_close:
            session.close()


# ==================================================================
# ANALYTICS & METRICS
# ==================================================================

def get_average_confidence(
    days: int = 7,
    session: Optional[Session] = None
) -> float:
    """
    Calcule la confiance moyenne sur les N derniers jours.

    Args:
        days: Nombre de jours
        session: Session DB

    Returns:
        float: Confiance moyenne [0-1]
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        since = datetime.utcnow() - timedelta(days=days)

        result = (
            session.query(func.avg(Prediction.confidence_score))
            .filter(Prediction.predicted_at >= since)
            .scalar()
        )

        return float(result) if result else 0.0

    finally:
        if should_close:
            session.close()


def get_average_dice(
    days: int = 7,
    session: Optional[Session] = None
) -> float:
    """
    Calcule le Dice moyen sur les N derniers jours.

    Args:
        days: Nombre de jours
        session: Session DB

    Returns:
        float: Dice moyen [0-1]
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        since = datetime.utcnow() - timedelta(days=days)

        result = (
            session.query(func.avg(Prediction.dice_score))
            .filter(
                and_(
                    Prediction.predicted_at >= since,
                    Prediction.dice_score != None
                )
            )
            .scalar()
        )

        return float(result) if result else 0.0

    finally:
        if should_close:
            session.close()


def get_dashboard_stats(session: Optional[Session] = None) -> dict:
    """
    Récupère les statistiques pour le dashboard.

    Returns:
        dict: Dictionnaire de statistiques
    """
    should_close = session is None
    if session is None:
        session = get_session()

    try:
        stats = {
            "total_images": session.query(Image).count(),
            "total_predictions": session.query(Prediction).count(),
            "total_annotations": session.query(Annotation).count(),
            "validated_annotations": count_validated_annotations(session),
            "pending_validation": len(get_images_pending_validation(limit=1000, session=session)),
            "avg_confidence_7d": get_average_confidence(days=7, session=session),
            "avg_dice_7d": get_average_dice(days=7, session=session),
            "active_model": None
        }

        # Modèle actif
        active_model = get_active_model_version(session)
        if active_model:
            stats["active_model"] = active_model.version

        return stats

    finally:
        if should_close:
            session.close()


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

if __name__ == "__main__":
    """
    Test des opérations CRUD.
    """
    print("=" * 60)
    print("TESTING CRUD OPERATIONS")
    print("=" * 60)

    from database.models import init_db

    # Initialiser la DB
    init_db()

    # Test create operations
    print("\n1. Creating test data...")

    # Créer version de modèle
    model_v1 = create_model_version(
        version="v1.0-test",
        model_path="data/models/test.tflite",
        is_active=True
    )
    print(f"Created: {model_v1}")

    # Créer image
    image = create_image(
        filename="test.jpg",
        filepath="data/uploads/test.jpg",
        width=512,
        height=512
    )
    print(f"Created: {image}")

    # Créer prédiction
    prediction = create_prediction(
        image_id=image.id,
        model_version_id=model_v1.id,
        mask_path="data/predictions/test_mask.png",
        confidence_score=0.85,
        uncertainty_level=UncertaintyLevel.LOW,
        entropy=0.12
    )
    print(f"Created: {prediction}")

    # Créer annotation
    annotation = create_annotation(
        image_id=image.id,
        mask_path="data/annotations/test_mask.png",
        validated_by="test_user",
        was_corrected=False
    )
    print(f"Created: {annotation}")

    # Test read operations
    print("\n2. Testing read operations...")

    recent = get_recent_images(limit=5)
    print(f"Recent images: {len(recent)}")

    pending = get_images_pending_validation()
    print(f"Pending validation: {len(pending)}")

    validated_count = count_validated_annotations()
    print(f"Validated annotations: {validated_count}")

    # Test analytics
    print("\n3. Testing analytics...")

    stats = get_dashboard_stats()
    print("Dashboard stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("✅ CRUD OPERATIONS TEST PASSED")
    print("=" * 60)
