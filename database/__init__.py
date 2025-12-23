"""
Database module for Pork Segmentation Supervisor.

This module provides:
- ORM models (SQLAlchemy)
- CRUD operations
- Database initialization
- Session management
"""

# Models
from .models import (
    Annotation,
    Image,
    Metric,
    ModelVersion,
    Prediction,
    UncertaintyLevel,
    ValidationStatus,
    drop_all_tables,
    get_session,
    init_db,
)

# CRUD operations
from .crud import (
    count_validated_annotations,
    create_annotation,
    create_image,
    create_model_version,
    create_prediction,
    get_active_model_version,
    get_annotations_for_training,
    get_average_confidence,
    get_average_dice,
    get_dashboard_stats,
    get_image_by_checksum,
    get_image_by_id,
    get_images_pending_validation,
    get_predictions_by_uncertainty,
    get_recent_images,
    mark_annotations_as_used,
    update_annotation_status,
    update_prediction_metrics,
)

__all__ = [
    # Models
    "Image",
    "Prediction",
    "Annotation",
    "ModelVersion",
    "Metric",
    "ValidationStatus",
    "UncertaintyLevel",
    # Database functions
    "init_db",
    "get_session",
    "drop_all_tables",
    # CRUD - Image
    "create_image",
    "get_image_by_id",
    "get_image_by_checksum",
    "get_recent_images",
    # CRUD - Prediction
    "create_prediction",
    "update_prediction_metrics",
    "get_predictions_by_uncertainty",
    # CRUD - Annotation
    "create_annotation",
    "update_annotation_status",
    "get_images_pending_validation",
    "count_validated_annotations",
    "get_annotations_for_training",
    "mark_annotations_as_used",
    # CRUD - Model Version
    "create_model_version",
    "get_active_model_version",
    # Analytics
    "get_average_confidence",
    "get_average_dice",
    "get_dashboard_stats",
]
