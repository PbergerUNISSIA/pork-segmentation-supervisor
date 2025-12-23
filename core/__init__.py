"""
Core module for Pork Segmentation Supervisor.
"""

from .image_processor import ImageProcessor
from .metrics import (
    calculate_all_metrics,
    calculate_confidence,
    calculate_dice,
    calculate_entropy,
    calculate_iou,
    calculate_margin,
    calculate_precision_recall,
    calculate_variance,
    is_uncertain,
)
from .model_manager import ModelManager

# Active Learning
from .active_learning import (
    ActiveLearningEngine,
    AnnotationCandidate,
    SamplingStrategy,
    SelectionCriterion,
)

# Training Pipeline
from .training_pipeline import TrainingPipeline

# Model Versioning & A/B Testing
from .model_versioning import (
    ABTestManager,
    DeploymentStrategy,
    ModelStatus,
    ModelVersioningManager,
)

# Drift Detection
from .drift_detection import DriftDetector, DriftSeverity, DriftType

__all__ = [
    # Core components
    "ModelManager",
    "ImageProcessor",
    # Metrics
    "calculate_dice",
    "calculate_iou",
    "calculate_precision_recall",
    "calculate_entropy",
    "calculate_variance",
    "calculate_margin",
    "calculate_confidence",
    "is_uncertain",
    "calculate_all_metrics",
    # Active Learning
    "ActiveLearningEngine",
    "AnnotationCandidate",
    "SamplingStrategy",
    "SelectionCriterion",
    # Training Pipeline
    "TrainingPipeline",
    # Model Versioning & A/B Testing
    "ModelVersioningManager",
    "ABTestManager",
    "DeploymentStrategy",
    "ModelStatus",
    # Drift Detection
    "DriftDetector",
    "DriftType",
    "DriftSeverity",
]
