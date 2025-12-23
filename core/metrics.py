"""
====================================================================
PORK SEGMENTATION SUPERVISOR - METRICS
====================================================================
Calcul des métriques de segmentation et de confiance pour Active Learning.
====================================================================
"""

from typing import Dict, Tuple
import numpy as np
from config.settings import settings
from utils.logger import logger


def calculate_dice(prediction: np.ndarray, ground_truth: np.ndarray, smooth: float = 1e-6) -> float:
    """Calcule le coefficient de Dice."""
    prediction = (prediction > 0).astype(np.float32)
    ground_truth = (ground_truth > 0).astype(np.float32)
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(gt_flat) + smooth)
    return float(dice)


def calculate_iou(prediction: np.ndarray, ground_truth: np.ndarray, smooth: float = 1e-6) -> float:
    """Calcule l'Intersection over Union."""
    prediction = (prediction > 0).astype(np.float32)
    ground_truth = (ground_truth > 0).astype(np.float32)
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def calculate_precision_recall(prediction: np.ndarray, ground_truth: np.ndarray, smooth: float = 1e-6) -> Tuple[float, float]:
    """Calcule la Precision et le Recall."""
    prediction = (prediction > 0).astype(np.float32)
    ground_truth = (ground_truth > 0).astype(np.float32)
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    tp = np.sum(pred_flat * gt_flat)
    fp = np.sum(pred_flat * (1 - gt_flat))
    fn = np.sum((1 - pred_flat) * gt_flat)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return float(precision), float(recall)


def calculate_entropy(probabilities: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calcule l'entropie moyenne des prédictions."""
    probs_clipped = np.clip(probabilities, epsilon, 1.0)
    pixel_entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=-1)
    mean_entropy = np.mean(pixel_entropy)
    num_classes = probabilities.shape[-1]
    max_entropy = np.log(num_classes)
    normalized_entropy = mean_entropy / max_entropy
    return float(normalized_entropy)


def calculate_variance(probabilities: np.ndarray) -> float:
    """Calcule la variance moyenne des prédictions."""
    pixel_variance = np.var(probabilities, axis=-1)
    mean_variance = np.mean(pixel_variance)
    return float(mean_variance)


def calculate_margin(probabilities: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calcule la marge moyenne."""
    sorted_probs = np.sort(probabilities, axis=-1)[:, :, ::-1]
    if probabilities.shape[-1] >= 2:
        pixel_margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
    else:
        pixel_margin = sorted_probs[:, :, 0]
    mean_margin = np.mean(pixel_margin)
    return float(mean_margin)


def calculate_confidence(probabilities: np.ndarray, method: str = None) -> float:
    """Calcule un score de confiance selon la méthode configurée."""
    method = method or settings.UNCERTAINTY_METHOD

    if method == "entropy":
        entropy = calculate_entropy(probabilities)
        confidence = 1.0 - entropy
    elif method == "variance":
        variance = calculate_variance(probabilities)
        max_variance = 0.25
        normalized_variance = min(variance / max_variance, 1.0)
        confidence = 1.0 - normalized_variance
    elif method == "margin":
        margin = calculate_margin(probabilities)
        confidence = margin
    else:
        raise ValueError(f"Invalid uncertainty method: {method}")

    return float(confidence)


def is_uncertain(probabilities: np.ndarray, threshold: float = None, method: str = None) -> bool:
    """Détermine si une prédiction est incertaine."""
    threshold = threshold or settings.MODEL_CONFIDENCE_THRESHOLD
    confidence = calculate_confidence(probabilities, method)
    return confidence < threshold


def calculate_all_metrics(prediction: np.ndarray, ground_truth: np.ndarray, probabilities: np.ndarray = None) -> Dict[str, float]:
    """Calcule toutes les métriques pour une prédiction."""
    metrics = {}
    metrics["dice"] = calculate_dice(prediction, ground_truth)
    metrics["iou"] = calculate_iou(prediction, ground_truth)
    precision, recall = calculate_precision_recall(prediction, ground_truth)
    metrics["precision"] = precision
    metrics["recall"] = recall
    if precision + recall > 0:
        metrics["f1"] = 2 * (precision * recall) / (precision + recall)
    else:
        metrics["f1"] = 0.0

    if probabilities is not None:
        metrics["entropy"] = calculate_entropy(probabilities)
        metrics["variance"] = calculate_variance(probabilities)
        metrics["margin"] = calculate_margin(probabilities)
        metrics["confidence"] = calculate_confidence(probabilities)
        metrics["is_uncertain"] = is_uncertain(probabilities)

    return metrics
