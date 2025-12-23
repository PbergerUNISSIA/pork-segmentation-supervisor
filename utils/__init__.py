"""
Utilities module for Pork Segmentation Supervisor.
"""

from .logger import logger, log_function_call, LogContext
from .visualization import (
    create_comparison_plot,
    create_uncertainty_heatmap,
    format_confidence_badge,
    format_metric_card,
    format_status_badge,
    plot_confidence_distribution,
    plot_inference_time_distribution,
    plot_metrics_over_time,
    plot_uncertainty_pie,
    plot_validation_progress,
)

__all__ = [
    # Logger
    "logger",
    "log_function_call",
    "LogContext",
    # Visualization
    "plot_confidence_distribution",
    "plot_metrics_over_time",
    "plot_uncertainty_pie",
    "plot_validation_progress",
    "plot_inference_time_distribution",
    "create_uncertainty_heatmap",
    "create_comparison_plot",
    "format_metric_card",
    "format_confidence_badge",
    "format_status_badge",
]
