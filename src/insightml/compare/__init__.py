"""Stage 4: Comparative analysis — metrics, curves, significance, error analysis."""

from insightml.compare.comparator import ModelComparator
from insightml.compare.curves import (
    actual_vs_predicted,
    confusion_matrices,
    metric_bar_chart,
    pr_curves,
    residual_plots,
    roc_curves,
)
from insightml.compare.error_analysis import ErrorAnalysisResult, analyze_errors
from insightml.compare.metrics_table import ComparisonTable
from insightml.compare.pareto import get_pareto_models, pareto_front
from insightml.compare.significance import corrected_ttest_matrix, mcnemar_matrix

__all__ = [
    "ModelComparator",
    "ComparisonTable",
    "analyze_errors",
    "ErrorAnalysisResult",
    "pareto_front",
    "get_pareto_models",
    "roc_curves",
    "pr_curves",
    "confusion_matrices",
    "residual_plots",
    "actual_vs_predicted",
    "metric_bar_chart",
    "mcnemar_matrix",
    "corrected_ttest_matrix",
]
