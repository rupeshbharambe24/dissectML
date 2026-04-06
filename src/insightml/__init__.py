"""
InsightML — The missing middle layer between EDA and AutoML.

Unified pipeline from deep data understanding to model comparison,
in as few as 3 function calls.

Quick start::

    import insightml as iml

    # Deep EDA (v0.1+)
    eda = iml.explore(df)
    eda.overview.show()
    eda.correlations.heatmap()

    # Model battle (v0.2+)
    models = iml.battle(df, target="price")

    # Full pipeline (v0.4+)
    report = iml.analyze(df, target="price")
    report.export("report.html")
"""

from __future__ import annotations

from insightml._version import __version__
from insightml._config import InsightMLConfig, config_context, get_config, set_config
from insightml.exceptions import InsightMLError
from insightml.eda import explore
from insightml.battle import battle
from insightml.intelligence import analyze_intelligence

__all__ = [
    "__version__",
    # Public API
    "explore",
    "battle",
    "analyze_intelligence",
    "analyze",
    # Config
    "InsightMLConfig",
    "get_config",
    "set_config",
    "config_context",
    # Exceptions
    "InsightMLError",
]


def analyze(data, target: str, *, task: str = "auto", **kwargs):
    """Full pipeline: EDA → Intelligence → Battle → Compare → Report.

    Available in v0.4+. Current version is v0.1 (EDA-only).
    """
    raise NotImplementedError(
        "iml.analyze() is available in v0.4+. "
        f"Current: {__version__} (EDA-only). "
        "Use iml.explore(df) for deep EDA."
    )
