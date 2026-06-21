"""InsightPipeline — orchestrates all 5 stages end-to-end."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dissectml._config import DissectMLConfig, get_config
from dissectml.core.base import PipelineContext
from dissectml.core.data_container import DataContainer


class InsightPipeline:
    """Lightweight stage orchestrator that passes context between stages.

    .. note::
        The end-to-end pipeline is driven by :func:`dissectml.analyze`, which
        calls each stage directly; it does **not** use this class. Only
        :meth:`run_eda` is currently implemented here — :meth:`run` is a stub.
        Use ``dml.analyze()`` for the full pipeline or ``dml.explore()`` /
        ``dml.battle()`` for individual stages.

    Stages (full pipeline, via ``dml.analyze``):
        1. EDA         — Deep exploratory data analysis
        2. Intelligence — Pre-model intelligence (leakage, readiness, recommendations)
        3. Battle       — Multi-model parallel training + CV scoring
        4. Compare      — Statistical comparison, error analysis, SHAP, Pareto
        5. Report       — HTML report generation
    """

    def __init__(self, config: DissectMLConfig | None = None) -> None:
        self.config = config or get_config()
        self.context = PipelineContext(config=self.config)

    def run_eda(self, container: DataContainer) -> Any:
        """Run Stage 1: Deep EDA."""
        from dissectml.eda import EDAStage
        stage = EDAStage()
        result = stage.timed_run(container, self.context)
        self.context.eda_result = result
        return result

    def run(
        self,
        data: str | pd.DataFrame,
        target: str | None = None,
        task: str = "auto",
    ) -> Any:
        """Run the complete pipeline and return an AnalysisReport.

        Not implemented on this class. Use :func:`dissectml.analyze` instead,
        which runs all five stages end-to-end.
        """
        container = DataContainer.from_input(data, target=target, task=task, config=self.config)

        # Stage 1: EDA (the only stage wired into this orchestrator)
        self.run_eda(container)

        # Stages 2-5 are not orchestrated here; dml.analyze() drives them directly.
        raise NotImplementedError(
            "Full pipeline orchestration via InsightPipeline.run() is not "
            "implemented. Use dml.analyze(df, target=...) for the full pipeline, "
            "or dml.explore(df) for EDA only."
        )
