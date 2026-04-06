"""EDA stage — deep exploratory data analysis."""

from insightml.eda.result import EDAResult, explore
from insightml.core.base import BaseStage, StageResult
from insightml.core.data_container import DataContainer


class EDAStage(BaseStage):
    """Pipeline stage wrapper for EDA (used by InsightPipeline)."""

    @property
    def name(self) -> str:
        return "EDA"

    def run(self, container: DataContainer, context) -> EDAResult:
        from insightml.eda.result import EDAResult
        result = EDAResult(
            container.df,
            target=container.target,
            task=container.task.value,
            config=context.config,
        )
        return result


__all__ = ["EDAResult", "EDAStage", "explore"]
