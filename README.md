# InsightML

**The missing middle layer between EDA and AutoML.**

InsightML (`insightml`) is a Python library that unifies deep exploratory data analysis with comparative model analysis — the full journey from "What is my data?" to "Which model is best and WHY?", in as few as 3 function calls.

## Installation

```bash
pip install insightml
```

With optional extras:

```bash
pip install insightml[boost]    # XGBoost, LightGBM, CatBoost
pip install insightml[explain]  # SHAP explainability
pip install insightml[full]     # Everything
```

## Quick Start

```python
import insightml as iml

# Deep EDA (v0.1+)
eda = iml.explore(df)
eda.overview.show()
eda.correlations.heatmap()
eda.missing.patterns()
eda.outliers.plot()

# Model battle (v0.2+)
models = iml.battle(df, target="price")
models.compare()

# Full pipeline (v0.4+)
report = iml.analyze(df, target="price", task="regression")
report.export("report.html")
```

## Roadmap

| Version | Features |
|---------|----------|
| v0.1 | Deep EDA (`iml.explore()`) |
| v0.2 | Multi-model training (`iml.battle()`) |
| v0.3 | Pre-model intelligence bridge |
| v0.4 | Comparative analysis + HTML reports |
| v0.5 | Polars backend, PDF export, scaling |

## License

MIT
