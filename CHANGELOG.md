# Changelog

All notable changes to DissectML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-06-21

### Fixed
- Removed the silent `n_iter=5` cap in `ModelTuner`; tuned mode now honors the requested `n_iter` (default 20).
- Tuned model scores are now fully consistent: every metric, standard deviation, and out-of-fold prediction is recomputed for the tuned estimator on the battle's CV folds. Previously only the primary metric was overwritten, leaving the remaining metrics and OOF predictions from the untuned model (which Stage-4 error analysis, ROC, and significance tests relied on).
- Long axis labels (model and feature names) on horizontal bar charts are no longer clipped — enabled Plotly `automargin` on the theme axes.

### Changed
- Renamed the Plotly template `INSIGHTML_TEMPLATE` to `DISSECTML_TEMPLATE`.

### Deprecated
- `dissectml.viz.INSIGHTML_TEMPLATE` is kept as an alias for `DISSECTML_TEMPLATE` and may be removed in a future release.

### Removed
- Dead `report/sections/` module — the HTML renderer builds its sections internally.
- Unused `_MNAR_IMPUTER` constant in battle preprocessing.

### Documentation
- Corrected the model catalog count to 19 classifiers + 19 regressors (38 models) in README, docs, and PLAN.
- Corrected misleading `InsightPipeline` docstrings (the full pipeline is driven by `dml.analyze()`).
- Updated the remaining `iml` import alias to `dml` in `DissectML.html` examples.

## [0.1.2] - 2026-04-07

### Changed
- Renamed package from `insightml` to `dissectml` on PyPI
- Renamed project from InsightML to DissectML across all references
- Changed import alias from `iml` to `dml`
- Housing dataset now fetched from sklearn on first use and cached locally (removed 1.9MB bundled CSV from wheel)
- Added report screenshot to README
- Added Colab demo notebook with "Open in Colab" badge

### Added
- 74 new tests: report sections, `_lazy`, pipeline, progress, `_io`, `_sampling` (599 total)
- API reference docs (6 mkdocstrings pages)
- Community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, issue/PR templates
- `catboost_info/` added to .gitignore, junk files excluded from sdist

### Fixed
- Removed `numpy<2.0` upper bound constraint
- Fixed repo URLs in mkdocs.yml and pyproject.toml
- Fixed PowerShell parse error in CI (added `shell: bash`)
- Fixed `render_html_report()` unsupported `inline_plotly` kwarg

## [0.1.0] - 2026-04-06

### Added
- `dml.explore(df)` — Deep EDA with lazy evaluation
- Dataset overview: type detection, column profiles, memory stats
- Univariate analysis: distributions, KDE, descriptive stats
- Bivariate analysis: cross-type pair analysis
- Correlation analysis: unified matrix (Pearson/Spearman/Cramer's V/eta)
- Missing data intelligence: Little's MCAR test, MAR/MNAR classification
- Outlier detection: IQR, Z-score, Isolation Forest with consensus
- Statistical tests: normality, independence, variance, group comparison
- Cluster discovery: auto K-Means + DBSCAN with profiling
- Feature interactions: interaction strength, non-linearity detection
- Target analysis: class balance, distribution, feature-target relationships
- `dml.battle(df, target)` — parallel CV across 19 classifiers / 19 regressors
- EDA-informed preprocessing (KNN imputer, Robust scaler, OrdinalEncoder)
- ModelRegistry, MODEL_CATALOG, ModelTuner (quick/tuned/custom modes)
- `dml.analyze_intelligence(df, target)` — 4-pronged leakage detection, VIF, condition number
- Data readiness score 0–100 with grade (A–F) and penalty waterfall
- Composite feature importance ranking (MI + correlation + F-score)
- Algorithm recommendations engine (7 algorithm profiles)
- `ModelComparator` — McNemar test, corrected paired t-test, Pareto front, error analysis
- ROC/PR curves, confusion matrices, residual plots, actual vs predicted
- SHAP model comparison (TreeExplainer / LinearExplainer / KernelExplainer)
- `dml.analyze(df, target)` — full 5-stage pipeline, returns `AnalysisReport`
- `AnalysisReport.export(path)` — self-contained interactive HTML report
- `AnalysisReport.show()` — export + open in browser
- `dml.load_titanic()` / `dml.load_housing()` — built-in demo datasets
- `dml.to_pandas()` — Polars DataFrame / file path / dict / numpy array conversion
- `report/pdf_renderer.py` — optional PDF export via WeasyPrint
- Report sections module (`report/sections/`), Jinja2 templates, CSS/JS assets
- GitHub Actions CI/CD (`.github/workflows/ci.yml`, `release.yml`)
- MkDocs Material documentation site
- 4 example Jupyter notebooks (quickstart, deep EDA, model battle, full pipeline)
- 472 tests (453 passing, 3 skipped — tabulate/weasyprint optional deps)
