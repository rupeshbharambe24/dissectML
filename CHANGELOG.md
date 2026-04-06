# Changelog

All notable changes to InsightML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: package structure, pyproject.toml, CI config
- Core infrastructure: DataContainer, BaseStage, PipelineContext
- Configuration system: InsightMLConfig with global/context/per-call overrides
- Exception hierarchy: InsightMLError and 10 subclasses
- Type system: TaskType, ColumnType, MissingnessType, TuningMode enums
- File I/O: CSV, Excel, Parquet, JSON via pandas readers
- Smart sampling: stratified, temporal, random fallback
- Visualization infrastructure: InsightML Plotly theme

## [0.1.0] - TBD

### Added
- `iml.explore(df)` — Deep EDA with lazy evaluation
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
