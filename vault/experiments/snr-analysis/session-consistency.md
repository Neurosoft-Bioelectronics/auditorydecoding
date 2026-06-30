---
id: session-consistency
title: "Session Performance Consistency Across Baselines"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: strongly-supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - session-consistency
  - baselines
  - decoding
depends_on:
  - "[[snr-vs-decoding-performance]]"
leads_to: []
---

# Session Performance Consistency Across Baselines

## Question

Is session performance consistent across different baseline architectures? Does session quality dominate over model choice?

## Hypothesis

Session quality (driven by neural signal quality) dominates over model choice in determining decoding performance. Session rankings should be highly correlated across baselines.

## Background

If sessions with good signal quality consistently decode well and sessions with poor signal quality consistently decode poorly — regardless of which model is used — then the choice of *which sessions to include* matters more than *which model to train*. This has important practical implications for experimental design and data collection.

## Method

Compute pairwise Spearman rank correlations of session F1 across all 6 baselines. Use Kendall's W (coefficient of concordance) to assess overall agreement. Generate session performance heatmap.

- Script: `scripts/snr_baseline_correlation.py`

## Results

| Metric | Value |
|--------|-------|
| Kendall's W | 0.747 |
| Mean pairwise Spearman ρ | 0.70 |

Session performance is highly consistent across baselines. Session rankings are largely architecture-independent. Some sessions are intrinsically easier to decode due to their neural signal quality.

Session-level variability dominates model-level variability: the variability in F1 driven by which session is being decoded is far larger than the variability introduced by model choice.

## Interpretation

With Kendall's W = 0.747 and mean pairwise ρ = 0.70, session rankings are highly stable regardless of architecture. The SNR analysis can serve as a reliable a priori predictor of decoder-friendly sessions. Sessions with high max broadband SNR or strong tonotopic tuning will consistently produce good results across any reasonable architecture.

The practical implication is that when evaluating new decoding methods, the choice of *evaluation sessions* is critical — a method tested only on high-SNR sessions will appear better than one tested on low-SNR sessions, regardless of actual model quality.

## Verdict

**Outcome:** Strongly Supported — Session rankings are highly consistent across baselines (Kendall's W = 0.747, mean pairwise ρ = 0.70). Session quality dominates model choice.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Baseline Correlation Table](results/baseline_correlation_table.csv)
- [Session Summary Table](results/session_summary_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[snr-vs-decoding-performance]]
