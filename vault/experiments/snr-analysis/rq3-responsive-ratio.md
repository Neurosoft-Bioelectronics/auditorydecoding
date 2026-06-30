---
id: rq3
title: "Trial-by-Trial Reliability (Responsive Ratio)"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - responsive-ratio
  - trial-reliability
depends_on:
  - "[[rq1-channel-session-snr]]"
leads_to:
  - "[[rq6-power-ratio-snr]]"
---

# RQ3: Trial-by-Trial Reliability (Responsive Ratio)

## Question

What proportion of trials show a reliable energy increase during stimulus compared to baseline?

## Hypothesis

Active channels (by SNR classification) show stimulus energy increase well above the 50% chance level on individual trials.

## Background

The evoked-potential SNR relies on trial averaging. The responsive ratio provides an independent, single-trial validation: for each trial, does the stimulus window have higher variance than the rest window? By chance, this should be ~50%. A responsive ratio well above 50% confirms that the channel consistently responds to stimulation on individual trials.

## Method

For each channel, compute the fraction of trials where Var(stimulus) > Var(rest). Compare responsive ratios between Active and Dead channel groups. Test correlation with broadband SNR.

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Active Channels | Dead Channels |
|--------|----------------|---------------|
| Mean Responsive Ratio | 72.9% | 47.0% |
| Median Responsive Ratio | 72.8% | 49.2% |

Pearson correlation between BB SNR and responsive ratio (active channels): r = 0.516, p = 0.034.

## Interpretation

Active channels show a responsive ratio significantly above 50%, confirming reliable trial-by-trial stimulus-evoked energy increases. Dead channels sit right at chance level (~49%), behaving like pure noise. The responsive ratio serves as an independent validation of the SNR-based channel classification. The moderate positive correlation (r = 0.516) confirms that higher-SNR channels respond more consistently across trials.

## Verdict

**Outcome:** Supported — Active channels average 72.9% responsive ratio vs. 47.0% for dead channels, independently validating the SNR classification.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Channel Quality Table](results/channel_quality_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq1-channel-session-snr]]
- Leads to: [[rq6-power-ratio-snr]]
