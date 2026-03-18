from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from temporaldata import IrregularTimeSeries

from auditorydecoding.preprocessing import (
    PreprocessingPipeline,
    Segment,
    split_segments,
)


def plot_preprocessing_stages(
    ecog: IrregularTimeSeries,
    pipeline: PreprocessingPipeline,
    channel_names: np.ndarray | list[str] | None = None,
    n_snapshots: int = 3,
    snapshot_duration: float = 5.0,
    channels: list[int] | None = None,
    n_channels: int = 8,
) -> plt.Figure:
    """Visualise the signal at multiple time points after each preprocessing step.

    Produces a grid of subplots with one **row per stage** (raw + each step)
    and one **column per snapshot**.  Snapshots are 5-second windows picked at
    evenly-spaced times across different recording segments so you can see how
    the preprocessing behaves at various points in the session.

    Parameters
    ----------
    ecog
        The **raw** (un-preprocessed) ``IrregularTimeSeries``.  The pipeline
        must already be fitted.
    pipeline
        Fitted :class:`PreprocessingPipeline`.
    channel_names
        Array of channel name strings (length = n_channels in the **raw**
        signal).  Passed by the dataset so the y-axis shows real electrode
        names.  After a ``ChannelSelector`` step only the surviving names are
        shown.
    n_snapshots
        How many time windows to display (columns).
    snapshot_duration
        Duration of each snapshot in seconds.
    channels
        Explicit list of channel indices to display.  Overrides *n_channels*.
    n_channels
        How many channels to show (evenly spaced) when *channels* is ``None``.
    """
    segments = split_segments(ecog)
    stages = pipeline.transform_incremental(segments)

    n_stages = len(stages)
    stage_labels = ["Raw"] + pipeline.step_names

    snapshot_specs = _pick_snapshots(segments, n_snapshots, snapshot_duration)

    fig, axes = plt.subplots(
        n_stages,
        n_snapshots,
        figsize=(6 * n_snapshots, 3 * n_stages),
        squeeze=False,
    )

    stage_channel_names = _build_stage_channel_names(
        pipeline, channel_names, segments[0].signal.shape[1]
    )

    for row, (stage_segments, stage_label) in enumerate(
        zip(stages, stage_labels)
    ):
        names_for_row = stage_channel_names[row]

        for col, (seg_idx, t_start, t_end) in enumerate(snapshot_specs):
            seg = stage_segments[seg_idx]
            ts = seg.timestamps
            mask = (ts >= t_start) & (ts <= t_end)
            signal = seg.signal[mask]
            timestamps = ts[mask]

            ax = axes[row, col]
            if signal.size == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
                continue

            total_ch = signal.shape[1]
            if channels is not None:
                ch_idx = np.asarray(channels)
            else:
                ch_idx = np.linspace(
                    0, total_ch - 1, min(n_channels, total_ch), dtype=int
                )

            spacing = 4 * np.std(signal[:, ch_idx])
            if spacing == 0:
                spacing = 1.0
            for i, ch in enumerate(ch_idx):
                ax.plot(
                    timestamps,
                    signal[:, ch] + i * spacing,
                    linewidth=0.4,
                    color="k",
                )

            ax.set_yticks([i * spacing for i in range(len(ch_idx))])
            if names_for_row is not None:
                ax.set_yticklabels(
                    [names_for_row[ch] for ch in ch_idx], fontsize=7
                )
            else:
                ax.set_yticklabels([f"ch {ch}" for ch in ch_idx], fontsize=7)

            if row == 0:
                ax.set_title(
                    f"Segment {seg_idx}  [{t_start:.1f} – {t_end:.1f} s]",
                    fontsize=9,
                )
            if col == 0:
                ax.set_ylabel(stage_label, fontsize=10, fontweight="bold")
            if row == n_stages - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle("Preprocessing stages", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _pick_snapshots(
    segments: list[Segment],
    n_snapshots: int,
    duration: float,
) -> list[tuple[int, float, float]]:
    """Choose *n_snapshots* ``(segment_idx, t_start, t_end)`` tuples spread
    across different segments and times.

    Picks windows where data actually exists by detecting contiguous blocks
    within each segment's timestamps (gaps > 10× median dt are treated as
    recording boundaries).
    """
    contiguous_blocks: list[tuple[int, float, float]] = []

    for seg_idx, seg in enumerate(segments):
        ts = seg.timestamps
        if len(ts) < 2:
            contiguous_blocks.append((seg_idx, ts[0], ts[0]))
            continue

        dt = np.diff(ts)
        median_dt = np.median(dt)
        gap_indices = np.where(dt > 10 * median_dt)[0]

        block_starts = np.concatenate([[0], gap_indices + 1])
        block_ends = np.concatenate([gap_indices, [len(ts) - 1]])

        for bs, be in zip(block_starts, block_ends):
            contiguous_blocks.append((seg_idx, ts[bs], ts[be]))

    candidates: list[tuple[int, float, float]] = []
    for seg_idx, blk_start, blk_end in contiguous_blocks:
        blk_dur = blk_end - blk_start
        if blk_dur < duration:
            if blk_dur > 0:
                candidates.append((seg_idx, blk_start, blk_end))
            continue
        for frac in (0.1, 0.5, 0.9):
            t0 = blk_start + frac * (blk_dur - duration)
            candidates.append((seg_idx, t0, t0 + duration))

    if not candidates:
        return [
            (0, segments[0].timestamps[0], segments[0].timestamps[0] + duration)
        ]

    if len(candidates) <= n_snapshots:
        return candidates

    step = len(candidates) / n_snapshots
    return [candidates[int(i * step)] for i in range(n_snapshots)]


def _build_stage_channel_names(
    pipeline: PreprocessingPipeline,
    raw_channel_names: np.ndarray | list[str] | None,
    n_raw_channels: int,
) -> list[np.ndarray | None]:
    """Return a list of channel-name arrays, one per stage (raw + each step)."""
    from auditorydecoding.preprocessing import ChannelSelector

    if raw_channel_names is None:
        return [None] * (len(pipeline.steps) + 1)

    names = np.asarray(raw_channel_names)
    result: list[np.ndarray | None] = [names]

    for step in pipeline.steps:
        if isinstance(step, ChannelSelector) and step._indices is not None:
            names = names[step._indices]
        result.append(names)

    return result
