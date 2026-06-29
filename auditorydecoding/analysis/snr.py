"""iEEG signal quality analysis via paired rest-stimulus epochs.

This module implements the SNR protocol defined in SNR_EXP.md. All functions
operate on numpy arrays extracted from torch_brain Data objects so they remain
testable and reusable outside a specific pipeline.

Typical usage::

    from auditorydecoding.analysis.snr import (
        load_session,
        extract_epochs,
        compute_channel_snr,
        compute_responsive_ratio,
        compute_tonotopic_tuning,
        build_channel_table,
        build_session_table,
    )

    session = load_session(path)
    epochs  = extract_epochs(session)
    ch_tbl  = build_channel_table(session, epochs)
    ses_tbl = build_session_table(ch_tbl)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from torch_brain.data import Data


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Minimal metadata extracted from a torch_brain Data object."""

    session_id: str
    signal: np.ndarray  # (n_samples, n_channels)
    timestamps: np.ndarray  # (n_samples,)
    sampling_rate: float
    channel_names: np.ndarray  # (n_channels,)
    ecog_mask: np.ndarray  # bool (n_channels,) — True for ECOG channels

    # Trial timing arrays (only stimulus-on trials)
    stim_starts: np.ndarray  # (n_trials,) in seconds
    stim_ends: np.ndarray  # (n_trials,)
    stim_labels: np.ndarray  # (n_trials,) e.g. "stim_1000Hz"

    # Rest ("off") timing arrays — one per stimulus trial
    rest_starts: np.ndarray  # (n_trials,)
    rest_ends: np.ndarray  # (n_trials,)


@dataclass
class EpochArrays:
    """Windowed signal arrays aligned to rest/stimulus pairs.

    All arrays have shape (n_epochs, n_channels, n_time) where *n_time* is
    the number of samples in a 0.5 s window.
    """

    rest: np.ndarray
    stimulus: np.ndarray
    stim_labels: np.ndarray  # (n_epochs,) frequency label per epoch
    window_samples: int  # samples per 0.5 s window


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_session(path: str | Path, lazy: bool = False) -> SessionInfo:
    """Load a processed .h5 session into a lightweight :class:`SessionInfo`.

    Parameters
    ----------
    path : str or Path
        Path to a ``*.h5`` file produced by the Neurosoft pipeline.
    lazy : bool
        If True keep the HDF5 file handle open (faster but caller must close).

    Returns
    -------
    SessionInfo
    """
    data = Data.load(path, lazy=lazy)
    try:
        return _data_to_session_info(data)
    finally:
        if not lazy:
            data.close()


def _data_to_session_info(data: Data) -> SessionInfo:
    signal = np.asarray(data.ecog.signal)  # (T, C)
    timestamps = np.asarray(data.ecog.timestamps)  # (T,)
    sfreq = 1.0 / float(np.median(np.diff(timestamps)))

    ch_names = np.asarray(data.channels.id)
    ch_types = np.asarray(data.channels.type)
    ecog_mask = np.array([t.lower() == "ecog" for t in ch_types])

    on_off_labels = np.asarray(data.on_vs_off_trials.behavior_labels)
    on_off_starts = np.asarray(data.on_vs_off_trials.start)
    on_off_ends = np.asarray(data.on_vs_off_trials.end)

    stim_starts = np.asarray(data.acoustic_stim_trials.start)
    stim_ends = np.asarray(data.acoustic_stim_trials.end)
    stim_labels = np.asarray(data.acoustic_stim_trials.behavior_labels)

    rest_starts, rest_ends = _pair_rest_to_stim(
        on_off_labels, on_off_starts, on_off_ends, stim_starts
    )

    return SessionInfo(
        session_id=str(data.session.id),
        signal=signal,
        timestamps=timestamps,
        sampling_rate=round(sfreq),
        channel_names=ch_names,
        ecog_mask=ecog_mask,
        stim_starts=stim_starts,
        stim_ends=stim_ends,
        stim_labels=stim_labels,
        rest_starts=rest_starts,
        rest_ends=rest_ends,
    )


def _pair_rest_to_stim(
    on_off_labels: np.ndarray,
    on_off_starts: np.ndarray,
    on_off_ends: np.ndarray,
    stim_starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each stimulus trial, find the preceding "off" interval.

    The convention is: each stimulus onset is preceded by a 0.5 s rest window.
    We match by finding the "off" interval whose *end* is closest to (and not
    after) each stimulus start.
    """
    off_mask = on_off_labels == "off"
    off_starts = on_off_starts[off_mask]
    off_ends = on_off_ends[off_mask]

    rest_s = np.empty_like(stim_starts)
    rest_e = np.empty_like(stim_starts)

    for i, t_stim in enumerate(stim_starts):
        # Candidates: off intervals ending at or before stimulus onset (small tolerance)
        candidates = np.where(off_ends <= t_stim + 0.01)[0]
        if len(candidates) == 0:
            # Fallback: use interval just before stimulus
            rest_e[i] = t_stim
            rest_s[i] = t_stim - 0.5
        else:
            best = candidates[np.argmax(off_ends[candidates])]
            rest_s[i] = off_starts[best]
            rest_e[i] = off_ends[best]

    return rest_s, rest_e


# ---------------------------------------------------------------------------
# Epoch extraction & preprocessing
# ---------------------------------------------------------------------------


def _time_to_index(timestamps: np.ndarray, t: float) -> int:
    """Return the sample index closest to time *t*."""
    return int(np.searchsorted(timestamps, t, side="left"))


def extract_epochs(
    session: SessionInfo,
    window_duration: float = 0.5,
    channels: np.ndarray | None = None,
) -> EpochArrays:
    """Cut the continuous signal into aligned (rest, stimulus) epoch pairs.

    Parameters
    ----------
    session : SessionInfo
    window_duration : float
        Duration of each window in seconds (default 0.5).
    channels : array of bool, optional
        Channel mask. Defaults to ``session.ecog_mask``.

    Returns
    -------
    EpochArrays
    """
    if channels is None:
        channels = session.ecog_mask

    win_samples = int(round(window_duration * session.sampling_rate))
    sig = session.signal[:, channels]  # (T, C_sel)
    ts = session.timestamps

    rest_list, stim_list, label_list = [], [], []

    for i in range(len(session.stim_starts)):
        r_start = _time_to_index(ts, session.rest_starts[i])
        r_end = r_start + win_samples
        s_start = _time_to_index(ts, session.stim_starts[i])
        s_end = s_start + win_samples

        if r_end > len(ts) or s_end > len(ts):
            continue

        rest_list.append(sig[r_start:r_end].T)  # (C, W)
        stim_list.append(sig[s_start:s_end].T)  # (C, W)
        label_list.append(session.stim_labels[i])

    return EpochArrays(
        rest=np.stack(rest_list),  # (N, C, W)
        stimulus=np.stack(stim_list),  # (N, C, W)
        stim_labels=np.array(label_list),  # (N,)
        window_samples=win_samples,
    )


def baseline_correct(epochs: EpochArrays) -> EpochArrays:
    """Subtract the per-epoch, per-channel rest-window mean from both windows.

    This implements the *local baseline correction* step from the protocol.
    Returns a **new** EpochArrays (does not mutate in-place).
    """
    rest_mean = epochs.rest.mean(axis=-1, keepdims=True)  # (N, C, 1)
    return EpochArrays(
        rest=epochs.rest - rest_mean,
        stimulus=epochs.stimulus - rest_mean,
        stim_labels=epochs.stim_labels,
        window_samples=epochs.window_samples,
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _apply_bandpass(
    signal_3d: np.ndarray,
    low: float,
    high: float,
    sfreq: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass on axis=-1 of an (N, C, T) array."""
    nyq = sfreq / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal_3d, axis=-1).astype(signal_3d.dtype)


def _apply_notch(
    signal_3d: np.ndarray,
    freq: float,
    sfreq: float,
    quality: float = 30.0,
) -> np.ndarray:
    """Zero-phase notch filter on axis=-1 of an (N, C, T) array."""
    b, a = iirnotch(freq, quality, sfreq)
    return filtfilt(b, a, signal_3d, axis=-1).astype(signal_3d.dtype)


def apply_broadband_filter(
    epochs: EpochArrays,
    sfreq: float,
    lowcut: float = 1.0,
    highcut: float = 300.0,
    notch_freq: float = 50.0,
) -> EpochArrays:
    """Broadband pipeline: 1–300 Hz bandpass + 50 Hz notch."""
    rest = _apply_bandpass(epochs.rest, lowcut, highcut, sfreq)
    stim = _apply_bandpass(epochs.stimulus, lowcut, highcut, sfreq)
    rest = _apply_notch(rest, notch_freq, sfreq)
    stim = _apply_notch(stim, notch_freq, sfreq)
    return EpochArrays(
        rest=rest,
        stimulus=stim,
        stim_labels=epochs.stim_labels,
        window_samples=epochs.window_samples,
    )


def apply_high_gamma_filter(
    epochs: EpochArrays,
    sfreq: float,
    lowcut: float = 70.0,
    highcut: float = 150.0,
) -> EpochArrays:
    """High-gamma pipeline: 70–150 Hz zero-phase bandpass."""
    rest = _apply_bandpass(epochs.rest, lowcut, highcut, sfreq)
    stim = _apply_bandpass(epochs.stimulus, lowcut, highcut, sfreq)
    return EpochArrays(
        rest=rest,
        stimulus=stim,
        stim_labels=epochs.stim_labels,
        window_samples=epochs.window_samples,
    )


def apply_low_frequency_filter(
    epochs: EpochArrays,
    sfreq: float,
    lowcut: float = 1.0,
    highcut: float = 70.0,
    notch_freq: float = 50.0,
) -> EpochArrays:
    """Low-frequency pipeline: 1–70 Hz bandpass + 50 Hz notch.

    This isolates slow cortical potentials and lower-frequency evoked
    components while discarding high-frequency noise and muscle artifacts.
    """
    rest = _apply_bandpass(epochs.rest, lowcut, highcut, sfreq)
    stim = _apply_bandpass(epochs.stimulus, lowcut, highcut, sfreq)
    rest = _apply_notch(rest, notch_freq, sfreq)
    stim = _apply_notch(stim, notch_freq, sfreq)
    return EpochArrays(
        rest=rest,
        stimulus=stim,
        stim_labels=epochs.stim_labels,
        window_samples=epochs.window_samples,
    )


# ---------------------------------------------------------------------------
# RQ1 — Channel-level & session-level SNR
# ---------------------------------------------------------------------------


def compute_channel_snr(epochs: EpochArrays) -> np.ndarray:
    r"""Evoked-potential SNR per channel.

    .. math::

        \text{SNR}_c = \frac{\operatorname{Var}(\bar{S}_c(t))}
                            {\frac{1}{N}\sum_{i=1}^N \operatorname{Var}(R_{i,c}(t))}

    where :math:`\bar{S}_c(t)` is the ERP (trial-averaged stimulus waveform)
    and the denominator is the mean single-trial rest variance.

    Parameters
    ----------
    epochs : EpochArrays
        Baseline-corrected and (optionally) filtered epochs.

    Returns
    -------
    snr : np.ndarray, shape (n_channels,)
    """
    erp = epochs.stimulus.mean(axis=0)  # (C, W)
    erp_var = erp.var(axis=-1)  # (C,)

    rest_var = epochs.rest.var(axis=-1)  # (N, C)
    mean_rest_var = rest_var.mean(axis=0)  # (C,)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(mean_rest_var > 0, erp_var / mean_rest_var, 0.0)
    return snr


def compute_power_ratio_snr(epochs: EpochArrays) -> np.ndarray:
    r"""Power-ratio SNR per channel (phase-insensitive).

    Unlike :func:`compute_channel_snr`, this metric does **not** rely on
    phase-locked ERPs.  Instead it compares the mean single-trial power
    (temporal variance) during stimulus to the mean single-trial power
    during rest:

    .. math::

        \text{SNR}_{\text{power}, c}
            = \frac{\frac{1}{N}\sum_{i=1}^N \operatorname{Var}(S_{i,c}(t))}
                   {\frac{1}{N}\sum_{i=1}^N \operatorname{Var}(R_{i,c}(t))}

    A value > 1 means that, on average, individual stimulus epochs carry
    more total energy than matched rest epochs — regardless of whether
    the response waveform is consistent across trials.

    Parameters
    ----------
    epochs : EpochArrays
        Baseline-corrected and (optionally) filtered epochs.

    Returns
    -------
    snr : np.ndarray, shape (n_channels,)
    """
    stim_var = epochs.stimulus.var(axis=-1)  # (N, C)
    rest_var = epochs.rest.var(axis=-1)  # (N, C)

    mean_stim_var = stim_var.mean(axis=0)  # (C,)
    mean_rest_var = rest_var.mean(axis=0)  # (C,)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(mean_rest_var > 0, mean_stim_var / mean_rest_var, 0.0)
    return snr


def classify_channels(
    snr: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a boolean mask — True for channels with SNR above *threshold*."""
    return snr > threshold


def session_snr(snr: np.ndarray, active_mask: np.ndarray) -> float:
    """Mean SNR across active channels only."""
    if active_mask.sum() == 0:
        return 0.0
    return float(snr[active_mask].mean())


# ---------------------------------------------------------------------------
# RQ3 — Trial-by-trial responsive ratio
# ---------------------------------------------------------------------------


def compute_responsive_ratio(epochs: EpochArrays) -> np.ndarray:
    """Fraction of trials where stimulus variance exceeds rest variance.

    Parameters
    ----------
    epochs : EpochArrays

    Returns
    -------
    ratio : np.ndarray, shape (n_channels,)
        Values in [0, 1].
    """
    rest_var = epochs.rest.var(axis=-1)  # (N, C)
    stim_var = epochs.stimulus.var(axis=-1)  # (N, C)
    responsive = (stim_var > rest_var).astype(float)
    return responsive.mean(axis=0)  # (C,)


# ---------------------------------------------------------------------------
# RQ4 — Tonotopic tuning metric
# ---------------------------------------------------------------------------


def compute_tonotopic_tuning(epochs: EpochArrays) -> np.ndarray:
    """Standard deviation of per-frequency SNR across stimulus types.

    A high value indicates that a channel is preferentially responsive to
    certain frequencies (tonotopic tuning).

    Parameters
    ----------
    epochs : EpochArrays

    Returns
    -------
    tuning : np.ndarray, shape (n_channels,)
    """
    unique_freqs = np.unique(epochs.stim_labels)
    if len(unique_freqs) <= 1:
        n_channels = epochs.rest.shape[1]
        return np.zeros(n_channels)

    freq_snrs = []
    for freq in unique_freqs:
        mask = epochs.stim_labels == freq
        sub = EpochArrays(
            rest=epochs.rest[mask],
            stimulus=epochs.stimulus[mask],
            stim_labels=epochs.stim_labels[mask],
            window_samples=epochs.window_samples,
        )
        freq_snrs.append(compute_channel_snr(sub))

    freq_snr_matrix = np.stack(freq_snrs, axis=0)  # (n_freqs, n_channels)
    return freq_snr_matrix.std(axis=0)


# ---------------------------------------------------------------------------
# ERP extraction (for plotting)
# ---------------------------------------------------------------------------


def compute_erp(epochs: EpochArrays) -> np.ndarray:
    """Grand-average ERP per channel: mean stimulus waveform across trials.

    Returns
    -------
    erp : np.ndarray, shape (n_channels, n_time)
    """
    return epochs.stimulus.mean(axis=0)


# ---------------------------------------------------------------------------
# Aggregate table builders
# ---------------------------------------------------------------------------


def build_channel_table(
    session: SessionInfo,
    broadband_epochs: EpochArrays,
    high_gamma_epochs: EpochArrays,
    snr_threshold: float = 0.5,
    low_freq_epochs: EpochArrays | None = None,
) -> dict[str, np.ndarray]:
    """Compute all per-channel metrics and return a dict suitable for a DataFrame.

    Parameters
    ----------
    session : SessionInfo
    broadband_epochs : EpochArrays
        Baseline-corrected, broadband-filtered epochs (ECOG channels only).
    high_gamma_epochs : EpochArrays
        Baseline-corrected, high-gamma-filtered epochs (ECOG channels only).
    snr_threshold : float
    low_freq_epochs : EpochArrays, optional
        Baseline-corrected, low-frequency-filtered epochs (1–70 Hz).

    Returns
    -------
    dict with keys matching the Channel-Level Quality Table columns.
    """
    bb_snr = compute_channel_snr(broadband_epochs)
    hg_snr = compute_channel_snr(high_gamma_epochs)
    active = classify_channels(bb_snr, threshold=snr_threshold)
    resp = compute_responsive_ratio(broadband_epochs)
    tuning = compute_tonotopic_tuning(broadband_epochs)

    ch_names = session.channel_names[session.ecog_mask]

    bb_power_snr = compute_power_ratio_snr(broadband_epochs)
    hg_power_snr = compute_power_ratio_snr(high_gamma_epochs)

    table = {
        "session_id": np.full(len(ch_names), session.session_id),
        "channel_id": ch_names,
        "status": np.where(active, "Active", "Dead"),
        "broadband_snr": bb_snr,
        "high_gamma_snr": hg_snr,
        "broadband_resp_ratio": resp,
        "tuning_metric": tuning,
        "broadband_power_snr": bb_power_snr,
        "high_gamma_power_snr": hg_power_snr,
    }

    if low_freq_epochs is not None:
        lf_snr = compute_channel_snr(low_freq_epochs)
        lf_resp = compute_responsive_ratio(low_freq_epochs)
        lf_tuning = compute_tonotopic_tuning(low_freq_epochs)
        lf_active = classify_channels(lf_snr, threshold=snr_threshold)
        lf_power_snr = compute_power_ratio_snr(low_freq_epochs)
        table["lowfreq_snr"] = lf_snr
        table["lowfreq_resp_ratio"] = lf_resp
        table["lowfreq_tuning_metric"] = lf_tuning
        table["lowfreq_status"] = np.where(lf_active, "Active", "Dead")
        table["lowfreq_power_snr"] = lf_power_snr

    return table


def build_session_table(
    channel_table: dict[str, np.ndarray],
) -> dict[str, object]:
    """Aggregate channel-level metrics into a single session-level row.

    Parameters
    ----------
    channel_table : dict
        Output of :func:`build_channel_table`.

    Returns
    -------
    dict with keys matching the Session-Level Summary Table columns.
    """
    active = channel_table["status"] == "Active"
    row: dict[str, object] = {
        "session_id": channel_table["session_id"][0],
        "total_channels": len(active),
        "active_channels": int(active.sum()),
        "mean_active_snr": (
            float(channel_table["broadband_snr"][active].mean())
            if active.any()
            else 0.0
        ),
        "mean_active_resp_ratio": (
            float(channel_table["broadband_resp_ratio"][active].mean())
            if active.any()
            else 0.0
        ),
        "max_tuning_metric": float(channel_table["tuning_metric"].max()),
        "mean_broadband_power_snr": float(
            channel_table["broadband_power_snr"].mean()
        ),
        "mean_active_power_snr": (
            float(channel_table["broadband_power_snr"][active].mean())
            if active.any()
            else 0.0
        ),
    }

    if "lowfreq_snr" in channel_table:
        lf_active = channel_table["lowfreq_status"] == "Active"
        row["lowfreq_active_channels"] = int(lf_active.sum())
        row["mean_lowfreq_snr"] = (
            float(channel_table["lowfreq_snr"][lf_active].mean())
            if lf_active.any()
            else 0.0
        )
        row["mean_lowfreq_resp_ratio"] = (
            float(channel_table["lowfreq_resp_ratio"][lf_active].mean())
            if lf_active.any()
            else 0.0
        )
        row["max_lowfreq_tuning_metric"] = float(
            channel_table["lowfreq_tuning_metric"].max()
        )
        row["mean_lowfreq_power_snr"] = float(
            channel_table["lowfreq_power_snr"].mean()
        )

    return row
