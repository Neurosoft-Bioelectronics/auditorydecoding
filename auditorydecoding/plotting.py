from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def plot_pca_variance(pca, log_scale=False):
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance ratio")
    plt.bar(
        range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_
    )
    if log_scale:
        plt.gca().set_yscale("log")
    else:
        plt.gca().set_ylim(0)

    plt.show()


def plot_signal(signal, start=0, end=1, chann_names=None, fs=1000):
    """
    Plots the first `seconds` of all 24 channels with separate y locations (stacked traces),
    keeping scaling consistent between all channels.

    Parameters:
    - signal: numpy array of shape (time, 24)
    - seconds: number of seconds to plot
    - fs: sampling rate in Hz (default 1000)
    """
    num_samples = int((end - start) * fs)
    start_sample = int(start * fs)
    end_sample = int(end * fs)
    t = (np.arange(num_samples) / fs) + start
    n_channels = signal.shape[1]
    data = signal[start_sample:end_sample, :]

    y_min = np.min(data)
    y_max = np.max(data)
    y_range = y_max - y_min

    offset = y_range * 0.25

    plt.figure(figsize=(8, 0.25 * n_channels))
    for ch in range(n_channels):
        plt.plot(
            t,
            data[:, ch] + offset * ch,
            label=chann_names[ch] if chann_names is not None else f"ch {ch}",
            alpha=0.7,
        )

    plt.xlabel("Time (s)")
    if chann_names is not None:
        plt.yticks([offset * ch for ch in range(n_channels)], chann_names)
    else:
        plt.yticks(
            [offset * ch for ch in range(n_channels)],
            [f"ch {ch}" for ch in range(n_channels)],
        )
    plt.ylabel("Channel")
    plt.title(f"First {end - start} seconds of all 24 channels (stacked)")
    plt.tight_layout()
    plt.gca().set_xlim(start, end)
    plt.show()


def plot_covariance(cov_matrix):
    plt.figure(figsize=(5, 4))
    plt.imshow(cov_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Covariance")
    plt.xlabel("Channel Index")
    plt.ylabel("Channel Index")
    plt.title("Covariance Matrix")


def animate_pca_timeseries(data, interval=100):
    """
    Animates a 2D PCA timeseries.

    Parameters:
    - data: np.ndarray of shape (N, 2)
    - interval: delay between frames in milliseconds
    """
    n_frames = len(data)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set plot limits based on data range
    ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
    ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)
    ax.set_title("PCA Timeseries Evolution")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Initialize empty scatter plot
    # We use viridis (cmap) and map colors to the index of the point
    scat = ax.scatter(
        [], [], c=[], cmap="viridis", vmin=0, vmax=n_frames, edgecolor="k", s=50
    )
    plt.colorbar(scat, label="Timepoint (Index)")

    def update(frame):
        # Update the positions: slice data up to the current frame
        scat.set_offsets(data[: frame + 1])

        # Update colors: provide an array of indices for the colormap
        scat.set_array(np.arange(frame + 1))

        return (scat,)

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True
    )

    # Close the plot to prevent a static ghost image from showing up
    plt.close()

    return HTML(ani.to_html5_video())
