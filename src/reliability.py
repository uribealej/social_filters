"""Trial-to-trial neuron reliability filtering."""

from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu

import src.plotting as plotting

def filter_neurons_by_trial_reliability(
    dfof: np.ndarray,                    # (T, n_neurons)
    trial_aligned_traces: dict,         # stim_id -> (n_neurons, win_length, n_trials)
    stimuli_ids: list,
    fps_2p: float,
    plots_path: Path,
    prefix: str = "",
    costom_threshold: float | None = None,
    folder_name: str | None = None,
    make_plots: bool = True,
    save_indices: bool = True,
    hist_bins: int = 70,
    custom_threshold: float | None = None,
):
    """

    Compute trial-to-trial reliability for each neuron, select neurons with
    high reliability using an Otsu threshold, optionally plot, and save
    kept neuron indices to disk.

    For each neuron and each stimulus condition, we compute the trial-to-trial
    reliability as the mean pairwise Pearson correlation across all trials
    in a peri-stimulus time window (based on trial_aligned_traces). This yields,
    for every neuron, one reliability value per stimulus. We then take, for each
    neuron, the maximum reliability across stimuli to obtain a single reliability
    score per neuron. To define an objective threshold separating reliable from
    unreliable neurons, we apply Otsu’s method on the distribution of these
    maximum reliability scores. Otsu’s method finds the threshold that best
    separates the distribution into two classes by maximizing the between-class
    variance. Neurons with a reliability score greater than or equal to this
    Otsu threshold are kept for subsequent analyses.

    Parameters
    ----------
    dfof : array, shape (T, n_neurons)
        ΔF/F traces over time (T) for each neuron.
    trial_aligned_traces : dict
        Mapping stim_id -> array of shape (n_neurons, win_length, n_trials),
        containing trial-aligned neural responses.
    stimuli_ids : list
        List of stimulus IDs present in trial_aligned_traces.
    fps_2p : float
        2P sampling rate in Hz, used to convert frames to seconds in rasters.
    plots_path : Path
        Base path used only to save kept neuron indices (.npy).
    prefix : str
        Optional string to prepend in the saved filename.
    custom_threshold : float or None
        Optional fixed reliability threshold. If None, Otsu's threshold is used.
    costom_threshold : float or None
        Deprecated spelling of custom_threshold, kept for older notebooks.
    filename_mid : str
        Optional middle part of the saved filename (e.g. session ID).
    folder_name : str or None
        Optional subfolder inside plots_path where indices are saved.
        If None, indices are saved directly in plots_path.
    make_plots : bool
        If True, show histogram, curve of #ROIs vs threshold, and accepted/
        rejected rasters. Plots are not saved to disk.
    save_indices : bool
        If True, save kept neuron indices as a .npy file in the chosen folder.
    hist_bins : int
        Number of bins to use for the reliability histogram.

    Returns
    -------
    result : dict
        Dictionary containing:
        - reliability_per_stim
        - max_stimuli_correlation
        - nanfiltered_max
        - otsu_threshold
        - kept_mask
        - kept_neuron_indices

    """

    T, n_neurons = dfof.shape
    if custom_threshold is not None and costom_threshold is not None:
        raise ValueError("Use only one of custom_threshold or deprecated costom_threshold")
    threshold_override = custom_threshold if custom_threshold is not None else costom_threshold

    # --- reliability per neuron x stimulus ---
    reliability_per_stim = np.full((n_neurons, len(stimuli_ids)), np.nan, dtype=float)

    for j, stim in enumerate(stimuli_ids):
        aligned_neural_traces = trial_aligned_traces[stim]  # (n_neurons, win_length, n_trials)
        _, _, n_trials = aligned_neural_traces.shape

        if n_trials < 2:
            # can't compute correlations with 1 trial → leave as NaN
            continue

        for i in range(n_neurons):
            neuron_trace = aligned_neural_traces[i, :, :]  # (time, trials)

            # Compute trial-to-trial correlation matrix (trials x trials)
            corr_mat = np.corrcoef(neuron_trace.T)

            # Ignore self-correlations by setting diagonal to NaN
            np.fill_diagonal(corr_mat, np.nan)

            # Mean correlation (reliability) across all trial pairs
            reliability_per_stim[i, j] = np.nanmean(corr_mat)

    # --- max reliability across stimuli & Otsu threshold ---
    max_stimuli_correlation = np.nanmax(reliability_per_stim, axis=1)
    nanfiltered_max = max_stimuli_correlation[np.isfinite(max_stimuli_correlation)]

    if nanfiltered_max.size > 0:
        otsu_threshold = float(threshold_otsu(nanfiltered_max))
    else:
        otsu_threshold = np.nan

    if threshold_override is not None:
        otsu_threshold = float(threshold_override)

    kept_mask = np.isfinite(max_stimuli_correlation) & (max_stimuli_correlation >= otsu_threshold)
    kept_neuron_indices = np.flatnonzero(kept_mask)
    kept_pct = (100.0 * kept_neuron_indices.size / n_neurons) if n_neurons else 0.0

    # --- save indices (but not figures) ---
    if save_indices:
        if plots_path is None:
            raise ValueError("plots_path must be provided if save_indices=True")

        # Decide where to save
        out_dir = plots_path / folder_name if folder_name is not None else plots_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save kept indices
        out_idx = out_dir / f"{prefix}_kept_neuron_indices.npy"
        np.save(out_idx, kept_neuron_indices)
        print(f"Saved kept neuron indices to: {out_idx}")

    # --- plots (purely for visualization, not saved) ---
    if make_plots and nanfiltered_max.size > 0:
        plotting.plot_reliability_diagnostics(
            dfof=dfof,
            nanfiltered_max=nanfiltered_max,
            otsu_threshold=otsu_threshold,
            kept_mask=kept_mask,
            kept_neuron_indices=kept_neuron_indices,
            fps_2p=fps_2p,
            hist_bins=hist_bins,
        )

    return {
        "reliability_per_stim": reliability_per_stim,
        "max_stimuli_correlation": max_stimuli_correlation,
        "nanfiltered_max": nanfiltered_max,
        "otsu_threshold": otsu_threshold,
        "kept_mask": kept_mask,
        "kept_neuron_indices": kept_neuron_indices,
    }
