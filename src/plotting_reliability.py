"""Reliability selection and response-overlap diagnostic figures."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def plot_accepted_rejected_rasters(
    dfof: np.ndarray,             # (n_neurons, n_frames)
    t=None,                       # (n_frames,) OR None OR scalar dt
    kept_mask: np.ndarray=None,   # (n_neurons,), boolean
    vmax: float = None,
    vmin: float = 0.0,
    perc_for_vmax: float = 99.0,
    sort_by_peak_time: bool = False,
    share_color_scale: bool = True,
):
    assert dfof.ndim == 2, "dfof must be (n_neurons, n_frames)"
    n_neurons, n_frames = dfof.shape

    # --- Build time axis / extent ---
    if t is None:
        x0, x1 = 0.0, float(n_frames - 1)
        x_label = "Frame"
    elif np.isscalar(t):  # t is a sampling interval (dt in seconds)
        dt = float(t)
        x0, x1 = 0.0, dt * (n_frames - 1)
        x_label = "Time (s)"
    else:
        t = np.asarray(t)
        assert t.ndim == 1 and t.size == n_frames, "t must be 1-D with length n_frames"
        x0, x1 = float(t[0]), float(t[-1])
        x_label = "Time"

    if kept_mask is None:
        kept_mask = np.ones(n_neurons, dtype=bool)
    else:
        kept_mask = np.asarray(kept_mask, dtype=bool)
        assert kept_mask.shape[0] == n_neurons, "kept_mask length must match n_neurons"

    kept_idx = np.flatnonzero(kept_mask)
    rej_idx  = np.setdiff1d(np.arange(n_neurons), kept_idx)

    # --- Color scaling ---
    if vmax is None:
        finite_vals = dfof[np.isfinite(dfof)]
        vmax = np.percentile(finite_vals, perc_for_vmax) if finite_vals.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(dfof) if np.isfinite(np.nanmax(dfof)) else 1.0

    vmax_kept = vmax
    vmax_rej  = vmax
    if not share_color_scale:
        if kept_idx.size:
            tmp = np.percentile(dfof[kept_idx], perc_for_vmax)
            vmax_kept = tmp if np.isfinite(tmp) and tmp > 0 else vmax
        if rej_idx.size:
            tmp = np.percentile(dfof[rej_idx], perc_for_vmax)
            vmax_rej  = tmp if np.isfinite(tmp) and tmp > 0 else vmax

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    def mat_for(idx):
        M = dfof[idx] if idx.size else np.zeros((1, n_frames))
        if sort_by_peak_time and idx.size > 1:
            order = np.argsort(np.argmax(M, axis=1))
            M = M[order]
        return M

    for (title, idx, vmax_here, ax) in [
        ("Accepted", kept_idx, vmax_kept, axes[0]),
        ("Rejected", rej_idx,  vmax_rej,  axes[1]),
    ]:
        M = mat_for(idx)
        im = ax.imshow(
            M,
            aspect='auto',
            interpolation='nearest',
            origin='lower',
            extent=[x0, x1, 0, M.shape[0]],
            vmin=vmin,
            vmax=vmax_here,
            cmap='gray_r',       # high = dark
        )
        ax.set_title(f"{title} (n={idx.size})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Neuron #")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("ΔF/F")

    if share_color_scale:
        fig.suptitle(f"ΔF/F rasters (shared vmin={vmin:.3g}, vmax={vmax:.3g})", y=1.02)
    else:
        fig.suptitle(
            f"ΔF/F rasters (vmin={vmin:.3g}, kept vmax={vmax_kept:.3g}, rejected vmax={vmax_rej:.3g})",
            y=1.02
        )
    return fig, axes


def plot_venn_3stim(
    response_type_by_id,
    stim_ids=(5, 6, 7),
    stim_labels=None,          # NEW: optional list of names, one per stim_id
    title=None,                # NEW: optional custom title
):
    # 1) Boolean masks: neuron is responsive (1 or 2) to each stim
    resp = {sid: (response_type_by_id[sid] != 0) for sid in stim_ids}

    # 2) Convert to sets of neuron indices
    sets = [set(np.where(resp[sid])[0]) for sid in stim_ids]
    A, B, C = sets
    s1, s2, s3 = stim_ids

    # 3) Print counts per region
    only_A = A - B - C
    only_B = B - A - C
    only_C = C - A - B
    AB_only = (A & B) - C
    AC_only = (A & C) - B
    BC_only = (B & C) - A
    ABC     = A & B & C

    print(f"Only {s1}: {len(only_A)}")
    print(f"Only {s2}: {len(only_B)}")
    print(f"Only {s3}: {len(only_C)}")
    print(f"{s1} & {s2} only: {len(AB_only)}")
    print(f"{s1} & {s3} only: {len(AC_only)}")
    print(f"{s2} & {s3} only: {len(BC_only)}")
    print(f"{s1} & {s2} & {s3}: {len(ABC)}")

    # --- NEW: build labels from stim_labels if provided ---
    if stim_labels is None:  # fallback: just use the IDs
        stim_labels = [f"stim {s1}", f"stim {s2}", f"stim {s3}"]
    else:
        assert len(stim_labels) == 3, "stim_labels must have length 3"

    # --- Plot Venn diagram ---
    plt.figure(figsize=(5, 5))
    venn3(sets, set_labels=stim_labels)

    if title is None:  # NEW: nicer default title using names
        title = "Responsive neurons to: " + ", ".join(stim_labels)
    plt.title(title)

    plt.show()


def plot_reliability_diagnostics(
    dfof,
    nanfiltered_max,
    otsu_threshold,
    kept_mask,
    kept_neuron_indices,
    fps_2p,
    hist_bins=70,
):
    """Render the three diagnostics used by reliability filtering."""
    n_neurons = dfof.shape[1]
    kept_pct = (100.0 * kept_neuron_indices.size / n_neurons) if n_neurons else 0.0

    histogram_figure, histogram_axis = plt.subplots(figsize=(6, 4))
    histogram_axis.hist(nanfiltered_max, bins=hist_bins)
    histogram_axis.axvline(otsu_threshold, linestyle="--", color="k")
    histogram_axis.set(
        title=(
            "Reliability of response across stimuli\n"
            f"Otsu {otsu_threshold:.2f} — kept {kept_pct:.1f}% "
            f"({kept_neuron_indices.size} ROIs)"
        ),
        xlabel="max avg intertrial correlation",
        ylabel="neuron count",
    )
    histogram_axis.spines["top"].set_visible(False)
    histogram_axis.spines["right"].set_visible(False)
    histogram_figure.tight_layout()
    plt.show()

    low = max(0.0, float(np.nanmin(nanfiltered_max)))
    high = min(1.0, max(float(np.nanmax(nanfiltered_max)), low + 1e-6))
    threshold_grid = np.linspace(low, high, 51)
    counts = [np.sum(nanfiltered_max >= threshold) for threshold in threshold_grid]

    curve_figure, curve_axis = plt.subplots(figsize=(6, 4))
    curve_axis.plot(threshold_grid, counts, marker="o", linewidth=1)
    curve_axis.set(
        title="ROIs kept vs threshold",
        xlabel="Threshold",
        ylabel="# ROIs kept",
    )
    curve_axis.spines["top"].set_visible(False)
    curve_axis.spines["right"].set_visible(False)
    curve_figure.tight_layout()
    plt.show()

    raster_figure, raster_axes = plot_accepted_rejected_rasters(
        dfof=dfof.T,
        t=1.0 / float(fps_2p),
        kept_mask=kept_mask,
        sort_by_peak_time=True,
        share_color_scale=True,
    )
    plt.show()
    return {
        "histogram": (histogram_figure, histogram_axis),
        "threshold_curve": (curve_figure, curve_axis),
        "rasters": (raster_figure, raster_axes),
    }
