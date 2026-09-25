"""Current V2 compatibility facade for significant-trace detection.

New analysis should import :mod:`src.significant_trace_detection`. This module
preserves the established V2 signatures while delegating to the canonical
implementation's ``current`` mode.
"""

from src import significant_trace_detection as _canonical


def gaussfit_neg(data, peak_threshold_fraction=0.2, trim_std=2.0):
    return _canonical.gaussfit_neg(
        data,
        peak_threshold_fraction=peak_threshold_fraction,
        trim_std=trim_std,
        mode=_canonical.CURRENT_MODE,
    )


def estimate_and_center_noise_model(deltaF_F, fit_function=gaussfit_neg, verbose=True):
    return _canonical.estimate_and_center_noise_model(
        deltaF_F,
        fit_function=fit_function,
        verbose=verbose,
    )


def normalize_dff(deltaF_center, sigma_vals, min_sigma=1e-12):
    return _canonical.normalize_dff(
        deltaF_center,
        sigma_vals,
        min_sigma=min_sigma,
        mode=_canonical.CURRENT_MODE,
    )


def extract_transition_points(norm_data):
    return _canonical.extract_transition_points(norm_data)


def estimate_kde_peak(points):
    return _canonical.estimate_kde_peak(points, mode=_canonical.CURRENT_MODE)


def generate_synthetic_noise(points, peak_x, peak_y, rng=None):
    return _canonical.generate_synthetic_noise(
        points,
        peak_x,
        peak_y,
        rng=rng,
        mode=_canonical.CURRENT_MODE,
    )


def create_grid(points, synthetic_noise, n_bins):
    return _canonical.create_grid(points, synthetic_noise, n_bins)


def histogram2d(points, bins, grid):
    return _canonical.histogram2d(points, bins, grid)


def compute_global_sigma(points, xev, yev, k_neighbors):
    return _canonical.compute_global_sigma(
        points,
        xev,
        yev,
        k_neighbors,
        mode=_canonical.CURRENT_MODE,
    )


def compute_significant_odds(
    points,
    density_data,
    density_noise,
    xev,
    yev=None,
    confCutOff=95,
    plotFlag=False,
    exclude_empty_bins=False,
):
    return _canonical.compute_significant_odds(
        points,
        density_data,
        density_noise,
        xev,
        yev=yev,
        confCutOff=confCutOff,
        plotFlag=plotFlag,
        exclude_empty_bins=exclude_empty_bins,
    )


def rasterize_with_odds(
    norm_data,
    mapOfOdds,
    xev,
    yev,
    fps,
    tauDecay,
    joint_map_mode="v1",
):
    return _canonical.rasterize_with_odds(
        norm_data,
        mapOfOdds,
        xev,
        yev,
        fps,
        tauDecay,
        joint_map_mode=joint_map_mode,
        mode=_canonical.CURRENT_MODE,
    )


def compute_noise_model_romano_fast_modular(
    deltaF_F,
    n_bins=1000,
    k_neighbors=100,
    confCutOff=95,
    plot_odds=False,
    fps=2.0,
    tauDecay=6.0,
    random_state=None,
    return_diagnostics=False,
    smoothing_mode="v1",
    joint_map_mode="v1",
    exclude_empty_bins=False,
):
    return _canonical.compute_noise_model_romano_fast_modular(
        deltaF_F,
        n_bins=n_bins,
        k_neighbors=k_neighbors,
        confCutOff=confCutOff,
        plot_odds=plot_odds,
        fps=fps,
        tauDecay=tauDecay,
        random_state=random_state,
        return_diagnostics=return_diagnostics,
        smoothing_mode=smoothing_mode,
        joint_map_mode=joint_map_mode,
        exclude_empty_bins=exclude_empty_bins,
        mode=_canonical.CURRENT_MODE,
    )


def plot_dff_and_raster(deltaF_center, raster, fps=2.0, vmax_dff=0.15, title=None):
    return _canonical.plot_dff_and_raster(
        deltaF_center,
        raster,
        fps=fps,
        vmax_dff=vmax_dff,
        title=title,
    )


def compare_significant_trace_versions(
    deltaF_F,
    n_bins=1000,
    k_neighbors=100,
    confCutOff=95,
    plot_odds=False,
    fps=2.0,
    tauDecay=6.0,
    random_state=None,
    vmax_dff=0.15,
    include_difference=True,
    title=None,
    show=True,
    smoothing_mode="v1",
    joint_map_mode="v1",
    exclude_empty_bins=False,
):
    return _canonical.compare_significant_trace_versions(
        deltaF_F,
        n_bins=n_bins,
        k_neighbors=k_neighbors,
        confCutOff=confCutOff,
        plot_odds=plot_odds,
        fps=fps,
        tauDecay=tauDecay,
        random_state=random_state,
        vmax_dff=vmax_dff,
        include_difference=include_difference,
        title=title,
        show=show,
        smoothing_mode=smoothing_mode,
        joint_map_mode=joint_map_mode,
        exclude_empty_bins=exclude_empty_bins,
    )


__all__ = [
    "compare_significant_trace_versions",
    "compute_global_sigma",
    "compute_noise_model_romano_fast_modular",
    "compute_significant_odds",
    "create_grid",
    "estimate_and_center_noise_model",
    "estimate_kde_peak",
    "extract_transition_points",
    "gaussfit_neg",
    "generate_synthetic_noise",
    "histogram2d",
    "normalize_dff",
    "plot_dff_and_raster",
    "rasterize_with_odds",
]
