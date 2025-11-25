"""
Title: Romano-style noise model & rasterization for calcium imaging
Author: Alejandro Uribe

Overview
--------
This script takes ΔF/F traces from a Suite2p plane and generates:
1) A *centered* ΔF/F matrix (baseline-corrected per ROI).
2) A binary *raster* of significant neural events.
3) Diagnostic transition-density objects (real vs. synthetic noise) and a
   significance map ("mapOfOdds") indicating where real dynamics exceed noise.

The method follows the Romano et al. idea: compare the empirical density of
frame-to-frame transitions ΔF/F(t) → ΔF/F(t+1) to a noise model, then keep
only transitions that are statistically unlikely to be noise and also satisfy
simple biophysical constraints (decay/rise). A two-panel plot visualizes
centered ΔF/F and the event raster with the same neuron ordering.

Data flow (high level)
----------------------
Input:  dFoF.npy (shape T × N), one column per ROI.

1) Per-ROI noise fit (gaussfit_neg):
   - For each ROI trace, estimate noise by fitting a Gaussian to the negative
     half of the KDE around the mode (assumes noise dominates negative side).
   - If the quadratic log-fit fails, fall back to a robust std within ±k·std.

2) Centering & normalization:
   - Subtract the fitted mean μ from each ROI → deltaF_center (T × N).
   - Normalize by σ to obtain z-scored traces (used for transitions).

3) Transition density estimation:
   - Build a large point cloud of (ΔF/F(t), ΔF/F(t+1)) across all ROIs/time.
   - Estimate the distribution of these transitions on a 2D grid.
   - Build a *synthetic noise* sample from the “negative quadrant” around
     the mode (lower activity region), matching mean/covariance.
   - Histogram both real and synthetic points on a shared grid and apply
     Gaussian smoothing; smoothing σ is chosen adaptively from k-NN distances
     on the grid for each distribution.

4) Significance map ("mapOfOdds"):
   - Mark grid bins where real density > (1 − confCutOff/100) · noise density.
     (e.g., confCutOff=95 → bins where real density > 5% of noise density).
   - Optionally visualize this mask over the (t, t+1) point cloud.

5) Biophysical constraints & event rasterization:
   - Remove the baseline “blob” around zero transitions (connected component).
   - Apply a decay constraint based on fps and a calcium decay constant
     (tauDecay): exclude transitions that decay implausibly fast.
   - Apply a simple rise cap to discard extreme upward jumps at the top row.
   - Rasterization: mark timepoints as events when short 2–3 point transition
     patterns fall within the *joint* significance map.

6) Visualization:
   - `plot_dff_and_raster()` shows two aligned heatmaps:
       top: centered ΔF/F (ΔF/F − μ), bottom: binary raster.
     Neurons are sorted once via PCA on activity so the panels line up.

Key functions & outputs
-----------------------
- `gaussfit_neg(trace) -> (sigma, mu, A, used_fallback)`
  Negative-half KDE Gaussian fit with robust fallback.

- `estimate_and_center_noise_model(deltaF_F) -> (deltaF_center, sigma_vals, mu_vals, A_vals, flags)`
  Per-ROI noise fits and centering.

- `compute_noise_model_romano_fast_modular(deltaF_F, ...) -> (mapOfOdds, deltaF_center, density_data, density_noise, xev, yev, raster, mapOfOddsJoint)`
  Full Romano-style pipeline (no plotting). Returns:
    • mapOfOdds: real>noise significance mask on the transition grid
    • deltaF_center: centered ΔF/F (T × N)
    • density_data/noise: smoothed histograms
    • xev, yev: grid coordinates
    • raster: binary events (T × N)
    • mapOfOddsJoint: significance map after baseline/decay/rise constraints

- `plot_dff_and_raster(deltaF_center, raster, ...) -> sort_idx`
  Two-panel figure (centered ΔF/F and raster) with consistent neuron order.

Important parameters
--------------------
- `n_bins`: grid resolution for histograms (trade-off: detail vs. memory).
- `k_neighbors`: k-NN used to set smoothing σ; larger → smoother densities.
- `confCutOff`: confidence (%). 95 means keep bins where real density exceeds
  5% of noise density (more conservative → fewer events).
- `fps`, `tauDecay`: shape decay constraint; set according to imaging rate and indicator kinetics.

Assumptions & caveats
---------------------
- The negative side of each ROI’s ΔF/F distribution is dominated by noise.
- The synthetic-noise model is built from the low-activity (negative quadrant)
  region; if too few points exist there, covariance estimation can fail.
- Event detection relies on short transition patterns; it is not a full
  deconvolution and will mark “transients,” not spikes.
- Large `n_bins` and dense datasets can be memory/time heavy; tune `n_bins`
  and consider downsampling time or ROIs if needed.

Minimal usage (as in this script)
---------------------------------
1) Load ΔF/F:
   `deltaF_F = np.load(plane_dir / "dFoF.npy")`
2) Run Romano pipeline:
   `mapOfOdds, deltaF_center, density_data, density_noise, xev, yev, raster, mapOfOddsJoint = compute_noise_model_romano_fast_modular(deltaF_F, ...)`
3) Plot:
   `sort_idx = plot_dff_and_raster(deltaF_center, raster, fps=2.0, vmax_dff=0.3)`

References
----------
- Romano et al. (2017). Transition-based significance mapping for calcium
  imaging (methodological inspiration). Adapted and simplified here.
"""
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from skimage.measure import label, regionprops
from pathlib import Path
import matplotlib.pyplot as plt

#%%
def gaussfit_neg(data, peak_threshold_fraction=0.2, trim_std=2.0):
    """
    Robust Gaussian fitting of the negative half of a KDE of the input data.

    This method is especially suitable for estimating the noise characteristics
    of ΔF/F traces from calcium imaging, where noise is assumed to dominate
    the negative side of the distribution.

    Parameters:
    -----------
    data : ndarray
        1D array of ΔF/F values for a single ROI over time.
    peak_threshold_fraction : float, optional (default=0.2)
        Fraction of the KDE peak height to include in the fit.
        Values below this threshold are excluded from the Gaussian fit.
    trim_std : float, optional (default=2.0)
        If the fit fails, fallback uses data within ±`trim_std` standard deviations
        to compute robust sigma.

    Returns:
    --------
    sigma : float
        Estimated standard deviation of the noise.
    mu : float
        Estimated mean of the Gaussian fit.
    A : float or np.nan
        Estimated amplitude of the Gaussian. If fallback is used, A is np.nan.
    used_fallback : bool
        True if the fallback method was used due to poor fit.
    """

    N = len(data)

    # Step 1: Estimate the KDE of the input data
    kde = gaussian_kde(data)
    x = np.linspace(np.min(data), np.max(data), 200)
    y = kde(x)

    # Step 2: Find the mode of the distribution (peak of KDE)
    ind_peak = np.argmax(y)

    # Step 3: Use only the left half of the distribution (assumed to be noise-dominated)
    x_fit = x[:ind_peak + 1]
    y_fit = y[:ind_peak + 1]

    # Keep only the part of the curve above a fraction of the peak
    ymax = np.max(y_fit)
    keep = y_fit > ymax * peak_threshold_fraction
    x_fit = x_fit[keep]
    y_fit = y_fit[keep] / N  # normalize to probability density

    # Abort if insufficient data to fit
    if len(x_fit) < 3:
        return np.nan, np.nan, np.nan, True

    try:
        # Step 4: Fit log(y) = a2 x^2 + a1 x + a0
        ylog = np.log(y_fit)
        coeffs = np.polyfit(x_fit, ylog, 2)
        A2, A1, A0 = coeffs

        # Validate parabola direction
        if A2 >= 0:
            raise ValueError("Parabola opens upwards — not a valid Gaussian")

        sigma = np.sqrt(-1 / (2 * A2))
        mu = A1 * sigma ** 2
        A = np.exp(A0 + mu ** 2 / (2 * sigma ** 2))

        if not np.isreal(sigma):
            raise ValueError("Sigma is complex")

        return float(sigma), float(mu), float(A), False  # fit successful

    except Exception:
        # Fallback: use robust std within ±trim_std
        dev = np.nanstd(data)
        mask = np.abs(data) <= trim_std * dev
        trimmed = data[mask]
        sigma = np.nanstd(trimmed)
        mu = np.nanmean(trimmed)
        return float(sigma), float(mu), np.nan, True  # fallback used



#%%
def estimate_and_center_noise_model(deltaF_F, fit_function=gaussfit_neg, verbose=True):
    """
    Fit Gaussian noise model for each ROI using the negative half of the ΔF/F distribution,
    then center each ROI's trace by subtracting the estimated mean (μ).

    This function is typically used after preprocessing (e.g., ΔF/F calculation)
    and helps prepare the data for statistical modeling or event detection.

    Parameters
    ----------
    deltaF_F : ndarray of shape (T, N)
        Preprocessed ΔF/F traces with T timepoints and N ROIs.
    fit_function : callable, optional
        Fitting function to use for noise modeling (default: gaussfit_neg).
        Should return (σ, μ, A, used_fallback) per ROI.
    verbose : bool
        Whether to print fitting statistics (e.g., number of fallback uses).

    Returns
    -------
    deltaF_centered : ndarray of shape (T, N)
        ΔF/F data with estimated μ subtracted per ROI.
    sigma_vals : ndarray of shape (N,)
        Estimated noise standard deviation (σ) per ROI.
    mu_vals : ndarray of shape (N,)
        Estimated Gaussian mean (μ) per ROI.
    A_vals : ndarray of shape (N,)
        Estimated Gaussian amplitude (A) per ROI.
    fallback_flags : list of bool
        List indicating whether each ROI required fallback estimation.
    """
    T, N = deltaF_F.shape
    sigma_vals = np.zeros(N)
    mu_vals = np.zeros(N)
    A_vals = np.zeros(N)
    fallback_flags = []

    # --- Fit noise model per ROI ---
    for i in range(N):
        trace = deltaF_F[:, i]
        sigma, mu, A, used_fallback = fit_function(trace)
        sigma_vals[i] = sigma
        mu_vals[i] = mu
        A_vals[i] = A
        fallback_flags.append(used_fallback)

    # --- Subtract estimated noise mean from each trace ---
    deltaF_centered = deltaF_F - mu_vals  # Broadcast subtraction across ROIs

    if verbose:
        n_fallback = sum(fallback_flags)
        print(f"[Noise Fit] Used fallback on {n_fallback} out of {N} ROIs "
              f"({100 * n_fallback / N:.1f}%)")

    return deltaF_centered, sigma_vals, mu_vals, A_vals, fallback_flags


def normalize_dff(deltaF_center, sigma_vals):
    """
    Normalize ΔF/F traces by dividing each neuron trace by its estimated noise level (σ).

    Args:
        deltaF_center (ndarray): ΔF/F matrix, shape (T, N)
        sigma_vals (ndarray): Noise levels, shape (N,)

    Returns:
        ndarray: Normalized ΔF/F matrix
    """
    return deltaF_center / sigma_vals

def extract_transition_points(norm_data):
    """
    Create a 2D point cloud of transitions between ΔF/F(t) and ΔF/F(t+1) for all neurons.

    Args:
        norm_data (ndarray): Normalized ΔF/F, shape (T, N)

    Returns:
        ndarray: 2D points of transitions, shape (~T*N, 2)
    """
    x = norm_data[:-1, :].flatten()
    y = norm_data[1:, :].flatten()
    valid = ~np.isnan(x) & ~np.isnan(y)
    return np.vstack((x[valid], y[valid])).T

def estimate_kde_peak(points):
    """
    Estimate the peak (mode) of ΔF/F(t) and ΔF/F(t+1) using KDE.

    Args:
        points (ndarray): 2D transitions, shape (N, 2)

    Returns:
        tuple: (peak_x, peak_y)
    """
    kde_x = gaussian_kde(points[:, 0])
    kde_y = gaussian_kde(points[:, 1])
    xi = np.linspace(np.min(points), np.max(points), 200)
    peak_x = xi[np.argmax(kde_x(xi))]
    peak_y = xi[np.argmax(kde_y(xi))]
    return peak_x, peak_y

def generate_synthetic_noise(points, peak_x, peak_y):
    """
    Create synthetic noise points from the lower-left (negative) quadrant of the data,
    using a multivariate Gaussian with matching mean and covariance.

    Args:
        points (ndarray): Original transition points, shape (N, 2)
        peak_x (float): Estimated mode of ΔF/F(t)
        peak_y (float): Estimated mode of ΔF/F(t+1)

    Returns:
        ndarray: Synthetic noise points, shape (M, 2)
    """
    points_neg = points[(points[:, 0] < peak_x) & (points[:, 1] < peak_y)]
    mu_noise = [peak_x, peak_y]
    if len(points_neg) < 2:
        raise ValueError("Not enough points to compute covariance for noise model.")
    cov_noise = np.cov(points_neg, rowvar=False)
    synthetic_noise = np.random.multivariate_normal(mu_noise, cov_noise, size=2 * len(points_neg))
    return synthetic_noise

def create_grid(points, synthetic_noise, n_bins):
    """
    Create a shared 2D grid for histogramming data and synthetic noise.

    Args:
        points (ndarray): Real data points, shape (N, 2)
        synthetic_noise (ndarray): Synthetic noise points, shape (M, 2)
        n_bins (int): Number of bins for histogram in each dimension

    Returns:
        tuple: (grid, xev, yev)
    """
    combined = np.concatenate([points.flatten(), synthetic_noise.flatten()])
    vmin, vmax = np.floor(np.min(combined)), np.ceil(np.max(combined))
    grid = np.linspace(vmin, vmax, n_bins)
    xev, yev = np.meshgrid(grid, grid)
    return grid, xev, yev

def histogram2d(points, bins, grid):
    """
    Create a 2D histogram of transition points based on a shared grid.

    Args:
        points (ndarray): Points to bin, shape (N, 2)
        bins (int): Number of bins
        grid (ndarray): Bin edges

    Returns:
        ndarray: 2D histogram, shape (bins, bins)
    """
    xi = np.searchsorted(grid, points[:, 0]) - 1
    yi = np.searchsorted(grid, points[:, 1]) - 1
    xi = np.clip(xi, 0, bins - 1)
    yi = np.clip(yi, 0, bins - 1)
    hist = np.zeros((bins, bins))
    for i in range(len(xi)):
        hist[yi[i], xi[i]] += 1
    return hist

def compute_global_sigma(points, xev, yev, k_neighbors):
    """
    Estimate the global Gaussian smoothing sigma by computing average distances
    to the k-nearest real data points from each histogram bin location.

    Args:
        points (ndarray): Real data transition points, shape (N, 2)
        xev (ndarray): Meshgrid x-coordinates
        yev (ndarray): Meshgrid y-coordinates
        k_neighbors (int): Number of neighbors for density estimation

    Returns:
        float: Estimated sigma for Gaussian filter
    """
    grid_points = np.column_stack([xev.ravel(), yev.ravel()])
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(points)
    dists = nn.kneighbors(grid_points, return_distance=True)[0]
    bin_spread = dists.mean(axis=1).reshape(xev.shape)
    return np.median(bin_spread)

# def plot_density_comparison(density_data, density_noise, xev, yev, show=True):
#     """
#     Plot side-by-side heatmaps of real vs synthetic noise densities.
#
#     Args:
#         density_data (ndarray): Smoothed density of real ΔF/F transitions
#         density_noise (ndarray): Smoothed density of synthetic noise
#         xev (ndarray): Meshgrid x-coordinates
#         yev (ndarray): Meshgrid y-coordinates
#         show (bool): Whether to display the plots
#     """
#     if not show:
#         return
#
#     extent = [xev.min(), xev.max(), yev.min(), yev.max()]
#     fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#
#     im1 = axs[0].imshow(density_data.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
#     axs[0].set_title("Real ΔF/F Transitions Density")
#     axs[0].set_xlabel("ΔF/F(t)")
#     axs[0].set_ylabel("ΔF/F(t+1)")
#     plt.colorbar(im1, ax=axs[0], label='Density')
#
#     im2 = axs[1].imshow(density_noise.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
#     axs[1].set_title("Synthetic Noise Density")
#     axs[1].set_xlabel("ΔF/F(t)")
#     axs[1].set_ylabel("ΔF/F(t+1)")
#     plt.colorbar(im2, ax=axs[1], label='Density')
#
#     plt.tight_layout()
    # plt.show()

def compute_significant_odds(points, density_data, density_noise, xev, yev=None, confCutOff=95, plotFlag=False):
    """
    Compute the significance mask where real transition density exceeds noise density by a confidence threshold.

    Args:
        points (ndarray): 2D transition points (ΔF/F(t), ΔF/F(t+1)), shape (M, 2)
        density_data (ndarray): Smoothed histogram of real transitions
        density_noise (ndarray): Smoothed histogram of synthetic noise transitions
        xev (ndarray): Meshgrid X-coordinates
        yev (ndarray, optional): Meshgrid Y-coordinates (for correct plotting)
        confCutOff (float): Confidence level for thresholding (default 95)
        plotFlag (bool): Whether to plot the significance map

    Returns:
        mapOfOdds (ndarray): Boolean mask (same shape as density_data) indicating significant bins
    """
    pCutOff = (100 - confCutOff) / 100.0
    mapOfOdds = density_noise <= (pCutOff * density_data)

    if plotFlag:
        if yev is None:
            yev = xev  # fallback if symmetric grid

        extent = [xev.min(), xev.max(), yev.min(), yev.max()]
        plt.figure(figsize=(6, 6))
        plt.plot(points[:, 0], points[:, 1], 'k.', alpha=0.2, markersize=1)
        plt.imshow(mapOfOdds.T, extent=extent, origin='lower', aspect='auto', alpha=0.6, cmap='Reds')
        plt.xlabel('z-transformed ΔF/F @ t', fontsize=14)
        plt.ylabel('z-transformed ΔF/F @ t+1', fontsize=14)
        plt.title(f'Significant Odds Mask (>{confCutOff}% confidence)')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    return mapOfOdds

def rasterize_with_odds(norm_data, mapOfOdds, xev, yev, fps, tauDecay):
    """
    Rasterize ΔF/F activity using statistically significant transition maps,
    adapted from Romano et al. (2017).

    This function identifies significant neural transients by intersecting
    statistical significance (mapOfOdds) with biologically plausible rise/decay
    constraints, and uses 3-point temporal patterns to mark transients.

    Parameters
    ----------
    norm_data : ndarray, shape (T, N)
        Z-scored ΔF/F traces (i.e., ΔF/F divided by per-neuron sigma).
    mapOfOdds : ndarray of bool, shape (B, B)
        Boolean matrix indicating statistically significant transitions from ΔF/F(t) to ΔF/F(t+1).
    xev, yev : ndarray, shape (B, B)
        Meshgrids used for binning ΔF/F(t) and ΔF/F(t+1) values.
    fps : float
        Sampling rate (frames per second).
    tauDecay : float
        Expected calcium decay time constant (in seconds).

    Returns
    -------
    raster : ndarray of int, shape (T, N)
        Binary matrix indicating significant neural transients (1 = event).
    mapOfOddsJoint : ndarray of bool, shape (B, B)
        Refined joint transition map after applying baseline, decay, and rise filters.
    """
    T, N = norm_data.shape
    transfDataMatrix = norm_data

    # --- Step 1: Remove baseline noise blob (connected to 0,0) ---
    mask = ~mapOfOdds  # noise regions
    labeled = label(mask)  # label connected regions
    props = regionprops(labeled)
    indZero = np.argmax(xev[0, :] > 0)  # bin closest to ΔF/F = 0

    region_index = None
    for i, region in enumerate(props):
        if any(p[1] == indZero for p in region.coords):
            region_index = region.label
            break
    # Create corrected map: keep all transitions except baseline blob
    mapOfOddsCorrected = np.ones_like(mapOfOdds, dtype=bool)
    if region_index is not None:
        mapOfOddsCorrected[labeled == region_index] = 0

    # Step 2: Build decay map — exclude transitions faster than biologically plausible decay
    noiseBias = 1.0  # controls strictness of decay exclusion
    factorDecay = np.exp(-1 / (fps * tauDecay))  # expected decay factor per frame
    decayMap = np.ones_like(mapOfOdds, dtype=bool)
    for col in range(mapOfOdds.shape[1]):
        # Remove transitions where ΔF/F(t+1) decays too steeply from ΔF/F(t)
        mask = yev[:, 0] < factorDecay * (xev[0, col] - noiseBias) - noiseBias
        decayMap[mask, col] = 0

   # Step 3: Build rise map — remove extreme up transitions at top of the grid
    riseMap = np.ones_like(mapOfOdds, dtype=bool)
    riseMap[-1, :] = 0  # Discard transitions ending at the highest ΔF/F(t+1) bin

    # Step 4: Final joint significance map
    mapOfOddsJoint = mapOfOddsCorrected & decayMap & riseMap

    # Step 5: Raster generation
    raster = np.zeros_like(transfDataMatrix, dtype=int)
    bin_edges = xev[0, :]  # bin values used to digitize ΔF/F traces

    # For each neuron
    for n in range(N):
        trace = transfDataMatrix[:, n]
        bins = np.digitize(trace, bin_edges) - 1  # Convert ΔF/F values to bin indices

        for t in range(2, T - 2):  # Leave padding for 2-point transitions
            try:
                # Test three 3-point transition patterns
                optA = mapOfOddsJoint[bins[t + 1], bins[t]] and mapOfOddsJoint[bins[t], bins[t - 1]]
                optB = mapOfOddsJoint[bins[t], bins[t - 1]] and mapOfOddsJoint[bins[t - 1], bins[t - 2]]
                optC = mapOfOddsJoint[bins[t + 2], bins[t + 1]] and mapOfOddsJoint[bins[t + 1], bins[t]]
                if optA or optB or optC:
                    raster[t, n] = 1  # Mark as event
            except IndexError:
                continue  # Skip if bin index out of bounds

    return raster, mapOfOddsJoint

#%%
def compute_noise_model_romano_fast_modular(
    deltaF_F,
    n_bins=1000,
    k_neighbors=100,
    confCutOff=95,
    plot_odds=False,
    # plot_density=False,
    fps=2.0,
    tauDecay=6.0,

):

    deltaF_center, sigma_vals, mu_vals, A_vals, fallback_flags = estimate_and_center_noise_model(
        deltaF_F
    )
    # --- Step 1: Normalize ΔF/F using per-neuron σ values
    norm_data = normalize_dff(deltaF_center, sigma_vals)

    # --- Step 2: Extract point cloud of ΔF/F(t), ΔF/F(t+1) transitions
    points = extract_transition_points(norm_data)

    # --- Step 3: Estimate the peak of KDE in each dimension to locate mode
    peak_x, peak_y = estimate_kde_peak(points)

    # --- Step 4: Generate synthetic noise from the negative quadrant
    # Modeled as a multivariate Gaussian fit to low activity region
    synthetic_noise = generate_synthetic_noise(points, peak_x, peak_y)

    # --- Step 5: Create a common 2D grid for histogramming real and noise data
    grid, xev, yev = create_grid(points, synthetic_noise, n_bins)

    # --- Step 6: Bin the real data and synthetic noise into histograms
    hist_data = histogram2d(points, n_bins, grid)
    hist_noise = histogram2d(synthetic_noise, n_bins, grid)

    # --- Step 7: Estimate adaptive smoothing scales via KNN on each distribution
    sigma_data = compute_global_sigma(points, xev, yev, k_neighbors)
    sigma_noise = compute_global_sigma(synthetic_noise, xev, yev, k_neighbors)

    # --- Step 8: Apply Gaussian smoothing to both histograms
    density_data = gaussian_filter(hist_data, sigma=sigma_data)
    density_noise = gaussian_filter(hist_noise, sigma=sigma_noise)
    eps = 1e-12
    density_data = density_data / (density_data.sum() + eps)
    density_noise = density_noise / (density_noise.sum() + eps)
    # --- Step 9: Optionally show real vs noise density comparison
    # if plot_density:
    #     plot_density_comparison(density_data, density_noise, xev, yev, show=True)

    # --- Step 10: Build a binary significance map (real density > noise threshold)
    mapOfOdds = compute_significant_odds(
        points,
        density_data,
        density_noise,
        xev,
        yev,
        confCutOff=confCutOff,
        plotFlag=plot_odds
    )

    # --- Step 11: Apply Romano rasterization with decay + rise constraints
    raster, mapOfOddsJoint = rasterize_with_odds(
        norm_data=norm_data,
        mapOfOdds=mapOfOdds,
        xev=xev,
        yev=yev,
        fps=fps,
        tauDecay=tauDecay
    )

    return mapOfOdds, deltaF_center, density_data, density_noise, xev, yev, raster, mapOfOddsJoint


def plot_dff_and_raster(deltaF_center, raster, fps=2.0, vmax_dff=0.3, title=None):
    """
    Plot one figure with two panels (same neuron order):
      top  : ΔF/F (centered)
      bottom: binary raster (0/1)

    Parameters
    ----------
    deltaF_center : (T, N) array
        Centered ΔF/F traces.
    raster : (T, N) array (0/1)
        Event raster aligned to the same neurons and time as deltaF_center.
    fps : float
        Frames per second (for time axis).
    vmax_dff : float
        Upper color limit for ΔF/F heatmap (vmin=0). Use None to auto-scale (99th pct).
    title : str | None
        Optional figure title.

    Returns
    -------
    sort_idx : (N,) ndarray
        Indices used to sort neurons.
    """
    # safety checks
    assert deltaF_center.shape == raster.shape, "deltaF_center and raster must have same shape (T, N)."
    T, N = deltaF_center.shape

    max_vals = np.nanmax(deltaF_center, axis=0)  # (N,)
    max_vals_safe = np.nan_to_num(max_vals, nan=-np.inf)  # all-NaN → very small
    sort_idx = np.argsort(-max_vals_safe)

    # --- sort and prepare axes ---
    dff_sorted = deltaF_center[:, sort_idx].T   # (N, T)
    ras_sorted = raster[:, sort_idx].T          # (N, T)
    time = np.arange(T) / float(fps)

    if vmax_dff is None:
        # auto-scale: robust upper limit
        vmax_dff = float(np.nanpercentile(dff_sorted, 99))
        if vmax_dff <= 0:  # keep it sensible
            vmax_dff = 0.3

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    im0 = ax0.imshow(
        dff_sorted,
        aspect='auto',
        cmap='gray_r',
        vmin=0, vmax=vmax_dff,
        extent=[time[0], time[-1], 0, N]
    )
    ax0.set_ylabel("# Neuron (max ΔF/F sorted)")
    ax0.set_title("ΔF/F (centered)")

    cbar0 = plt.colorbar(im0, ax=ax0)
    cbar0.set_label("ΔF/F")

    im1 = ax1.imshow(
        ras_sorted,
        aspect='auto',
        cmap='Greys',
        vmin=0, vmax=1,
        extent=[time[0], time[-1], 0, N]
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("# Neuron (same order)")
    ax1.set_title("Event raster")

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Activity (0/1)")

    if title:
        fig.suptitle(title, y=1.02)

    plt.show()
    return sort_idx

# fps = 2.0  # Imaging rate in Hz
# tauDecay = 6.0  # GCaMP6s decay time (sec)
# f_path = Path("C:/Users/suribear/OneDrive - Université de Lausanne/Lab/Data/2p/plane1")
#
# deltaF_F = np.load(f_path / "dFoF.npy")

# mapOfOdds, deltaF_center, density_data, density_noise, xev, yev, raster, mapOfOddsJoint = compute_noise_model_romano_fast_modular(
#     deltaF_F,
#     n_bins=1000,
#     k_neighbors=100,
#     confCutOff=90,
#     plot_odds=False,
#     fps = 2.0,       # or your actual imaging rate
#     tauDecay = 6.0,   # seconds (depends on calcium indicator, e.g. GCaMP6s ~ 1s)
# )
# sort_idx = plot_dff_and_raster(deltaF_center, raster, fps=2.0, vmax_dff=0.4)