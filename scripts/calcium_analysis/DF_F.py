# Define analysis parameters here
# -----------------------------------------
fps = 2.0               # Imaging rate in Hz
tau = 6.0               # GCaMP6s decay time (sec)
percentile = 8          # Percentile for baseline (e.g. 8th)
instability_ratio=0.1   # Baseline instability check (10× drop = 0.1)
# -----------------------------------------
from scipy.ndimage import uniform_filter1d
import numpy as np

# Load your data
fluo = np.load("C:/Users/suribear/OneDrive - Université de Lausanne/Lab/Data/2p/F.npy")
fluo = fluo.T

# ----------------------------------------------
# ROI Quality Filter: Dim Fluorescence Removal
# - Compute mean F per ROI
# - Remove ROIs with mean F << group average
# - Threshold = mean - 2 × std (data-driven)
# - Keeps only bright, likely-neuronal ROIs
# ----------------------------------------------

mean_fluo_per_roi = np.mean(fluo, axis=0)  # shape (N,)
mu = np.mean(mean_fluo_per_roi)
sigma = np.std(mean_fluo_per_roi)
min_fluo_threshold = mu - 2 * sigma

roi_mask = mean_fluo_per_roi >= min_fluo_threshold
removed_indices = np.where(~roi_mask)[0]

print(f"\nNumber of ROIs eliminated due to low mean fluorescence: {len(removed_indices)}")
print("Indices of dim ROIs:")
for i in removed_indices:
    print(f"  ROI {i}")

fluo_filtered = fluo[:, roi_mask]  # Remove dim ROIs
# ----------------------------------------------
# Function to Compute Smoothed Percentile Baseline

def compute_percentile_baseline(fluo_filtered, fps, tau, percentile=8, instability_ratio=0.1):
    """
    Compute smooth baseline using running percentile and smoothing.
    Discards ROIs with unstable baseline (drops > 10x).

    Parameters:
        fluo_filtered: array of shape (T(frames), N_filtered(Valid ROIs))
        percentile: low percentile for baseline estimation
        instability_ratio: discard if min(F0) < ratio * max(F0)

    Returns:
        F0: baseline array of shape (T, N_filtered), NaN for discarded ROIs
    """
    T, N = fluo_filtered.shape
    window_size_s = max(15, 40 * tau) # 15s or 40×tau
    w = int(window_size_s * fps)

    F0 = np.full_like(fluo_filtered, np.nan)  # fill with NaNs by default

    for n in range(N):
        trace = fluo_filtered[:, n]
        baseline = np.zeros_like(trace)

        # Sliding percentile window
        for t in range(w, T - w):
            window = trace[t - w:t + w + 1]
            baseline[t] = np.percentile(window, percentile)

        # Pad edges
        baseline[:w] = baseline[w]
        baseline[T - w:] = baseline[T - w - 1]

        # ---- Baseline instability check (10× drop = 0.1) ----
        if np.min(baseline) < instability_ratio * np.max(baseline):
            continue  # leave this column as NaN
        # Smooth baseline
        F0[:, n] = uniform_filter1d(baseline, size=w)

    return F0

# ----------------------------------------------
# Compute Baseline and Filter Unstable ROIs
# ----------------------------------------------

F0 = compute_percentile_baseline(fluo_filtered, fps, tau, percentile=percentile, instability_ratio=0.1)

unstable_mask = np.isnan(F0).all(axis=0)
unstable_indices = np.where(unstable_mask)[0]

print(f"\nNumber of unstable ROIs discarded due to baseline instability: {len(unstable_indices)}")
print("Indices of unstable ROIs:")
for i in unstable_indices:
    print(f"  ROI {i}")

# Keep only stable ROIs
F0_clean = F0[:, ~unstable_mask]
fluo_clean = fluo_filtered[:, ~unstable_mask]

# ----------------------------------------------
# Compute ΔF/F₀
# ----------------------------------------------
F0_safe = np.where(F0_clean == 0, np.finfo(float).eps, F0_clean)
deltaF_F = (fluo_clean - F0_clean) / F0_safe  # shape: (T, N_final)

print(f"\nΔF/F₀ computed successfully. Final number of ROIs: {deltaF_F.shape[1]}")


from sklearn.decomposition import PCA

# Assume deltaF_F is (T, N), we need (N, T) for PCA
data = deltaF_F.T  # shape: (neurons, time)
pca = PCA(n_components=1)
scores = pca.fit_transform(data)

# Sort neurons by their score along the first PC
sort_idx = np.argsort(scores[:, 0])
sorted_data = deltaF_F[:, sort_idx].T  # shape: (neurons, time)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
im = plt.imshow(sorted_data, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
plt.xlabel("Time (frame)")
plt.ylabel("# Neuron (sorted)")
cbar = plt.colorbar(im, label="ΔF/F")
plt.title("ΔF/F Raster Plot (sorted by activity)")
plt.tight_layout()
plt.show()


# Compute time vector in seconds
n_timepoints = sorted_data.shape[1]
time = np.arange(n_timepoints) / fps  # time axis in seconds

plt.figure(figsize=(10, 6))
im = plt.imshow(
    sorted_data,
    aspect='auto',
    cmap='gray_r',
    vmin=0, vmax=1,
    extent=[time[0], time[-1], 0, sorted_data.shape[0]]
)
plt.xlabel("Time (s)")
plt.ylabel("# Neuron (sorted)")
cbar = plt.colorbar(im, label="ΔF/F")
plt.title("ΔF/F Raster Plot (sorted by activity)")
plt.tight_layout()
plt.show()