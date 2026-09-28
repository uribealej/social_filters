"""Pure neuron ordering for raster and active-response diagnostics."""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
import numpy as np


def _first_post_onset_raster_time(raster_trace, time_s):
    """Return the first displayed significant-raster time at/after stimulus onset."""
    trace = np.asarray(raster_trace, dtype=float)
    times = np.asarray(time_s, dtype=float)
    active_after_onset = (times >= 0.0) & (trace > 0)
    active_indices = np.flatnonzero(active_after_onset)
    return float(times[active_indices[0]]) if active_indices.size else np.inf


def _diagnostic_sort_order(trace_matrix, decision_matrix, sort_mode):
    n_neurons = trace_matrix.shape[0]
    if sort_mode is None or sort_mode == "none":
        return np.arange(n_neurons)
    if sort_mode != "decision_then_mean":
        raise ValueError("sort_mode must be 'decision_then_mean', 'none', or None.")

    if n_neurons == 0:
        return np.array([], dtype=int)

    decision_bits = np.asarray(decision_matrix, dtype=int)
    weights = 2 ** np.arange(decision_bits.shape[1] - 1, -1, -1)
    signatures = decision_bits @ weights
    mean_strength = np.nanmean(trace_matrix, axis=1)
    mean_strength = np.nan_to_num(mean_strength, nan=-np.inf)
    return np.lexsort((-mean_strength, -signatures))


def compute_sort_orders(data, n_clusters=3, random_state=42):
    """
    data: array (neurons, time) – e.g. your chunked_data
    returns: dict mode -> index array
    """
    # (a) Max intensity
    max_per_neuron = np.nanmax(data, axis=0)
    maxint_sorted_idx = np.argsort(-max_per_neuron)

    # (b) PCA
    scores = PCA(n_components=n_clusters).fit_transform(data.T)
    pca_order = np.argsort(-scores[:, 0])

    # (c) KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data.T)
    kmeans_sorted_idx = np.argsort(kmeans.labels_)

    # (d) Hierarchical (Ward)
    Z_hier = linkage(data.T, method='ward')
    clusters = fcluster(Z_hier, t=n_clusters, criterion='maxclust')
    hier_sorted_idx = np.argsort(clusters)

    # # (e) Correlation on averaged traces
    # dist = pdist(data.T, metric='correlation')
    # Z_corr = linkage(dist, method='average')
    # corravg_sorted_idx = leaves_list(Z_corr)

    X = data.T

    # 1) detectar filas problemáticas (NaN/Inf o varianza 0)
    finite_rows = np.isfinite(X).all(axis=1)
    std_rows = np.nanstd(X, axis=1)
    non_const_rows = std_rows > 0

    good_rows = finite_rows & non_const_rows

    if not np.all(good_rows):
        print(f"[corravg] Ignorando {np.sum(~good_rows)} neuronas con NaN/Inf o varianza 0 para el ordenado.")

    X_good = X[good_rows, :]

    if X_good.shape[0] == 0:
        raise ValueError("[corravg] No quedan neuronas con datos finitos y varianza>0 para ordenar.")

    # 2) calcular distancias de correlación solo con las filas buenas
    dist = pdist(X_good, metric="correlation")
    if not np.isfinite(dist).all():
        raise ValueError("[corravg] Todavía hay NaN/Inf en la matriz de distancias incluso tras filtrar.")

    Z_corr = linkage(dist, method="average")
    order_good = leaves_list(Z_corr)  # índices relativos dentro de X_good

    # 3) mapear de vuelta a índices originales
    idx_all = np.arange(X.shape[0])
    idx_good = idx_all[good_rows]
    idx_bad = idx_all[~good_rows]

    Z_corr_order = np.concatenate([idx_good[order_good], idx_bad])

    return {
        "unsorted":    None,
        "max_intensity": maxint_sorted_idx,
        "pca":          pca_order,
        "kmeans":       kmeans_sorted_idx,
        "hier":         hier_sorted_idx,
        "corravg":      Z_corr_order,
    }


def compute_single_sort_order(data, sort_mode, n_clusters=3, random_state=42):
    """
    data: array (time, neurons) – e.g. chunked_data.T
    sort_mode: 'unsorted', 'max_intensity', 'pca', 'kmeans', 'hier', 'corravg'
    returns: index array or None (for 'unsorted')
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (time, neurons), got {data.shape}")

    if sort_mode == "unsorted":
        return None

    # (a) Max intensity per neuron
    if sort_mode == "max_intensity":
        max_per_neuron = np.nanmax(data, axis=0)  # (neurons,)
        return np.argsort(-max_per_neuron)

    # (b) PCA
    if sort_mode == "pca":
        scores = PCA(n_components=n_clusters).fit_transform(data.T)  # (neurons, n_components)
        return np.argsort(-scores[:, 0])

    # (c) KMeans
    if sort_mode == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data.T)
        return np.argsort(kmeans.labels_)

    # (d) Hierarchical (Ward)
    if sort_mode == "hier":
        Z_hier = linkage(data.T, method="ward")
        clusters = fcluster(Z_hier, t=n_clusters, criterion="maxclust")
        return np.argsort(clusters)

    # (e) Correlation on averaged traces
    # if sort_mode == "corravg":
    #     dist = pdist(data.T, metric="correlation")
    #     Z_corr = linkage(dist, method="average")
    #     return leaves_list(Z_corr)

    if sort_mode == "corravg":
        # data: (n_neurons, n_time)  ->  X: (n_neurons, n_time)
        X = data.T

        # 1) detectar filas problemáticas (NaN/Inf o varianza 0)
        finite_rows = np.isfinite(X).all(axis=1)
        std_rows = np.nanstd(X, axis=1)
        non_const_rows = std_rows > 0

        good_rows = finite_rows & non_const_rows

        if not np.all(good_rows):
            print(f"[corravg] Ignorando {np.sum(~good_rows)} neuronas con NaN/Inf o varianza 0 para el ordenado.")

        X_good = X[good_rows, :]

        if X_good.shape[0] == 0:
            raise ValueError("[corravg] No quedan neuronas con datos finitos y varianza>0 para ordenar.")

        # 2) calcular distancias de correlación solo con las filas buenas
        dist = pdist(X_good, metric="correlation")
        if not np.isfinite(dist).all():
            raise ValueError("[corravg] Todavía hay NaN/Inf en la matriz de distancias incluso tras filtrar.")

        Z_corr = linkage(dist, method="average")
        order_good = leaves_list(Z_corr)  # índices relativos dentro de X_good

        # 3) mapear de vuelta a índices originales
        idx_all = np.arange(X.shape[0])
        idx_good = idx_all[good_rows]
        idx_bad = idx_all[~good_rows]

        neuron_order = np.concatenate([idx_good[order_good], idx_bad])
        return neuron_order

    raise ValueError(f"Unknown sort_mode {sort_mode!r}. "
                     f"Use one of: 'unsorted', 'max_intensity', 'pca', 'kmeans', 'hier', 'corravg'.")
