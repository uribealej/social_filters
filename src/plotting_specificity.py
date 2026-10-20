"""Stimulus specificity, selectivity and vector-similarity figures."""

import numpy as np
import matplotlib.pyplot as plt
from src.plotting_common import (
    _resolve_analysis_label,
    _preferred_stimulus_colors,
)


def plot_family_sparseness_cdf(per_fish, mean_cdf, fish_ids, ax, title,
                              individual_style="identified"):
    """Plot prepared per-fish and equal-fish lifetime-sparseness CDFs.

    Args:
        per_fish (DataFrame): One selected side/family/population, with fish_id,
            sparseness, cumulative_fraction. Missing distributions stay gaps.
        mean_cdf (DataFrame): Matching equal-fish cumulative_fraction values
            with valid_fish counts, in sparseness order.
        fish_ids (sequence): Stable fish/color order.
        ax (Axes): Destination axes.
        title (str): Explicit side/family label.
        individual_style (str): identified keeps colored fish curves;
            anonymous_points shows gray evaluation-point dots and a black mean.

    Returns:
        Axes: Both axes cover [0, 1]. No calculation, saving, or forced display.
    """
    if individual_style not in {"identified", "anonymous_points"}:
        raise ValueError("individual_style must be identified or anonymous_points.")
    anonymous = individual_style == "anonymous_points"
    colors = plt.get_cmap("tab10")
    for index, fid in enumerate(fish_ids):
        rows = per_fish.loc[per_fish["fish_id"] == fid].sort_values("sparseness")
        if anonymous:
            ax.plot(rows["sparseness"], rows["cumulative_fraction"], "o",
                    color="0.6", alpha=0.45, markersize=0.8,
                    label="Fish" if index == 0 else "_nolegend_")
        else:
            ax.step(rows["sparseness"], rows["cumulative_fraction"], where="post",
                    color=colors(index % 10), alpha=0.65, linewidth=1, label=str(fid))
    rows = mean_cdf.sort_values("sparseness")
    ax.step(rows["sparseness"], rows["cumulative_fraction"], where="post",
            color="black", linewidth=0.75 if anonymous else 2,
            label="Mean" if anonymous else "Fish mean CDF")
    n_valid = int(rows["valid_fish"].max()) if len(rows) else 0
    ax.set(title=f"{title}\n({n_valid}/{len(fish_ids)} fish)", xlim=(0, 1), ylim=(0, 1.025),
           xlabel="Lifetime sparseness\n(positive responses)", ylabel="Cumulative fraction of\ndefined neurons")
    if n_valid == 0:
        ax.text(0.5, 0.5, "No defined sparseness", transform=ax.transAxes, ha="center", va="center")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_signed_response_profiles(responses, ax, title, color_limit, cmap="RdBu_r"):
    """Draw sorted neuron-by-condition signed activity, with gray missing values.

    Args:
        responses (DataFrame): Display-ordered neurons by condition, in z-score
            activity units (signed AUC divided by observed integration span).
        ax (Axes): Destination axes.
        title (str): Fish, side, and population description.
        color_limit (float): Positive shared absolute color bound, without
            changing or normalizing the supplied response values.
        cmap (str): Diverging colormap centered at zero.

    Returns:
        AxesImage: Image for a shared colorbar. Empty populations are annotated;
        NaNs/infinities are masked. No files are saved and no display is forced.
    """
    if not np.isfinite(color_limit) or color_limit <= 0:
        raise ValueError("color_limit must be finite and positive.")
    values = responses.to_numpy(dtype=float)
    shown = values if len(values) else np.full((1, len(responses.columns)), np.nan)
    image = ax.imshow(np.ma.masked_invalid(shown), aspect="auto", interpolation="nearest",
                      cmap=plt.get_cmap(cmap).with_extremes(bad="0.9"),
                      vmin=-color_limit, vmax=color_limit)
    ax.set(title=title, xticks=np.arange(len(responses.columns)),
           xticklabels=responses.columns, ylabel="Neurons (display order)")
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    if not len(values):
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No neurons", transform=ax.transAxes, ha="center", va="center")
    return image


def plot_stimulus_similarity_matrix(matrix, ax, title, metric, cmap=None):
    """Draw a similarity matrix with fixed metric limits and gray undefined cells.

    Args:
        matrix (DataFrame): Ordered stimulus-by-stimulus similarities.
        ax (Axes): Destination axes.
        title (str): Explicit population/side label.
        metric (str): jaccard uses [0, 1]; pearson/cosine use [-1, 1].
        cmap (str or None): Optional palette, shared across panels for the metric.

    Returns:
        AxesImage: Image for a shared colorbar. No saving or forced display.
    """
    if metric not in {"jaccard", "pearson", "cosine"}:
        raise ValueError("metric must be jaccard, pearson, or cosine.")
    signed = metric != "jaccard"
    palette = plt.get_cmap(cmap or ("RdBu_r" if signed else "viridis")).with_extremes(bad="0.9")
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(np.ma.masked_invalid(values), vmin=-1 if signed else 0,
                      vmax=1, cmap=palette, aspect="equal")
    ax.set(title=title, xlabel="Stimulus B", ylabel="Stimulus A",
           xticks=np.arange(matrix.shape[1]), xticklabels=matrix.columns,
           yticks=np.arange(matrix.shape[0]), yticklabels=matrix.index)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for i, j in np.ndindex(values.shape):
        value = values[i, j]
        dark = abs(value) > 0.55 if signed else value <= 0.55
        ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "—", ha="center", va="center",
                fontsize="small", color="white" if np.isfinite(value) and dark else "black")
    return image


def plot_similarity_heatmaps(
    pearson_similarity_matrix,
    cosine_similarity_matrix,
    figsize=(10, 4),
    cmap="vlag",
):
    """Plot Pearson and cosine stimulus-vector similarity heatmaps."""
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax, matrix, title in (
        (axes[0], pearson_similarity_matrix, "Pearson similarity"),
        (axes[1], cosine_similarity_matrix, "Cosine similarity"),
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
        )
        ax.set_title(title)
    return fig, axes


def plot_similarity_by_distance(
    pair_similarity,
    similarity_column="pearson_similarity",
    ax=None,
    title=None,
):
    """Plot pairwise stimulus-vector similarity grouped by selected-order distance."""
    if similarity_column not in pair_similarity.columns:
        raise ValueError(f"pair_similarity is missing {similarity_column!r}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    plot_df = pair_similarity.dropna(subset=[similarity_column]).copy()
    if not plot_df.empty:
        pair_offsets = plot_df.groupby("segment_distance").cumcount()
        pair_counts = plot_df.groupby("segment_distance")[similarity_column].transform("size")
        pair_jitter = (pair_offsets - (pair_counts - 1) / 2.0) * 0.04
        ax.scatter(
            plot_df["segment_distance"] + pair_jitter,
            plot_df[similarity_column],
            color="0.25",
            s=45,
            alpha=0.85,
            label="Pairs",
        )
        distance_means = plot_df.groupby("segment_distance", as_index=False)[
            similarity_column
        ].mean()
        ax.plot(
            distance_means["segment_distance"],
            distance_means[similarity_column],
            marker="D",
            color="tab:red",
            linewidth=1.5,
            label="Mean",
        )
        ax.legend(frameon=False)

    ylabel = similarity_column.replace("_", " ").capitalize()
    ax.set_xlabel("Segment distance")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} by segment distance")
    return ax


def plot_stimulus_specificity_sparseness(
    summary_table,
    selected_stimulus_labels=None,
    analysis_label=None,
    ax=None,
    palette="tab10",
    alpha=0.75,
    s=18,
):
    """
    Plot lifetime sparseness against maximum raw response.
    """
    required = ["lifetime_sparseness", "max_response", "preferred_stimulus"]
    missing = [column for column in required if column not in summary_table.columns]
    if missing:
        raise ValueError("summary_table is missing column(s): " + ", ".join(missing))

    if selected_stimulus_labels is None:
        selected_stimulus_labels = [
            label
            for label in summary_table["preferred_stimulus"].dropna().unique().tolist()
        ]
    selected_stimulus_labels = list(selected_stimulus_labels)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    color_map = _preferred_stimulus_colors(selected_stimulus_labels, palette=palette)
    plotted_labels = []
    for label in selected_stimulus_labels:
        mask = summary_table["preferred_stimulus"] == label
        data = summary_table.loc[mask, ["lifetime_sparseness", "max_response"]]
        finite = np.isfinite(data["lifetime_sparseness"]) & np.isfinite(data["max_response"])
        if not np.any(finite):
            continue
        ax.scatter(
            data.loc[finite, "lifetime_sparseness"],
            data.loc[finite, "max_response"],
            color=color_map[label],
            label=label,
            alpha=alpha,
            s=s,
            linewidths=0,
        )
        plotted_labels.append(label)

    label_text = _resolve_analysis_label(summary_table, analysis_label)
    title = "Lifetime sparseness vs max response"
    if label_text:
        title = f"{title} - {label_text}"
    ax.set_title(title)
    ax.set_xlabel("Lifetime sparseness")
    ax.set_ylabel("Max z-score AUC")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if plotted_labels:
        ax.legend(title="Preferred", frameon=False, fontsize=8)
    return ax


def plot_stimulus_specificity_selectivity_index(
    summary_table,
    selected_stimulus_labels=None,
    analysis_label=None,
    ax=None,
    palette="tab10",
    alpha=0.75,
    s=18,
):
    """
    Plot raw-response selectivity index against maximum raw response.
    """
    required = ["selectivity_index", "max_response", "preferred_stimulus"]
    missing = [column for column in required if column not in summary_table.columns]
    if missing:
        raise ValueError("summary_table is missing column(s): " + ", ".join(missing))

    if selected_stimulus_labels is None:
        selected_stimulus_labels = [
            label
            for label in summary_table["preferred_stimulus"].dropna().unique().tolist()
        ]
    selected_stimulus_labels = list(selected_stimulus_labels)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    color_map = _preferred_stimulus_colors(selected_stimulus_labels, palette=palette)
    plotted_labels = []
    for label in selected_stimulus_labels:
        mask = summary_table["preferred_stimulus"] == label
        data = summary_table.loc[mask, ["selectivity_index", "max_response"]]
        finite = np.isfinite(data["selectivity_index"]) & np.isfinite(data["max_response"])
        if not np.any(finite):
            continue
        ax.scatter(
            data.loc[finite, "selectivity_index"],
            data.loc[finite, "max_response"],
            color=color_map[label],
            label=label,
            alpha=alpha,
            s=s,
            linewidths=0,
        )
        plotted_labels.append(label)

    label_text = _resolve_analysis_label(summary_table, analysis_label)
    title = "Selectivity index vs max response"
    if label_text:
        title = f"{title} - {label_text}"
    ax.set_title(title)
    ax.set_xlabel("Selectivity index")
    ax.set_ylabel("Max z-score AUC")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    if plotted_labels:
        ax.legend(title="Preferred", frameon=False, fontsize=8)
    return ax


def plot_active_stimuli_histogram(
    summary_table,
    selected_stimulus_labels=None,
    analysis_label=None,
    ax=None,
    color="0.35",
):
    """
    Plot the distribution of active-stimulus counts.
    """
    if "n_active_stimuli" not in summary_table.columns:
        raise ValueError("summary_table is missing n_active_stimuli.")

    if selected_stimulus_labels is not None:
        max_count = len(selected_stimulus_labels)
    elif len(summary_table) > 0:
        max_count = int(np.nanmax(summary_table["n_active_stimuli"]))
    else:
        max_count = 0

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    counts = (
        summary_table["n_active_stimuli"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .reindex(range(max_count + 1), fill_value=0)
    )
    ax.bar(counts.index, counts.values, color=color, width=0.8)
    label_text = _resolve_analysis_label(summary_table, analysis_label)
    title = "Active stimuli per neuron"
    if label_text:
        title = f"{title} - {label_text}"
    ax.set_title(title)
    ax.set_xlabel("Number of active stimuli")
    ax.set_ylabel("Neurons")
    ax.set_xticks(range(max_count + 1))
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_preferred_stimulus_distribution(
    summary_table,
    selected_stimulus_labels,
    analysis_label=None,
    ax=None,
    palette="tab10",
):
    """
    Plot preferred-stimulus counts in the selected stimulus order.
    """
    if "preferred_stimulus" not in summary_table.columns:
        raise ValueError("summary_table is missing preferred_stimulus.")

    selected_stimulus_labels = list(selected_stimulus_labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    counts = (
        summary_table["preferred_stimulus"]
        .value_counts()
        .reindex(selected_stimulus_labels, fill_value=0)
    )
    color_map = _preferred_stimulus_colors(selected_stimulus_labels, palette=palette)
    ax.bar(
        counts.index,
        counts.values,
        color=[color_map[label] for label in selected_stimulus_labels],
        width=0.8,
    )
    label_text = _resolve_analysis_label(summary_table, analysis_label)
    title = "Preferred stimulus distribution"
    if label_text:
        title = f"{title} - {label_text}"
    ax.set_title(title)
    ax.set_xlabel("Preferred stimulus")
    ax.set_ylabel("Neurons")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_stimulus_specificity_summary(
    summary_table,
    selected_stimulus_labels,
    analysis_label=None,
    figsize=(15, 4),
    palette="tab10",
):
    """
    Plot the three Slice 6 stimulus-specificity summary panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    plot_stimulus_specificity_sparseness(
        summary_table,
        selected_stimulus_labels=selected_stimulus_labels,
        analysis_label=analysis_label,
        ax=axes[0],
        palette=palette,
    )
    plot_active_stimuli_histogram(
        summary_table,
        selected_stimulus_labels=selected_stimulus_labels,
        analysis_label=analysis_label,
        ax=axes[1],
    )
    plot_preferred_stimulus_distribution(
        summary_table,
        selected_stimulus_labels=selected_stimulus_labels,
        analysis_label=analysis_label,
        ax=axes[2],
        palette=palette,
    )
    return fig, axes
