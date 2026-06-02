from prompt_toolkit.contrib.telnet import TelnetServer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import re
import src.stimuli_timeline as st
from matplotlib.patches import Patch
import pandas as pd


def _natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def list_stimulus_names(stimuli_path, pattern="*trajectory.*"):
    """
    Return stimulus names from trajectory files, stripping the `_trajectory` suffix.
    """
    path = Path(stimuli_path)
    files = list(path.glob(pattern))

    if not files and (path / "stimuli").is_dir():
        files = list((path / "stimuli").glob(pattern))

    names = []
    for stim_file in files:
        name = stim_file.stem
        if name.endswith("_trajectory"):
            name = name[: -len("_trajectory")]
        names.append(name)

    return sorted(set(names), key=_natural_sort_key)


def _stimulus_style_key(name):
    key = name

    for left, right in (("Left", "Right"), ("Le", "Ri")):
        if key.startswith(left):
            return key[len(left):]
        if key.startswith(right):
            return key[len(right):]
        if key.endswith(left):
            return key[: -len(left)]
        if key.endswith(right):
            return key[: -len(right)]

    if re.match(r"^F[LR]", key):
        return "F" + key[2:]

    if len(key) > 1 and key[-1] in {"L", "R"}:
        return key[:-1]

    return key


def _is_right_side_stimulus(name):
    return (
        name.startswith(("Right", "Ri"))
        or name.endswith(("Right", "Ri"))
        or bool(re.match(r"^F[R]", name))
        or (len(name) > 1 and name.endswith("R"))
    )


def build_stimulus_style_maps(
    stimuli_path=None,
    stimuli_names=None,
    palette="tab20",
    color_overrides=None,
    linestyle_overrides=None,
):
    """
    Build color and linestyle dictionaries from stimulus names.

    Left/right stimulus pairs share a color; right-side variants use dashed lines.
    """
    if stimuli_names is None:
        if stimuli_path is None:
            raise ValueError("Provide either stimuli_path or stimuli_names.")
        stimuli_names = list_stimulus_names(stimuli_path)

    stimuli_names = sorted(set(stimuli_names), key=_natural_sort_key)
    color_overrides = color_overrides or {}
    linestyle_overrides = linestyle_overrides or {}

    style_keys = []
    for name in stimuli_names:
        key = _stimulus_style_key(name)
        if key not in style_keys:
            style_keys.append(key)

    cmap = plt.get_cmap(palette, max(len(style_keys), 1))
    color_by_key = {key: cmap(i) for i, key in enumerate(style_keys)}

    stimuli_colors = {
        name: color_overrides.get(name, color_by_key[_stimulus_style_key(name)])
        for name in stimuli_names
    }
    stimuli_linestyles = {
        name: linestyle_overrides.get(name, "--" if _is_right_side_stimulus(name) else "-")
        for name in stimuli_names
    }

    return stimuli_colors, stimuli_linestyles


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


def _resolve_analysis_label(summary_table, analysis_label=None):
    if analysis_label is not None:
        return str(analysis_label)
    if "analysis_label" in summary_table.columns and len(summary_table) > 0:
        values = summary_table["analysis_label"].dropna().unique()
        if len(values) == 1:
            return str(values[0])
    return None


def _preferred_stimulus_colors(labels, palette="tab10"):
    labels = list(labels)
    cmap = plt.get_cmap(palette, max(len(labels), 1))
    return {label: cmap(idx) for idx, label in enumerate(labels)}


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


def _plot_lme_fixed_effects(fixed_effects, include_intercept=False, figsize=None):
    coef_df = pd.DataFrame(fixed_effects).copy()
    if coef_df.empty:
        return None, None
    if not include_intercept:
        coef_df = coef_df[coef_df["term"] != "Intercept"].copy()
    if coef_df.empty:
        return None, None

    model_names = coef_df["model_name"].drop_duplicates().tolist()
    if figsize is None:
        figsize = (8, max(2.8, 2.3 * len(model_names)))
    fig, axes = plt.subplots(len(model_names), 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, model_name in zip(axes, model_names):
        model_df = coef_df[coef_df["model_name"] == model_name].copy()
        model_df = model_df.sort_values("coefficient")
        y = np.arange(model_df.shape[0])
        ax.axvline(0, color="0.35", linewidth=1, linestyle="--")
        xerr = None
        if {"ci_lower", "ci_upper"}.issubset(model_df.columns):
            lower = model_df["coefficient"] - model_df["ci_lower"]
            upper = model_df["ci_upper"] - model_df["coefficient"]
            if np.isfinite(lower).all() and np.isfinite(upper).all():
                xerr = np.vstack([lower.to_numpy(), upper.to_numpy()])
        ax.errorbar(
            model_df["coefficient"],
            y,
            xerr=xerr,
            fmt="o",
            color="0.15",
            ecolor="0.45",
            capsize=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(model_df["term"])
        ax.set_title(model_name)
        ax.set_xlabel("Fixed-effect coefficient")
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, axes


def _plot_lme_model_comparison(model_comparison, figsize=(7, 4)):
    comparison = pd.DataFrame(model_comparison).copy()
    if comparison.empty:
        return None, None
    comparison = comparison[comparison["status"] == "success"].copy()
    comparison = comparison.dropna(subset=["aic", "bic"], how="all")
    if comparison.empty:
        return None, None

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(comparison.shape[0])
    width = 0.38
    if "aic" in comparison.columns:
        ax.bar(x - width / 2, comparison["aic"], width=width, label="AIC")
    if "bic" in comparison.columns:
        ax.bar(x + width / 2, comparison["bic"], width=width, label="BIC")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["model_name"], rotation=35, ha="right")
    ax.set_ylabel("Information criterion")
    ax.set_title("Model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def _plot_lme_observed_vs_fitted(fit_results, response_table, figsize=None):
    if response_table is None:
        return None, None
    df = pd.DataFrame(response_table)
    successful = [
        (name, payload["result"])
        for name, payload in fit_results.items()
        if payload.get("status") == "success" and payload.get("result") is not None
    ]
    if not successful:
        return None, None

    if figsize is None:
        figsize = (5 * len(successful), 4)
    fig, axes = plt.subplots(1, len(successful), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    observed = df["response"].to_numpy(dtype=float)
    for ax, (model_name, result) in zip(axes, successful):
        fitted = np.asarray(result.fittedvalues, dtype=float)
        finite = np.isfinite(observed) & np.isfinite(fitted)
        ax.scatter(fitted[finite], observed[finite], s=10, alpha=0.25, linewidths=0)
        if np.any(finite):
            low = float(np.nanmin([fitted[finite].min(), observed[finite].min()]))
            high = float(np.nanmax([fitted[finite].max(), observed[finite].max()]))
            ax.plot([low, high], [low, high], color="0.35", linewidth=1, linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("Fitted response")
        ax.set_ylabel("Observed response")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, axes


def _plot_lme_response_distribution(response_table, figsize=(10, 4)):
    if response_table is None:
        return None, None
    df = pd.DataFrame(response_table).copy()
    required = {"stimulus_class", "position_id", "response"}
    if df.empty or not required.issubset(df.columns):
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, col, title in zip(
        axes,
        ["stimulus_class", "position_id"],
        ["Response by stimulus class", "Response by position"],
    ):
        labels = df[col].dropna().drop_duplicates().tolist()
        values = [df.loc[df[col] == label, "response"].dropna().to_numpy() for label in labels]
        positions = np.arange(1, len(labels) + 1)
        if values:
            ax.boxplot(values, positions=positions, showfliers=False)
        ax.axhline(0, color="0.35", linewidth=1, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Response")
    fig.tight_layout()
    return fig, axes


def plot_lme_model_outputs(results, response_table=None, include_intercept=False):
    """
    Plot fixed effects, model comparison, fitted values, and response summaries.
    """
    figures = {}
    fixed_fig, fixed_axes = _plot_lme_fixed_effects(
        results.get("fixed_effects"),
        include_intercept=include_intercept,
    )
    if fixed_fig is not None:
        figures["fixed_effects"] = (fixed_fig, fixed_axes)

    comparison_fig, comparison_ax = _plot_lme_model_comparison(
        results.get("model_comparison")
    )
    if comparison_fig is not None:
        figures["model_comparison"] = (comparison_fig, comparison_ax)

    fitted_fig, fitted_axes = _plot_lme_observed_vs_fitted(
        results.get("fit_results", {}),
        response_table,
    )
    if fitted_fig is not None:
        figures["observed_vs_fitted"] = (fitted_fig, fitted_axes)

    distribution_fig, distribution_axes = _plot_lme_response_distribution(response_table)
    if distribution_fig is not None:
        figures["response_distribution"] = (distribution_fig, distribution_axes)

    return figures


def plot_motion_delta_distribution(
    delta_df,
    value_col="delta_integral",
    segment_col="stimulus",
    side_col="side",
    stimuli=None,
    segments=None,
    side="all",
    sides=None,
    figsize=(10, 4),
    point_alpha=0.25,
    point_size=12,
    show_points=True,
    title=None,
    ax=None,
):
    """
    Plot motion-minus-fixed distributions grouped by stimulus label.

    By default this draws one panel with all rows. Pass ``side="left"`` or
    ``side="right"`` only when you explicitly want a side-specific subset.
    """
    missing = {value_col, segment_col} - set(delta_df.columns)
    if side != "all":
        missing |= {side_col} - set(delta_df.columns)
    if missing:
        raise ValueError(f"delta_df is missing required column(s): {sorted(missing)}")

    if side is not None:
        if side not in {"left", "right", "all"}:
            raise ValueError("side must be None, 'left', 'right', or 'all'.")
        sides = (side,)
    elif sides is None:
        sides = ("all",)

    if stimuli is not None:
        segments = stimuli

    if segments is None:
        segments = delta_df[segment_col].dropna().drop_duplicates().tolist()
    else:
        segments = list(segments)

    if not segments:
        raise ValueError(f"No {segment_col} values available to plot.")

    if ax is None:
        fig, axes = plt.subplots(1, len(sides), figsize=figsize, sharey=True)
        if len(sides) == 1:
            axes = np.asarray([axes])
    else:
        axes = np.asarray(ax).ravel()
        fig = axes[0].figure
        if len(axes) != len(sides):
            raise ValueError("Number of axes must match number of sides.")

    for axis, side in zip(axes, sides):
        if side == "all":
            side_df = delta_df
            axis_title = "Selected stimuli"
        else:
            side_df = delta_df[delta_df[side_col] == side]
            axis_title = side.capitalize()
        values_by_segment = [
            side_df.loc[side_df[segment_col] == segment, value_col].dropna().to_numpy()
            for segment in segments
        ]

        positions = np.arange(1, len(segments) + 1)
        nonempty_positions = [pos for pos, vals in zip(positions, values_by_segment) if vals.size > 0]
        nonempty_values = [vals for vals in values_by_segment if vals.size > 0]
        if nonempty_values:
            axis.violinplot(
                nonempty_values,
                positions=nonempty_positions,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )

            if show_points:
                rng = np.random.default_rng(0)
                for pos, vals in zip(positions, values_by_segment):
                    if vals.size == 0:
                        continue
                    jitter = rng.uniform(-0.08, 0.08, size=vals.size)
                    axis.scatter(
                        np.full(vals.size, pos) + jitter,
                        vals,
                        s=point_size,
                        alpha=point_alpha,
                        linewidths=0,
                    )

        axis.axhline(0, color="0.4", linewidth=1, linestyle="--")
        axis.set_xticks(positions)
        axis.set_xticklabels(segments, rotation=35, ha="right")
        axis.set_title(axis_title)
        axis.set_xlabel(segment_col.replace("_", " "))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel(value_col)
    if title is None:
        title = value_col.replace("_", " ")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


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


def plot_active_trace_decision_diagnostic(
    diagnostic,
    fps_2p=2.0,
    t_pre_s=5.0,
    stimuli_durations=None,
    sort_mode="decision_then_mean",
    neuron_order=None,
    figsize=(12, 7),
    trace_cmap="Greys",
    decision_cmap="Greys",
    motion_onset_key="static_before_sec",
    motion_line_color="tab:red",
    motion_line_style="--",
    show_active_count_trace=False,
    active_count_threshold=0.5,
    active_count_color="tab:blue",
    active_count_ylabel="# Active neurons",
    active_count_height_ratio=1.2,
):
    """
    Plot pooled significant-response traces beside strict active decisions.
    """
    trace_matrix = np.asarray(diagnostic["trace_matrix"], dtype=float)
    decision_matrix = np.asarray(diagnostic["decision_matrix"], dtype=float)
    stim_labels = list(diagnostic["stim_labels"])
    block_widths = list(diagnostic["trace_block_widths"])
    block_timepoints = list(diagnostic.get("trace_block_timepoints", block_widths))
    block_reps = list(diagnostic.get("trace_block_reps", [1] * len(block_widths)))
    combine_mode = diagnostic.get("combine_mode", "mean")

    if trace_matrix.ndim != 2 or decision_matrix.ndim != 2:
        raise ValueError("trace_matrix and decision_matrix must be 2D.")
    if trace_matrix.shape[0] != decision_matrix.shape[0]:
        raise ValueError("trace_matrix and decision_matrix must share row count.")

    if neuron_order is None:
        neuron_order = _diagnostic_sort_order(trace_matrix, decision_matrix, sort_mode)
    else:
        neuron_order = np.asarray(neuron_order)
        if neuron_order.ndim != 1 or neuron_order.shape[0] != trace_matrix.shape[0]:
            raise ValueError(
                f"neuron_order has shape {neuron_order.shape}; "
                f"expected ({trace_matrix.shape[0]},)."
            )

    trace_sorted = trace_matrix[neuron_order]
    decision_sorted = decision_matrix[neuron_order]

    fig = plt.figure(figsize=figsize)
    if show_active_count_trace:
        gs = gridspec.GridSpec(
            2,
            4,
            width_ratios=[32, 1.0, 7, 0.8],
            height_ratios=[6, active_count_height_ratio],
            hspace=0.05,
            wspace=0.15,
        )
        ax_trace = fig.add_subplot(gs[0, 0])
        ax_active_count = fig.add_subplot(gs[1, 0], sharex=ax_trace)
        cax_trace = fig.add_subplot(gs[0, 1])
        ax_decision = fig.add_subplot(gs[0, 2], sharey=ax_trace)
        cax_decision = fig.add_subplot(gs[0, 3])
    else:
        gs = gridspec.GridSpec(
            1,
            4,
            width_ratios=[32, 1.0, 7, 0.8],
            wspace=0.15,
        )
        ax_trace = fig.add_subplot(gs[0, 0])
        ax_active_count = None
        cax_trace = fig.add_subplot(gs[0, 1])
        ax_decision = fig.add_subplot(gs[0, 2], sharey=ax_trace)
        cax_decision = fig.add_subplot(gs[0, 3])

    trace_im = ax_trace.imshow(
        trace_sorted,
        aspect="auto",
        interpolation="nearest",
        cmap=trace_cmap,
        vmin=0,
        vmax=1,
    )
    decision_im = ax_decision.imshow(
        decision_sorted,
        aspect="auto",
        interpolation="nearest",
        cmap=decision_cmap,
        vmin=0,
        vmax=1,
    )

    boundaries = np.cumsum(block_widths)
    starts = np.r_[0, boundaries[:-1]]
    centers = starts + np.asarray(block_widths) / 2.0

    for boundary in boundaries[:-1]:
        ax_trace.axvline(boundary - 0.5, color="0.65", linewidth=0.8, alpha=0.8)
        if ax_active_count is not None:
            ax_active_count.axvline(boundary - 0.5, color="0.65", linewidth=0.8, alpha=0.8)

    if stimuli_durations is not None:
        for start, label, width, n_time, n_reps in zip(
            starts,
            stim_labels,
            block_widths,
            block_timepoints,
            block_reps,
        ):
            duration = stimuli_durations.get(label)
            if duration is None or motion_onset_key not in duration:
                continue
            motion_frame = int(
                np.clip(
                    round((float(t_pre_s) + float(duration[motion_onset_key])) * float(fps_2p)),
                    0,
                    max(int(n_time) - 1, 0),
                )
            )

            if combine_mode == "concat":
                for rep_idx in range(int(n_reps)):
                    xpos = start + rep_idx * int(n_time) + motion_frame
                    if xpos < start + width:
                        ax_trace.axvline(
                            xpos - 0.5,
                            color=motion_line_color,
                            linestyle=motion_line_style,
                            linewidth=1.0,
                            alpha=0.9,
                        )
                        if ax_active_count is not None:
                            ax_active_count.axvline(
                                xpos - 0.5,
                                color=motion_line_color,
                                linestyle=motion_line_style,
                                linewidth=1.0,
                                alpha=0.9,
                            )
            else:
                xpos = start + motion_frame
                if xpos < start + width:
                    ax_trace.axvline(
                        xpos - 0.5,
                        color=motion_line_color,
                        linestyle=motion_line_style,
                        linewidth=1.0,
                        alpha=0.9,
                    )
                    if ax_active_count is not None:
                        ax_active_count.axvline(
                            xpos - 0.5,
                            color=motion_line_color,
                            linestyle=motion_line_style,
                            linewidth=1.0,
                            alpha=0.9,
                        )

    ax_trace.set_xticks(centers)
    ax_trace.set_xticklabels(stim_labels, rotation=45, ha="right")
    ax_trace.set_xlabel("Aligned time blocks")
    ax_trace.set_ylabel("Pooled neurons")

    if ax_active_count is not None:
        active_count = np.nansum(trace_matrix >= active_count_threshold, axis=0)
        x = np.arange(trace_matrix.shape[1])
        ax_active_count.plot(x, active_count, color=active_count_color, linewidth=1.5)
        ax_active_count.set_xlim(-0.5, trace_matrix.shape[1] - 0.5)
        ax_active_count.set_xticks(centers)
        ax_active_count.set_xticklabels(stim_labels, rotation=45, ha="right")
        ax_active_count.set_ylabel(active_count_ylabel)
        ax_active_count.set_xlabel("Aligned time blocks")
        ax_active_count.set_ylim(bottom=0)
        ax_active_count.spines["top"].set_visible(False)
        ax_active_count.spines["right"].set_visible(False)
        plt.setp(ax_trace.get_xticklabels(), visible=False)
        ax_trace.set_xlabel("")

    ax_decision.set_xticks(np.arange(len(stim_labels)))
    ax_decision.set_xticklabels(stim_labels, rotation=45, ha="right")
    ax_decision.set_xlabel("Strict active decision")
    ax_decision.tick_params(labelleft=False)

    if fps_2p > 0 and trace_matrix.shape[1] > 0:
        seconds = trace_matrix.shape[1] / float(fps_2p)
        ax_trace.set_title(f"Significant raster ({combine_mode}; {seconds:.1f} s shown)")
    else:
        ax_trace.set_title(f"Significant raster ({combine_mode})")
    ax_decision.set_title("Active")

    trace_label = (
        "Fraction of repetitions significant"
        if combine_mode == "mean"
        else "Significant activity (0/1)"
    )
    fig.colorbar(trace_im, cax=cax_trace, label=trace_label)
    fig.colorbar(decision_im, cax=cax_decision, label="Final active (0/1)")

    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.18, top=0.90)
    axes = (ax_trace, ax_decision)
    if ax_active_count is not None:
        axes = (ax_trace, ax_active_count, ax_decision)
    return fig, axes, neuron_order
# def add_stimuli_markers(ax, exp_log, stimuli_durations, stimuli_colors, time_offset=0, trace='movement'):
#     """
#     Add vertical lines for stimulus movement starts and return legend handles.
#
#     Parameters:
#     - ax: matplotlib Axes object
#     - exp_log: DataFrame with stimulus events and timestamps
#     - stimuli_durations: dict with durations, e.g., {'forward': {...}}
#     - stimuli_colors: dict mapping stimulus names to colors
#     - time_offset: optional offset (e.g., start time) to align timestamps (default=0)
#
#     Returns:
#     - legend_handles: list of matplotlib Line2D objects for legend
#     """
#     for _, row in exp_log.iterrows():
#         if 'stim' in row['event']:
#             stim_name = row['event'].split('_')[-1]
#             stim_start = row['timestamp'] - exp_log['timestamp'].min() - time_offset
#             if stim_name in stimuli_durations:
#                 if trace == 'movement':
#                     dur = stimuli_durations[stim_name]
#                     move_start = stim_start + dur['static_before_sec']
#                     color = stimuli_colors.get(stim_name, 'black')
#                     ax.axvline(move_start, color=color, alpha=0.8, linewidth=1.5)
#
#     # Legend with dummy lines
#     legend_handles = []
#     for stim_name, color in stimuli_colors.items():
#         line, = ax.plot([], [], color=color, label=stim_name, linewidth=4)
#         legend_handles.append(line)
#
#     return legend_handles
def add_stimuli_markers(ax, exp_log, stimuli_durations, stimuli_colors, time_offset=0, trace='movement',
                        stimuli_linestyles=None):
    if stimuli_linestyles is None:
        stimuli_linestyles = {}

    for _, row in exp_log.iterrows():
        if 'stim' in row['event']:
            stim_name = row['event'].split('_')[-1]
            stim_start = row['timestamp'] - exp_log['timestamp'].min() - time_offset
            if stim_name in stimuli_durations:
                if trace == 'movement':
                    dur = stimuli_durations[stim_name]
                    move_start = stim_start + dur['static_before_sec']
                    color = stimuli_colors.get(stim_name, 'black')
                    ls    = stimuli_linestyles.get(stim_name, '-')  # <- dashed here
                    ax.axvline(move_start, color=color, linestyle=ls, alpha=0.9, linewidth=1.8)

    # legend with dummy lines reflecting both color and style
    legend_handles = []
    for stim_name, color in stimuli_colors.items():
        ls = stimuli_linestyles.get(stim_name, '-')
        (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=4)
        legend_handles.append(line)
    return legend_handles


# def raster_with_stimuli(
#     ax, data, fps, fish_id, neuron_order=None, title_suffix='', min=0, max=0.4):
#     """
#     Plot raster of ΔF/F traces sorted by neuron_order, with stimulus markers and legend.
#
#     Parameters:
#     - ax: matplotlib Axes object
#     - deltaF_F: (frames x neurons) ΔF/F matrix
#     - fps: frames per second (for time axis)
#     - plane_name: string for labeling plot
#     - fish_id: string for labeling plot
#     - neuron_order: 1D array of neuron indices for sorting (length = neurons)
#     - title_suffix: optional extra string for plot title (e.g., clustering method)
#     """
#     deltaF_F=data
#     # If no neuron_order, keep original order
#     if neuron_order is None:
#         neuron_order = np.arange(deltaF_F.shape[1])
#
#     # Sort data by neuron_order
#     sorted_data = deltaF_F[:, neuron_order].T  # (neurons, time)
#     time_axis = np.arange(sorted_data.shape[1]) / fps
#
#     im = ax.imshow(
#         sorted_data,
#         aspect='auto',
#         cmap='gray_r',
#         vmin=min,
#         vmax=max,
#         extent=[time_axis[0], time_axis[-1], sorted_data.shape[0], 0]
#     )
#
#     ax.set_ylabel("# Neuron")
#     ax.set_title(f"{fish_id}  DF/F - {title_suffix}")
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#
#     return im


def raster_with_stimuli(
    ax,
    data,                 # (frames x neurons) matrix: either ΔF/F or 0/1 significant
    fps,
    fish_id,
    neuron_order=None,
    title_suffix='',
    vmin=None,
    vmax=None,
    perc_for_vmax=99.0,
    is_binary=None,
):
    """
    Plot raster of traces sorted by neuron_order.

    Parameters
    ----------
    ax : matplotlib Axes
    data : array, shape (n_frames, n_neurons)
        ΔF/F or significant traces (0/1 or bool).
    fps : float
        Frames per second (for time axis).
    fish_id : str
        Fish identifier for title.
    neuron_order : array-like, optional
        Indices to sort neurons. If None, use original order.
    title_suffix : str, optional
        Extra text for the title (e.g. "k-means", "significant").
    vmin, vmax : float, optional
        Color limits. If None, chosen automatically from the data.
    perc_for_vmax : float, optional
        Percentile for automatic vmax if not given (for continuous data).
    is_binary : bool, optional
        Force binary handling (0/1). If None, automatically detected.
    add_colorbar : bool, optional
        If True, add colorbar to the same Axes' figure.

    Returns
    -------
    im : AxesImage
    """
    data = np.asarray(data)

    # Expect (frames, neurons); if user passed (neurons, frames), you could auto-detect:
    # if data.shape[0] < data.shape[1]:  # optional heuristic
    #     data = data.T

    n_frames, n_neurons = data.shape

    # If no neuron_order, keep original order
    if neuron_order is None:
        neuron_order = np.arange(n_neurons)
    else:
        neuron_order = np.asarray(neuron_order)
        assert neuron_order.shape[0] == n_neurons, (
            f"neuron_order length {neuron_order.shape[0]} != n_neurons {n_neurons}"
        )

    # Sort data by neuron_order → (neurons, time)
    sorted_data = data[:, neuron_order].T
    time_axis = np.arange(sorted_data.shape[1]) / float(fps)

    # --- Detect whether this is binary (significant traces) ---
    if is_binary is None:
        unique_vals = np.unique(sorted_data[~np.isnan(sorted_data)])  # ignore NaNs
        is_binary = (
            unique_vals.size <= 3 and
            np.all(np.isin(unique_vals, [0, 1]))
        )

    # --- Choose colormap and vmin/vmax ---
    if is_binary:
        # For 0/1 significant traces
        if vmin is None:
            vmin = -0.05
        if vmax is None:
            vmax = 1.05
        interpolation = 'nearest'  # crisp pixels
    else:
        # Continuous ΔF/F
        if vmin is None:
            # Often you want 0 as baseline for dF/F
            vmin = 0.0
        if vmax is None:
            # Use a high percentile to avoid being dominated by a few outliers
            vmax = np.nanpercentile(sorted_data, perc_for_vmax)
        interpolation = 'nearest'

    im = ax.imshow(
        sorted_data,
        aspect='auto',
        cmap='gray_r',
        vmin=vmin,
        vmax=vmax,
        extent=[time_axis[0], time_axis[-1], sorted_data.shape[0], 0],
        interpolation=interpolation,
    )

    ax.set_ylabel("# Neuron", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_title(f"{fish_id} - {title_suffix}", fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return im



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

# and here we assume compute_sort_orders and raster_with_stimuli are in this module


def save_sorted_rasters_for_all_modes(
    dfof,
    exp_log,
    stimuli_durations,
    stimuli_colors,
    stimuli_linestyles,
    stimuli_ordered,
    fps_2p,
    window_pre,
    window_post,
    plots_path: Path,
    prefix: str,
    n_clusters: int = 5,
    random_state: int = 42,
    average_across_repeats: bool = False,
    folder_name: str | None = None,
    is_binary= False,
):
    """
    Generate experiment-level ΔF/F rasters aligned to stimulus onsets and save one PNG per
    sorting strategy (unsorted, max intensity, PCA, KMeans, hierarchical, correlation).

    If average_across_repeats is:
      - False → use individual trials (your "grouped" version)
      - True  → average across repeats (your "average" version)
    """

    # Ensure output dir exists
    plots_path.mkdir(parents=True, exist_ok=True)

    # ----- 1) Build chunks once --------------------------------------------
    chunked_data, trial_starts, move_starts, move_colors, stim_labels = st.extract_stimulus_chunks(
        deltaF_F=dfof,                 # (frames, neurons)
        exp_log=exp_log,
        stimuli_durations=stimuli_durations,
        stimuli_colors=stimuli_colors,
        fps=fps_2p,
        stimuli_ordered=stimuli_ordered,
        window_pre=window_pre,
        window_post=window_post,
        average_across_repeats=average_across_repeats,
    )
    if chunked_data is None or chunked_data.size == 0:
        raise ValueError("No stimulus chunks were extracted. Check exp_log, stimuli_ordered, and windows.")

    # chunked_data: (neurons, time)
    if not is_binary:
        # Continuous data
        data_for_sort = chunked_data.T  # (time, neurons) or whatever you expect
    else:
        # Binary / thresholded version
        data_for_sort = chunked_data.T
        data_for_sort[data_for_sort < 0.5] = 0
        data_for_sort[data_for_sort >= 0.5] = 1

    # ----- 2) Compute sort orders ------------------------------------------
    # Your compute_sort_orders currently expects data.T in your notebook;
    # keep it consistent here:
    sorters = compute_sort_orders(data_for_sort, n_clusters=n_clusters, random_state=random_state)

    def _validate_order(idx, n_neurons, name):
        if idx is None:
            return None
        if len(idx) != n_neurons:
            raise ValueError(f"{name} length ({len(idx)}) != number of neurons ({n_neurons}).")
        return idx

    # Decide labels depending on average_across_repeats
    if average_across_repeats:
        title_base = "average across trials"
        filename_mid = "average"
    else:
        title_base = "ordered by stimulus type"
        filename_mid = "grouped"

    # ----- 3) Plot once per sort mode --------------------------------------
    saved_files = []

    for mode, idx in sorters.items():
        neuron_order = _validate_order(idx, chunked_data.shape[0], name=mode)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = raster_with_stimuli(
            ax=ax,
            data=chunked_data.T,            # (time, neurons)
            fps=fps_2p,
            fish_id=prefix,                     # or fish_id if you prefer
            neuron_order=neuron_order,          # None = original order
            title_suffix=f"{title_base} | sort={mode}",

        )

        # Movement onset lines with per-stim color + linestyle
        for pos, stim_name in zip(move_starts, stim_labels):
            color = stimuli_colors.get(stim_name, "black")
            ls = stimuli_linestyles.get(stim_name, "-")  # default solid
            ax.axvline(pos / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.8)

        # Legend (color + linestyle)
        legend_handles = []
        for stim_name in stimuli_ordered:
            if stim_name in stimuli_colors:
                color = stimuli_colors[stim_name]
                ls = stimuli_linestyles.get(stim_name, "-")
                (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=1.5)
                legend_handles.append(line)

        ax.legend(
            handles=legend_handles,
            title="Movement onset\nacross stimuli",
            bbox_to_anchor=(1.2, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

        ax.set_xlabel("Chunks aligned to stimulus onset (s)")

        if not is_binary:
            fig.colorbar(im, ax=ax, label=r"$\Delta F/F$")
        else:
            fig.colorbar(im, ax=ax, label=r"Significant activity (0/1)")

        plt.subplots_adjust(right=0.85)

        # Decide where to save
        if folder_name is not None:
            out_dir = plots_path / folder_name
        else:
            out_dir = plots_path

        # Make sure the folder exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save
        out_png = out_dir / f"{prefix}_{filename_mid}_dfof_sorted_by_{mode}.png"
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close(fig)

        saved_files.append(out_png.name)

    print("✅ Saved:", *[f"- {name}" for name in saved_files], sep="\n")

def plot_sorted_chunks_single_mode(
    dfof,
    exp_log,
    stimuli_durations,
    stimuli_colors,
    stimuli_linestyles,
    stimuli_ordered,
    fps_2p,
    window_pre,
    window_post,
    sort_mode="kmeans",
    n_clusters=3,
    random_state=42,
    average_across_repeats=False,
    figsize=(12, 6),
    fish_id="",
    neuron_order=None,       # 👈 NEW: custom order (optional)
    sort_label=None, # 👈 NEW: label to show in the title
    is_binary =False,
    vmin=None,
    vmax=None,
):
    """
    Build stimulus-aligned chunks and plot a single raster.

    You can either:
      - let the function compute the order via `sort_mode`, or
      - pass your own `neuron_order` (1D index array).

    If `neuron_order` is provided, `sort_mode` is ignored for the ordering.
    """

    # 1) Build chunks
    chunked_data, trial_starts, move_starts, move_colors, stim_labels = st.extract_stimulus_chunks(
        deltaF_F=dfof,
        exp_log=exp_log,
        stimuli_durations=stimuli_durations,
        stimuli_colors=stimuli_colors,
        fps=fps_2p,
        stimuli_ordered=stimuli_ordered,
        window_pre=window_pre,
        window_post=window_post,
        average_across_repeats=average_across_repeats,

    )

    if chunked_data is None or chunked_data.size == 0:
        raise ValueError("No stimulus chunks were extracted. Check exp_log, stimuli_ordered, and windows.")
        # chunked_data: (neurons, time)
    n_neurons = chunked_data.shape[0]

    if not is_binary:
        # Continuous data
        data_for_sort = chunked_data.T  # (time, neurons) or whatever you expect
    else:
        # Binary / thresholded version
        data_for_sort = chunked_data.T
        data_for_sort[data_for_sort < 0.5] = 0
        data_for_sort[data_for_sort >= 0.5] = 1

        # 2) Decide which neuron_order to use
    if neuron_order is not None:
        # Use user-provided order, with sanity check
        neuron_order = np.asarray(neuron_order)
        if neuron_order.ndim != 1 or neuron_order.shape[0] != n_neurons:
            raise ValueError(
                f"Custom neuron_order has shape {neuron_order.shape}, "
                f"expected ({n_neurons},)."
            )
        label = sort_label or "custom"
    else:
        # Compute only the requested sort order
        neuron_order = compute_single_sort_order(
            data_for_sort,
            sort_mode=sort_mode,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        label = sort_label or sort_mode

    # 3) Plot raster
    fig, ax = plt.subplots(figsize=figsize)
    im = raster_with_stimuli(
        ax=ax,
        data=data_for_sort,       # (time, neurons)
        fps=fps_2p,
        fish_id=fish_id,
        neuron_order=neuron_order,
        title_suffix=f"ordered by stimulus type | sort={label}",
        vmax= vmax,
        vmin=vmin,

    )

    # 4) Movement start lines — use style based on each chunk's stimulus label
    move_styles = [stimuli_linestyles.get(name, "-") for name in stim_labels]
    for pos, color, ls in zip(move_starts, move_colors, move_styles):
        ax.axvline(pos / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.0)

    ax.set_xlabel("Chunks aligned to stimulus onset (s)")

    # 5) Legend that matches color + linestyle (movement onsets)
    legend_handles = []
    for stim_name in stimuli_ordered:
        if stim_name in stimuli_colors:
            color = stimuli_colors[stim_name]
            ls = stimuli_linestyles.get(stim_name, "-")
            (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=2)
            legend_handles.append(line)

    # Movement legend on the *figure* (right side)
    mov_legend = fig.legend(
        handles=legend_handles,
        title="Movement onset\nacross stimuli",
        bbox_to_anchor=(0.88, 0.6),  # (x, y) in figure coords
        loc="center left",
        borderaxespad=0,
        frameon=False,
    )

    # 6) Colorbar / binary legend + layout
    if not is_binary:
        # Continuous ΔF/F → colorbar on the axes
        fig.colorbar(im, ax=ax, label=r"$\Delta F/F$",
                     fraction=0.17, pad=0.01)
    else:
        # Binary significance → legend on the axes (does NOT kill the fig legend)
        activity_handles = [
            Patch(facecolor='black', edgecolor='black', label='Sign. (1)'),
            Patch(facecolor='white', edgecolor='black', label='Not sign. (0)'),
        ]
        ax.legend(
            handles=activity_handles,
            title='Activity',
            loc='upper right',
            bbox_to_anchor=(1.2, 1.0),
            frameon=False,
        )

    fig.tight_layout()
    return fig, ax, neuron_order

    # # 5) Legend that matches color + linestyle
    # legend_handles = []
    # for stim_name, color in stimuli_colors.items():
    #     ls = stimuli_linestyles.get(stim_name, "-")
    #     (line,) = ax.plot([], [], color=color, linestyle=ls, label=stim_name, linewidth=2)
    #     legend_handles.append(line)
    #
    # ax.legend(
    #     handles=legend_handles,
    #     title="Movement onset\nacross stimuli",
    #     bbox_to_anchor=(1.15, 1),
    #     loc="upper left",
    #     borderaxespad=0,
    #     frameon=False,
    # )
    #
    # # 6) Colorbar + layout
    # if not is_binary:
    #     fig.colorbar(im, ax=ax, label=r"$\Delta F/F$")
    #     fig.tight_layout()
    # else:
    #
    #     # Binary significance → legend instead of colorbar
    #     legend_elements = [
    #         Patch(facecolor='black', edgecolor='black', label='Significant (1)'),
    #         Patch(facecolor='white', edgecolor='black', label='Not significant (0)'),
    #     ]
    #     ax.legend(handles=legend_elements, title='Activity', loc='upper right')
    # return fig, ax, neuron_order


# this function plot Df/F as a function of time per stimuli...

def plot_stimulus_means(
        mean_traces,
        stimuli_ids,
        stimuli_names,  # list of display names (used to look up styles)
        fps_2p,
        t_post_s,
        t_pre_s,
        stimuli_durations,
        title_prefix="neurons resposive to stimuli",
        show_sem=True,
        # NEW: pass styles in
        stimuli_colors: dict | None = None,  # e.g., {"LLB": (r,g,b), "FLB": ...}
        stimuli_linestyles: dict | None = None,  # e.g., {"FLB": "--", "FRB": "--"}
        # saving controls
        save: bool = False,
        plots_path: Path | str | None = None,
        prefix: str | None = None,
        dpi: int = 600,
        close_after: bool = False,
        kept_cells = None,  # optional: indices of cells to include
        comment="all_stimuli"  # for saving
):
    """
    Plots mean ± SEM per stimulus.
    Styles are looked up by stimulus *name* in `stimuli_colors` and `stimuli_linestyles`.
    """
    # defaults if not provided
    if stimuli_colors is None:     stimuli_colors = {}
    if stimuli_linestyles is None: stimuli_linestyles = {}
    pre_frames  = int(round(t_pre_s * fps_2p))
    post_frames = int(round(t_post_s * fps_2p))
    win_lenght = pre_frames + post_frames
    # time axis (s), 0 at static onset
    t = (np.arange(win_lenght) - pre_frames) / float(fps_2p)


    fig, ax = plt.subplots(figsize=(7, 4.5))
    color_by_stim = {}

    # optional warnings for missing styles
    missing_colors = [nm for nm in stimuli_names if nm not in stimuli_colors]
    missing_ls = [nm for nm in stimuli_names if nm not in stimuli_linestyles]
    if missing_colors:
        print("[warn] No color for:", missing_colors, "→ using Matplotlib defaults.")
    if missing_ls:
        print("[warn] No linestyle for:", missing_ls, "→ using solid '-'.")

    for i, stim in enumerate(stimuli_ids):
        name = stimuli_names[i]
        M = mean_traces[stim]

        if kept_cells is not None:
            M = M[kept_cells, :] # (n_kept, win_lenght)

        n_sel = M.shape[0]
        trace_mean = np.nanmean(M, axis=0)
        trace_sd = np.nanstd(M, axis=0)
        trace_sem = trace_sd / np.sqrt(max(n_sel, 1))

        color = stimuli_colors.get(name, None)  # None → mpl default
        ls = stimuli_linestyles.get(name, "-")  # default solid

        (line,) = ax.plot(t, trace_mean, label=name, color=color, linestyle=ls)
        color_by_stim[name] = line.get_color()

        if show_sem:
            ax.fill_between(
                t, trace_mean - trace_sem, trace_mean + trace_sem,
                alpha=0.18, color=line.get_color(), linewidth=0
            )

    # visual guides
    summary = summarize_durations(stimuli_durations)
    static_onset = 0
    motion_onset = summary.get("static_before_sec")
    motion_offset =summary.get("total_sec")
    ax.axvline(static_onset, linestyle="--", linewidth=1, color="grey", label="static onset")
    ax.text(static_onset - 0.5, 0.85, "static", rotation=90, va="bottom", ha="center",
            transform=ax.get_xaxis_transform())
    ax.axvline(motion_onset, linewidth=1, color="k", label="motion onset")
    ax.text(motion_onset + 1.5, 0.85, "motion", rotation=90, va="bottom", ha="center",
            transform=ax.get_xaxis_transform())
    ax.axvline(motion_offset, linestyle="--", linewidth=1, color="grey")

    # cosmetics
    ax.set(title=f"{title_prefix}",
           xlabel="Time (s) relative to onset",
           ylabel="ΔF/F")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()

    out_png = None
    if save:
        if plots_path is None or prefix is None:
            print("⚠️  Save was requested but `plots_path` or `prefix` is missing—skipping save.")
        else:
            plots_path = Path(plots_path)
            plots_path.mkdir(parents=True, exist_ok=True)
            out_png = plots_path / f"{prefix}_dfof_as_func_of_time_{comment}.png"
            fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
            print("✅ Saved:", out_png)

    if close_after:
        plt.close(fig)
    else:
        plt.show()

    return fig, ax, color_by_stim, out_png

## USAGE EXAMPLE:
# fig, ax, used_colors, out_path = plot_stimulus_means(
#     mean_traces=mean_traces,
#     stimuli_ids=stimuli_ids,
#     stimuli_names=stimuli_names,
#     title_prefix="",
#     fps_2p=fps_2p,
#     t_post_s=t_post_s,
#     t_pre_s=t_pre_s,
#     stimuli_durations= stimuli_durations,
#     plots_path=paths["plots_path"],       # Path object or str
#     prefix=paths['prefix'],               # e.g., "exp12_mouseA"
#     dpi=600,
#     save=False,
#     stimuli_colors=stimuli_colors,            # <— pass styles here
#     stimuli_linestyles=stimuli_linestyles,
#     close_after=False,
#     kept_cells = None,  # optional: indices of cells to include
#     comment="jhghjg" # for saving
# )
#

def summarize_durations(stimuli_durations):
    '''''
    Given a dict of stimuli durations, summarize common durations.
    If all stimuli have the same value for a field (within tolerance), keep that value.
    Otherwise, compute the mean across stimuli.
    Returns a dict with summarized durations.
    '''''
    fields = ["static_before_sec", "motion_sec", "total_sec"]
    summary = {}

    for field in fields:
        # collect values for this field across all stimuli
        vals = [d[field] for d in stimuli_durations.values() if field in d]

        if not vals:
            continue

        # if all equal (within floating point tolerance), keep that value
        if np.allclose(vals, vals[0]):
            summary[field] = vals[0]
        else:
            # otherwise use the mean
            summary[field] = float(np.mean(vals))

    return summary
# Example usage:
# summary = summarize_durations(stimuli_durations)
# print(summary)
# # {'static_before_sec': 8.0, 'motion_sec': 8.4, 'total_sec': 16.4}

#
# def plot_allfish_flat_raster(
#     data,
#     trial_aligned_traces,
#     stim_order,
#     stimuli_id_map,
#     stimuli_durations,
#     stimuli_colors,
#     stimuli_linestyles,
#     fps_2p,
#     t_pre_s,
#     combine_mode="concat",
#     *,
#     sort_mode="kmeans",
#     n_clusters=8,
#     random_state=0,
#     neuron_order=None,
#     sort_label=None,
#     is_binary=False,
#     figsize=(8, 6),
#     fish_id="all_fish",
# ):
#     """
#     Plot flattened matrix for all fish with movement onsets and optional sorting.
#
#     Parameters
#     ----------
#     data : array, shape (n_neurons, n_time)
#         Flattened matrix you want to plot.
#     trial_aligned_traces : dict or whatever your compute_move_lines_for_flat_matrix expects
#     stim_order, stimuli_id_map, stimuli_durations, stimuli_colors, stimuli_linestyles :
#         Metadata needed for movement onset lines and legend.
#     fps_2p : float
#         Imaging frame rate (Hz).
#     t_pre_s : float
#         Pre-stimulus window in seconds (used inside compute_move_lines_for_flat_matrix).
#     combine_mode : str
#         How trials were combined when building `data` ("concat", etc.).
#     sort_mode : str
#         Sorting mode for compute_single_sort_order (e.g. "kmeans", "pca", ...).
#     n_clusters : int
#         Number of clusters for kmeans mode.
#     random_state : int
#         Random state for kmeans.
#     neuron_order : 1D array or None
#         If provided, use this neuron order instead of computing a new one.
#     sort_label : str or None
#         Label to show in the plot title. If None, use sort_mode or "custom".
#     is_binary : bool
#         If True, threshold `data` at 0.5 and show as 0/1.
#     figsize : tuple
#         Figure size in inches.
#     fish_id : str
#         Just passed to plott.raster_with_stimuli (for title / annotation).
#
#     Returns
#     -------
#     fig, ax, im, neuron_order
#     """
#     # --- 1) Prepare data for sorting & plotting ---
#     data = np.asarray(data)
#     assert data.ndim == 2, "data must be (n_neurons, n_time)"
#
#     # data_for_sort is (time, neurons) for the raster function
#     data_for_sort = data.T.copy()  # (time, neurons)
#
#     if is_binary:
#         data_for_sort[data_for_sort < 0.5] = 0
#         data_for_sort[data_for_sort >= 0.5] = 1
#
#     n_time, n_neurons = data_for_sort.shape
#
#     # --- 2) Decide which neuron_order to use ---
#     if neuron_order is not None:
#         neuron_order = np.asarray(neuron_order)
#         if neuron_order.ndim != 1 or neuron_order.shape[0] != n_neurons:
#             raise ValueError(
#                 f"Custom neuron_order has shape {neuron_order.shape}, "
#                 f"expected ({n_neurons},)."
#             )
#         label = sort_label or "custom"
#     else:
#         # Compute only the requested sort order
#         neuron_order = compute_single_sort_order(
#             data_for_sort,
#             sort_mode=sort_mode,
#             n_clusters=n_clusters,
#             random_state=random_state,
#         )
#         label = sort_label or sort_mode
#
#     # --- 3) Plot the raster ---
#     fig, ax = plt.subplots(figsize=figsize)
#     im = raster_with_stimuli(
#         ax=ax,
#         data=data_for_sort,       # (time, neurons)
#         fps=fps_2p,
#         fish_id=fish_id,
#         neuron_order=neuron_order,
#         is_binary=is_binary,
#         title_suffix=f"ordered by stimulus type | sort={label}",
#     )
#
#     # --- 4) Movement onset lines ---
#     move_starts_s, move_colors, stim_labels = compute_move_lines_for_flat_matrix(
#         trial_aligned_traces=trial_aligned_traces,
#         stim_order=stim_order,
#         stimuli_id_map=stimuli_id_map,
#         stimuli_durations=stimuli_durations,
#         stimuli_colors=stimuli_colors,
#         fps_2p=fps_2p,
#         t_pre_s=t_pre_s,
#         combine_mode=combine_mode,
#     )
#
#     move_styles = [stimuli_linestyles.get(name, "-") for name in stim_labels]
#
#     for pos_s, color, ls in zip(move_starts_s, move_colors, move_styles):
#         # NOTE: if move_starts_s are in frames, pos_s / fps_2p converts to seconds
#         ax.axvline(pos_s / fps_2p, color=color, linestyle=ls,
#                    alpha=0.9, linewidth=1.0)
#
#     # ax.set_xlabel("Time (s)")
#
#     # --- 5) Movement legend: only what appears, in order of first appearance ---
#     seen = set()
#     handles = []
#     labels = []
#
#     for name in stim_labels:          # this follows actual plotted order
#         if name in seen:
#             continue
#         seen.add(name)
#
#         color = stimuli_colors[name]
#         ls = stimuli_linestyles.get(name, "-")
#
#         (line,) = ax.plot([], [], color=color, linestyle=ls,
#                           label=name, linewidth=2)
#         handles.append(line)
#         labels.append(name)
#
#     mov_legend = ax.legend(
#         handles=handles,
#         labels=labels,
#         title="Movement onset",
#         bbox_to_anchor=(1.2, 1),
#         loc="upper left",
#         borderaxespad=0,
#         frameon=False,
#         fontsize=12,  # legend entry text size
#         title_fontsize=14,  # legend title size
#     )
#
#     # --- 6) Colorbar ---
#     if not is_binary:
#         cbar=fig.colorbar(
#             im, ax=ax,
#             label=r"$\Delta F/F$",
#             fraction=0.17,
#
#         )
#         cbar.set_label(r"$\Delta F/F$", fontsize=14)
#         cbar.ax.tick_params(labelsize=12)  # tick labels size
#     else:
#         cbar=fig.colorbar(
#             im, ax=ax,
#             label=r"Significant activity (0/1)",
#             fraction=0.17,
#             pad=0.01,
#         )
#         cbar.set_label(r"Significant activity (0/1)", fontsize=14)
#         cbar.ax.tick_params(labelsize=12)  # tick labels size
#     return fig, ax, im, neuron_order
#
#
def compute_move_lines_for_flat_matrix(
    trial_aligned_traces,
    stim_order,
    stimuli_id_map,
    stimuli_durations,
    stimuli_colors,
    fps_2p,
    t_pre_s,
    combine_mode="concat",
):
    """
    trial_aligned_traces[stim_id] -> (n_neurons, n_time, n_reps)

    Returns
    -------
    move_starts_s : list of floats
        X positions in SECONDS where motion starts (for ax.axvline).
    move_colors   : list of colors
    stim_labels   : list of stimulus names (e.g. 'FL1', 'FR2', ...)
    """
    # invert map: id -> name
    id_to_name = {v: k for k, v in stimuli_id_map.items()}

    move_starts_s = []
    move_colors   = []
    stim_labels   = []

    frame_offset = 0  # global frame index along the flattened time axis

    for stim_id in stim_order:
        stim_name = id_to_name[stim_id]

        # data for this stimulus
        arr = trial_aligned_traces[stim_id]       # (neurons, time, reps)
        _, n_time, n_reps = arr.shape

        # movement onset INSIDE ONE TRIAL (in frames)
        static_before_sec = stimuli_durations[stim_name]["static_before_sec"]
        move_start_time_s = t_pre_s + static_before_sec
        move_start_in_trial = int(round(move_start_time_s * fps_2p))
        move_start_in_trial = np.clip(move_start_in_trial, 0, n_time - 1)

        if combine_mode == "concat":
            # one vertical line per repetition
            for rep in range(n_reps):
                global_frame = frame_offset + rep * n_time + move_start_in_trial
                move_starts_s.append(global_frame )
                move_colors.append(stimuli_colors[stim_name])
                stim_labels.append(stim_name)

            # this block length in frames
            frame_offset += n_time * n_reps

        elif combine_mode == "mean":
            # we averaged across reps, so only ONE trace per stimulus
            global_frame = frame_offset + move_start_in_trial
            move_starts_s.append(global_frame)
            move_colors.append(stimuli_colors[stim_name])
            stim_labels.append(stim_name)

            frame_offset += n_time

        else:
            raise ValueError("combine_mode must be 'concat' or 'mean'")

    return move_starts_s, move_colors, stim_labels
#
#
# def plot_allfish_flat_raster(
#     data,
#     trial_aligned_traces,
#     stim_order,
#     stimuli_id_map,
#     stimuli_durations,
#     stimuli_colors,
#     stimuli_linestyles,
#     fps_2p,
#     t_pre_s,
#     combine_mode="concat",
#     *,
#     sort_mode="kmeans",
#     n_clusters=8,
#     random_state=0,
#     neuron_order=None,
#     sort_label=None,
#     is_binary=False,
#     figsize=(8, 6),
#     fish_id="all_fish",
#     show_mean_trace=False,
#     mean_height_ratio=1.2,
#     mean_linewidth=1.5,
#     mean_color="black",
#     mean_ylabel="Mean activity",
# ):
#     """
#     Plot flattened matrix for all fish with movement onsets and optional sorting.
#
#     Parameters
#     ----------
#     data : array, shape (n_neurons, n_time)
#         Flattened matrix you want to plot.
#     trial_aligned_traces : dict or whatever your compute_move_lines_for_flat_matrix expects
#     stim_order, stimuli_id_map, stimuli_durations, stimuli_colors, stimuli_linestyles :
#         Metadata needed for movement onset lines and legend.
#     fps_2p : float
#         Imaging frame rate (Hz).
#     t_pre_s : float
#         Pre-stimulus window in seconds (used inside compute_move_lines_for_flat_matrix).
#     combine_mode : str
#         How trials were combined when building `data` ("concat", etc.).
#     sort_mode : str
#         Sorting mode for compute_single_sort_order (e.g. "kmeans", "pca", ...).
#     n_clusters : int
#         Number of clusters for kmeans mode.
#     random_state : int
#         Random state for kmeans.
#     neuron_order : 1D array or None
#         If provided, use this neuron order instead of computing a new one.
#     sort_label : str or None
#         Label to show in the plot title. If None, use sort_mode or "custom".
#     is_binary : bool
#         If True, threshold `data` at 0.5 and show as 0/1.
#     figsize : tuple
#         Figure size in inches.
#     fish_id : str
#         Passed to raster_with_stimuli (for title / annotation).
#     show_mean_trace : bool
#         If True, add a panel below the raster showing the mean across neurons.
#     mean_height_ratio : float
#         Relative height of the lower mean-trace panel.
#     mean_linewidth : float
#         Line width of the mean trace.
#     mean_color : str
#         Color of the mean trace.
#     mean_ylabel : str
#         Y label for the mean trace axis.
#
#     Returns
#     -------
#     fig, ax, im, neuron_order
#         `ax` is the raster axis, for backward compatibility.
#     """
#     # --- 1) Prepare data for sorting & plotting ---
#     data = np.asarray(data)
#     assert data.ndim == 2, "data must be (n_neurons, n_time)"
#
#     # data_for_sort is (time, neurons) for the raster function
#     data_for_sort = data.T.copy()  # (time, neurons)
#
#     if is_binary:
#         data_for_sort[data_for_sort < 0.5] = 0
#         data_for_sort[data_for_sort >= 0.5] = 1
#
#     n_time, n_neurons = data_for_sort.shape
#
#     # --- 2) Decide which neuron_order to use ---
#     if neuron_order is not None:
#         neuron_order = np.asarray(neuron_order)
#         if neuron_order.ndim != 1 or neuron_order.shape[0] != n_neurons:
#             raise ValueError(
#                 f"Custom neuron_order has shape {neuron_order.shape}, "
#                 f"expected ({n_neurons},)."
#             )
#         label = sort_label or "custom"
#     else:
#         neuron_order = compute_single_sort_order(
#             data_for_sort,
#             sort_mode=sort_mode,
#             n_clusters=n_clusters,
#             random_state=random_state,
#         )
#         label = sort_label or sort_mode
#
#     # --- 3) Create figure/axes ---
#     if show_mean_trace:
#         fig, (ax, ax_mean) = plt.subplots(
#             2, 1,
#             figsize=figsize,
#             sharex=True,
#             gridspec_kw={"height_ratios": [6, mean_height_ratio], "hspace": 0.05}
#         )
#     else:
#         fig, ax = plt.subplots(figsize=figsize)
#         ax_mean = None
#
#     # --- 4) Plot the raster ---
#     im = raster_with_stimuli(
#         ax=ax,
#         data=data_for_sort,       # (time, neurons)
#         fps=fps_2p,
#         fish_id=fish_id,
#         neuron_order=neuron_order,
#         is_binary=is_binary,
#         title_suffix=f"ordered by stimulus type | sort={label}",
#     )
#
#     # --- 5) Movement onset lines ---
#     move_starts_s, move_colors, stim_labels = compute_move_lines_for_flat_matrix(
#         trial_aligned_traces=trial_aligned_traces,
#         stim_order=stim_order,
#         stimuli_id_map=stimuli_id_map,
#         stimuli_durations=stimuli_durations,
#         stimuli_colors=stimuli_colors,
#         fps_2p=fps_2p,
#         t_pre_s=t_pre_s,
#         combine_mode=combine_mode,
#     )
#
#     move_styles = [stimuli_linestyles.get(name, "-") for name in stim_labels]
#
#     for pos_s, color, ls in zip(move_starts_s, move_colors, move_styles):
#         ax.axvline(
#             pos_s / fps_2p,
#             color=color,
#             linestyle=ls,
#             alpha=0.9,
#             linewidth=1.0,
#         )
#
#     # --- 6) Optional mean trace panel ---
#     if show_mean_trace:
#         mean_trace = np.nanmean(data, axis=0)   # (n_time,)
#         time_s = np.arange(mean_trace.size) / fps_2p
#
#         ax_mean.plot(
#             time_s,
#             mean_trace,
#             color=mean_color,
#             linewidth=mean_linewidth,
#         )
#
#         for pos_s, color, ls in zip(move_starts_s, move_colors, move_styles):
#             ax_mean.axvline(
#                 pos_s / fps_2p,
#                 color=color,
#                 linestyle=ls,
#                 alpha=0.9,
#                 linewidth=1.0,
#             )
#
#         ax_mean.set_ylabel(mean_ylabel, fontsize=12)
#         ax_mean.set_xlabel("Time (s)", fontsize=12)
#         ax_mean.tick_params(labelsize=10)
#         ax_mean.spines["top"].set_visible(False)
#
#     # --- 7) Movement legend: only what appears, in order of first appearance ---
#     seen = set()
#     handles = []
#     labels = []
#
#     for name in stim_labels:
#         if name in seen:
#             continue
#         seen.add(name)
#
#         color = stimuli_colors[name]
#         ls = stimuli_linestyles.get(name, "-")
#
#         (line,) = ax.plot([], [], color=color, linestyle=ls, label=name, linewidth=2)
#         handles.append(line)
#         labels.append(name)
#
#     mov_legend = ax.legend(
#         handles=handles,
#         labels=labels,
#         title="Movement onset",
#         bbox_to_anchor=(1.2, 1),
#         loc="upper left",
#         borderaxespad=0,
#         frameon=False,
#         fontsize=12,
#         title_fontsize=14,
#     )
#
#     # --- 8) Colorbar ---
#     if not is_binary:
#         cbar = fig.colorbar(
#             im,
#             ax=ax,
#             label=r"$\Delta F/F$",
#             fraction=0.17,
#         )
#         cbar.set_label(r"$\Delta F/F$", fontsize=14)
#         cbar.ax.tick_params(labelsize=12)
#     else:
#         cbar = fig.colorbar(
#             im,
#             ax=ax,
#             label=r"Significant activity (0/1)",
#             fraction=0.17,
#             pad=0.01,
#         )
#         cbar.set_label(r"Significant activity (0/1)", fontsize=14)
#         cbar.ax.tick_params(labelsize=12)
#
#     return fig, ax, im, neuron_order

from matplotlib import gridspec

def plot_allfish_flat_raster(
    data,
    trial_aligned_traces,
    stim_order,
    stimuli_id_map,
    stimuli_durations,
    stimuli_colors,
    stimuli_linestyles,
    fps_2p,
    t_pre_s,
    combine_mode="concat",
    *,
    sort_mode="kmeans",
    n_clusters=8,
    random_state=0,
    neuron_order=None,
    sort_label=None,
    is_binary=False,
    figsize=(8, 6),
    fish_id="all_fish",
    show_mean_trace=False,
    mean_height_ratio=1.2,
    mean_linewidth=1.5,
    mean_color="black",
    mean_ylabel="Mean activity",
    vmin=None,
    vmax=None,
):
    data = np.asarray(data)
    assert data.ndim == 2, "data must be (n_neurons, n_time)"

    data_for_sort = data.T.copy()

    if is_binary:
        data_for_sort[data_for_sort < 0.5] = 0
        data_for_sort[data_for_sort >= 0.5] = 1

    n_time, n_neurons = data_for_sort.shape

    if neuron_order is not None:
        neuron_order = np.asarray(neuron_order)
        if neuron_order.ndim != 1 or neuron_order.shape[0] != n_neurons:
            raise ValueError(
                f"Custom neuron_order has shape {neuron_order.shape}, "
                f"expected ({n_neurons},)."
            )
        label = sort_label or "custom"
    else:
        neuron_order = compute_single_sort_order(
            data_for_sort,
            sort_mode=sort_mode,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        label = sort_label or sort_mode

    # ------------------------------------------------------------------
    # Axes layout: explicitly reserve a separate colorbar column
    # so the raster and mean panels keep identical widths
    # ------------------------------------------------------------------
    if show_mean_trace:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[40, 1.2],
            height_ratios=[6, mean_height_ratio],
            hspace=0.05,
            wspace=0.12,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_mean = fig.add_subplot(gs[1, 0], sharex=ax)
        cax = fig.add_subplot(gs[0, 1])   # colorbar axis only for top panel
    else:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            1, 2,
            width_ratios=[40, 1.2],
            wspace=0.12,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_mean = None
        cax = fig.add_subplot(gs[0, 1])

    # --- raster ---
    im = raster_with_stimuli(
        ax=ax,
        data=data_for_sort,
        fps=fps_2p,
        fish_id=fish_id,
        neuron_order=neuron_order,
        is_binary=is_binary,
        title_suffix=f"ordered by stimulus type | sort={label}",
        vmin=vmin,
        vmax=vmax,
    )

    move_starts_s, move_colors, stim_labels = compute_move_lines_for_flat_matrix(
        trial_aligned_traces=trial_aligned_traces,
        stim_order=stim_order,
        stimuli_id_map=stimuli_id_map,
        stimuli_durations=stimuli_durations,
        stimuli_colors=stimuli_colors,
        fps_2p=fps_2p,
        t_pre_s=t_pre_s,
        combine_mode=combine_mode,
    )

    move_styles = [stimuli_linestyles.get(name, "-") for name in stim_labels]

    for pos_s, color, ls in zip(move_starts_s, move_colors, move_styles):
        ax.axvline(pos_s / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.0)

    # --- mean panel ---
    if show_mean_trace:
        mean_trace = np.nanmean(data, axis=0)
        time_s = np.arange(mean_trace.size) / fps_2p

        ax_mean.plot(time_s, mean_trace, color=mean_color, linewidth=mean_linewidth)

        for pos_s, color, ls in zip(move_starts_s, move_colors, move_styles):
            ax_mean.axvline(pos_s / fps_2p, color=color, linestyle=ls, alpha=0.9, linewidth=1.0)

        ax_mean.set_ylabel(mean_ylabel, fontsize=12)
        ax_mean.set_xlabel("Time (s)", fontsize=12)
        ax_mean.tick_params(labelsize=10)
        ax_mean.spines["top"].set_visible(False)

        # Hide duplicated top x tick labels
        plt.setp(ax.get_xticklabels(), visible=False)

    # --- legend ---
    seen = set()
    handles = []
    labels = []

    for name in stim_labels:
        if name in seen:
            continue
        seen.add(name)

        color = stimuli_colors[name]
        ls = stimuli_linestyles.get(name, "-")
        (line,) = ax.plot([], [], color=color, linestyle=ls, label=name, linewidth=2)
        handles.append(line)
        labels.append(name)

    ax.legend(
        handles=handles,
        labels=labels,
        title="Movement onset",
        bbox_to_anchor=(1.18, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )

    # --- colorbar in dedicated axis ---
    if not is_binary:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r"$\Delta F/F$", fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Significant activity (0/1)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    return fig, ax, im, neuron_order
