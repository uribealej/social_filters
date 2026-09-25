"""Static-flicker recruitment summaries and fish-level statistical inference."""

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.response_metrics import compute_static_flicker_trial_metrics


def build_static_flicker_recruitment_analysis(
    all_fish_data,
    fish_ids,
    side_stimuli,
    fps_2p=2.0,
    t_pre_s=5.0,
    static_window_s=4.0,
    flicker_window_s=4.0,
    static_center_offset_s=-4.0,
    flicker_center_offset_s=4.0,
    min_consecutive_active_frames=2,
    min_active_trial_fraction=0.5,
):
    """Build static--flicker recruitment outputs across a configured fish cohort."""
    per_fish = {}
    for fish_id in fish_ids:
        if fish_id not in all_fish_data:
            raise KeyError(f"Fish {fish_id!r} is not present in all_fish_data.")
        fish = all_fish_data[fish_id]
        per_fish[fish_id] = compute_static_flicker_trial_metrics(
            trial_aligned_traces_zscore=fish["trial_aligned_traces_z_core"],
            trial_aligned_traces_raster=fish["trial_aligned_traces_raster"],
            side_stimuli=side_stimuli,
            stimuli_durations=fish["stimuli_durations"],
            stimuli_id_map=fish["stimuli_id_map"],
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            static_window_s=static_window_s,
            flicker_window_s=flicker_window_s,
            static_center_offset_s=static_center_offset_s,
            flicker_center_offset_s=flicker_center_offset_s,
            min_consecutive_active_frames=min_consecutive_active_frames,
            min_active_trial_fraction=min_active_trial_fraction,
            kept_neuron_indices=fish.get("kept_neuron_indices"),
            fish_id=fish_id,
        )

    table_names = [
        "trial_metrics",
        "stimulus_metrics",
        "neuron_stimulus_metrics",
        "shared_neuron_metrics",
        "window_validation",
        "classification_raster_data",
    ]
    combined = {
        name: pd.concat([per_fish[fish_id][name] for fish_id in fish_ids], ignore_index=True)
        for name in table_names
    }
    neuron_metrics = combined["neuron_stimulus_metrics"]
    valid = neuron_metrics.loc[neuron_metrics["valid_neuron"]].copy()
    categories = ["non-responsive", "static-only", "shared", "newly recruited"]
    category_counts = (
        valid.groupby(["fish_id", "side", "stim_id", "stimulus", "category"], observed=False)
        .size()
        .unstack("category", fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    valid_counts = valid.groupby(["fish_id", "side", "stim_id", "stimulus"]).size().rename("valid_neurons")
    fish_side_summary = category_counts.join(valid_counts).reset_index()
    for category in categories:
        fish_side_summary[f"{category}_proportion"] = (
            fish_side_summary[category] / fish_side_summary["valid_neurons"].replace(0, np.nan)
        )
    fish_side_summary["newly_recruited_over_valid"] = fish_side_summary[
        "newly recruited_proportion"
    ]
    flicker_active = valid.groupby(["fish_id", "side", "stim_id", "stimulus"])["flicker_active"].sum().rename("flicker_active_neurons")
    fish_side_summary = fish_side_summary.merge(
        flicker_active.reset_index(), on=["fish_id", "side", "stim_id", "stimulus"], how="left", validate="one_to_one"
    )
    fish_side_summary["newly_recruited_over_flicker_active"] = (
        fish_side_summary["newly recruited"]
        / fish_side_summary["flicker_active_neurons"].replace(0, np.nan)
    )
    pooled_category_summary = summarize_pooled_static_flicker_categories(neuron_metrics)

    shared = combined["shared_neuron_metrics"].copy()
    shared_median_delta_auc = (
        shared.groupby(["fish_id", "side", "stim_id", "stimulus"], as_index=False)["delta_auc"]
        .median()
        .rename(columns={"delta_auc": "median_delta_auc"})
    )
    fish_median_delta_auc = fish_side_summary[["fish_id", "side", "stim_id", "stimulus"]].merge(
        shared_median_delta_auc,
        on=["fish_id", "side", "stim_id", "stimulus"],
        how="left",
        validate="one_to_one",
    )
    shared_mean_delta_auc = (
        shared.groupby(["fish_id", "side", "stim_id", "stimulus"], as_index=False)["delta_auc"]
        .mean()
        .rename(columns={"delta_auc": "mean_delta_auc"})
    )
    fish_mean_delta_auc = fish_side_summary[["fish_id", "side", "stim_id", "stimulus"]].merge(
        shared_mean_delta_auc,
        on=["fish_id", "side", "stim_id", "stimulus"],
        how="left",
        validate="one_to_one",
    )
    fish_level_statistics = compute_static_flicker_fish_level_statistics(fish_median_delta_auc)
    recruited = valid.loc[valid["category"] == "newly recruited"]
    shared_valid = valid.loc[valid["category"] == "shared"]
    recruitment = recruited.groupby(["fish_id", "side", "stim_id", "stimulus"])["flicker_auc"].sum().rename("recruitment")
    amplification = shared_valid.groupby(["fish_id", "side", "stim_id", "stimulus"])["delta_auc"].sum().rename("amplification")
    recruitment_amplification = (
        fish_side_summary[["fish_id", "side", "stim_id", "stimulus"]]
        .merge(recruitment.reset_index(), on=["fish_id", "side", "stim_id", "stimulus"], how="left")
        .merge(amplification.reset_index(), on=["fish_id", "side", "stim_id", "stimulus"], how="left")
        .fillna({"recruitment": 0.0, "amplification": 0.0})
    )
    return {
        **combined,
        "fish_side_summary": fish_side_summary,
        "pooled_category_summary": pooled_category_summary,
        "fish_median_delta_auc": fish_median_delta_auc,
        "fish_mean_delta_auc": fish_mean_delta_auc,
        "fish_level_statistics": fish_level_statistics,
        "recruitment_amplification": recruitment_amplification,
        "per_fish": per_fish,
    }


def summarize_pooled_static_flicker_categories(neuron_stimulus_metrics):
    """Return descriptive category counts/proportions after pooling cells across fish.

    This is deliberately a visualization summary: fish remain the biological
    replicate in ``fish_side_summary`` and related inferential summaries.
    """
    metrics = pd.DataFrame(neuron_stimulus_metrics).copy()
    valid = metrics.loc[metrics["valid_neuron"]].copy()
    categories = ["non-responsive", "static-only", "shared", "newly recruited"]
    group_columns = ["side", "stim_id", "stimulus"]
    counts = (
        valid.groupby(group_columns + ["category"], observed=False)
        .size()
        .unstack("category", fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    summary = counts.reset_index()
    summary["pooled_valid_neurons"] = summary[categories].sum(axis=1)
    for category in categories:
        summary[f"{category}_proportion"] = (
            summary[category] / summary["pooled_valid_neurons"].replace(0, np.nan)
        )
    summary["n_fish"] = (
        valid.groupby(group_columns)["fish_id"].nunique().reindex(counts.index).to_numpy()
    )
    return summary


def _exact_sign_flip_pvalue(values):
    """Two-sided exact sign-flip p-value for a one-sample fish-level effect."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    observed = abs(float(np.mean(values)))
    null_statistics = np.asarray([
        abs(float(np.mean(values * signs))) for signs in product((-1.0, 1.0), repeat=values.size)
    ])
    return float(np.mean(null_statistics >= observed - np.finfo(float).eps))


def _wilcoxon_signed_rank_pvalue(values, alternative="two-sided"):
    """Wilcoxon signed-rank p-value, robust to zero differences."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size or np.allclose(values, 0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="wilcox", alternative=alternative, method="auto").pvalue)
    except ValueError:
        return np.nan


def _bootstrap_median_ci(values, n_boot=10000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan
    draws = np.random.default_rng(seed).choice(values, size=(int(n_boot), values.size), replace=True)
    return tuple(np.quantile(np.median(draws, axis=1), [0.025, 0.975]))


def _holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    ranked = finite[np.argsort(values[finite])]
    running = 0.0
    total = len(ranked)
    for rank, index in enumerate(ranked):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def compute_static_flicker_fish_level_statistics(fish_median_delta_auc):
    """Summarize fish medians with one-sided zero tests and two-sided contrasts."""
    data = pd.DataFrame(fish_median_delta_auc).copy()
    rows = []
    for side, side_data in data.groupby("side", sort=False):
        positions = side_data[["stim_id", "stimulus"]].drop_duplicates().sort_values("stim_id")
        for position in positions.itertuples():
            values = side_data.loc[side_data["stim_id"] == position.stim_id, "median_delta_auc"].dropna().to_numpy(float)
            ci_low, ci_high = _bootstrap_median_ci(values, seed=int(position.stim_id))
            rows.append({
                "side": side, "test": "ΔAUC > 0", "stimulus_a": position.stimulus,
                "stimulus_b": "", "n_fish": len(values), "effect_median": np.median(values) if len(values) else np.nan,
                "ci_low": ci_low, "ci_high": ci_high, "p_wilcoxon": _wilcoxon_signed_rank_pvalue(values, alternative="greater"),
            })
        for first_index in range(len(positions)):
            for second_index in range(first_index + 1, len(positions)):
                first, second = positions.iloc[first_index], positions.iloc[second_index]
                paired = side_data.loc[side_data["stim_id"].isin([first.stim_id, second.stim_id])]
                paired = paired.pivot(index="fish_id", columns="stim_id", values="median_delta_auc").dropna()
                differences = (paired[second.stim_id] - paired[first.stim_id]).to_numpy(float)
                ci_low, ci_high = _bootstrap_median_ci(differences, seed=int(first.stim_id * 100 + second.stim_id))
                rows.append({
                    "side": side, "test": "paired position contrast", "stimulus_a": first.stimulus,
                    "stimulus_b": second.stimulus, "n_fish": len(differences),
                    "effect_median": np.median(differences) if len(differences) else np.nan,
                    "ci_low": ci_low, "ci_high": ci_high, "p_wilcoxon": _wilcoxon_signed_rank_pvalue(differences),
                })
    results = pd.DataFrame(rows)
    results["p_holm"] = np.nan
    for side, indices in results.groupby("side").groups.items():
        results.loc[list(indices), "p_holm"] = _holm_adjust(results.loc[list(indices), "p_wilcoxon"])
    return results
