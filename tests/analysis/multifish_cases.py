"""Deterministic S07 inputs and readable, exact pre-extraction snapshots."""

from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


TIMING = dict(fps_2p=1.0, t_pre_s=1.0, motion_onset_s=3.0, tau_s=0.0)
FISH_IDS = ["fish_b", "fish_a"]
STIM_IDS = [7, 11, 3, 21]
STIM_LABELS = ["flickL", "boutL", "flickR", "boutR"]
SIDES = {"right": {"bout": "boutR", "flickers": ["flickR"]},
         "left": {"bout": "boutL", "flickers": ["flickL"]}}
STATIC_SETTINGS = dict(fps_2p=1.0, t_pre_s=1.0, static_window_s=2.0,
                       flicker_window_s=2.0, static_center_offset_s=-1.0,
                       flicker_center_offset_s=1.0)


@contextmanager
def trajectory_folder():
    """Use inherited workspace permissions for disposable CSVs on Windows."""
    folder = Path(__file__).resolve().parents[2] / ".test_cache" / f"s07_{uuid4().hex}"
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        for label in STIM_LABELS:
            (folder / f"{label}_trajectory.csv").unlink(missing_ok=True)
        folder.rmdir()


def cohort():
    """Return two fish with nontrivial source-row order and all response classes."""
    result = {}
    for fish_rank, fish_id in enumerate(reversed(FISH_IDS)):
        zscore, raster = {}, {}
        for stimulus_rank, stim_id in enumerate(STIM_IDS):
            values = np.arange(5 * 12 * 4, dtype=float).reshape(5, 12, 4)
            zscore[stim_id] = values / 32 + stimulus_rank + fish_rank / 2
            events = np.zeros((4, 12, 4), dtype=np.uint8)
            events[1:3, 2:4, :] = 1
            events[2:, 4:8, :] = 1
            if stim_id in [11, 21]:
                events[3, 4:8, :] = 0
                events[1, 5:8, :2] = 1
            raster[stim_id] = events
        result[fish_id] = {
            "trial_aligned_traces": zscore,
            "trial_aligned_traces_z_core": zscore,
            "trial_aligned_traces_raster": raster,
            "kept_neuron_indices": np.array([3, 0, 4, 1]),
            "stimuli_ids": STIM_IDS,
            "stimuli_id_map": dict(zip(STIM_LABELS, STIM_IDS)),
            "stimuli_durations": {label: dict(static_before_sec=3.0, motion_sec=4.0,
                total_sec=8.0, total_frames=8, motion_start_frame=3)
                for label in STIM_LABELS},
        }
    return result


def write_trajectories(folder):
    """Write small spatial-authority CSVs for the bout-position notebook path."""
    for label in STIM_LABELS:
        x = [0, 0, 0, 0, 0, 2, 2, 4] if label.startswith("bout") else [2] * 8
        pd.DataFrame({"dot_x": x, "dot_y": [0] * 8, "dot_radius": [1] * 8}).to_csv(
            Path(folder) / f"{label}_trajectory.csv", index=False)


def encode(value):
    """Encode exact values, ordering, types, axes, dtypes and categories as JSON."""
    if isinstance(value, pd.DataFrame):
        return {"type": "DataFrame", "index": encode(value.index),
                "columns": encode(value.columns),
                "data": [encode(value.iloc[:, i]) for i in range(value.shape[1])]}
    if isinstance(value, pd.Series):
        result = {"type": "Series", "dtype": str(value.dtype), "name": encode(value.name),
                  "values": encode(value.tolist())}
        if isinstance(value.dtype, pd.CategoricalDtype):
            result.update(categories=encode(value.cat.categories), ordered=value.cat.ordered)
        return result
    if isinstance(value, pd.Index):
        return {"type": type(value).__name__, "dtype": str(value.dtype),
                "name": encode(value.name), "values": encode(value.tolist())}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "values": encode(value.tolist())}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, dict):
        return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [encode(v) for v in value]}
    if isinstance(value, float) and not np.isfinite(value):
        return {"type": "float", "value": str(value)}
    return value


def cases(api, folder):
    """Run every S07 public family using the historical facade call patterns."""
    data = cohort()
    write_trajectories(folder)
    outputs = {}
    for mode in ["concat", "mean"]:
        for kind in ["dfof", "raster", "zscore"]:
            outputs[f"matrix_{mode}_{kind}"] = api.build_matrix_all_fish(
                data, STIM_IDS, fish_ids=FISH_IDS, combine_mode=mode, trace_type=kind)
        outputs[f"widths_{mode}"] = api.describe_matrix_widths(data, STIM_IDS, FISH_IDS, mode)
    outputs["string_keys"] = api.build_matrix_for_fish(
        {str(k): v for k, v in data["fish_b"]["trial_aligned_traces"].items()},
        STIM_IDS, [4, 0], "mean")
    outputs["responses"] = response = api.build_zscore_response_matrices_all_fish(
        data, STIM_LABELS, fish_ids=FISH_IDS + [FISH_IDS[0]], **TIMING)
    outputs["empty_responses"] = api.build_zscore_response_matrices_all_fish({}, [])
    outputs["active_counts"] = active_counts = api.build_active_neuron_matrices_all_fish(
        data, FISH_IDS, STIM_IDS, return_counts=True, **TIMING)
    active = active_counts[0]
    outputs["empty_active"] = api.build_active_neuron_matrices_all_fish({}, return_counts=True)
    summary = api.build_neuron_stimulus_summary_table(
        response["response_matrices"], active, response["row_metadata"],
        STIM_IDS, STIM_LABELS, "characterization")
    outputs["summary"] = summary
    outputs["metrics"] = metrics = api.add_selectivity_metrics_to_summary_table(summary, STIM_LABELS)
    outputs["classes"] = api.classify_stimulus_specificity_summary_table(
        metrics, STIM_LABELS, return_thresholds=True)
    outputs["specificity_order"] = api.build_stimulus_specificity_neuron_order(metrics, STIM_LABELS)
    outputs["diagnostic_order"] = api.build_stimulus_specificity_neuron_order(
        metrics, STIM_LABELS, response["row_metadata"].iloc[::-1])
    outputs["similarity"] = api.build_stimulus_vector_similarity(
        response["pooled_response_matrix"], STIM_LABELS[::-1])
    outputs["similarity_nonfinite"] = api.build_stimulus_vector_similarity(pd.DataFrame(
        {"nan": [np.nan, np.nan, np.nan], "zero": [0., 0., 0.], "mixed": [-1., np.nan, 2.]}))
    outputs["permutation"] = api.build_segment_selectivity_permutation_summary(
        data, FISH_IDS, STIM_IDS, STIM_LABELS, ["flickL", "flickR"],
        n_permutations=13, random_seed=19, **TIMING)
    for mode in ["mean", "concat"]:
        outputs[f"diagnostic_{mode}"] = api.build_pooled_active_trace_diagnostic(
            data, active, STIM_IDS, FISH_IDS + [FISH_IDS[0]], mode)
    outputs["empty_diagnostic"] = api.build_pooled_active_trace_diagnostic({}, {}, STIM_IDS)
    outputs["overlap"] = api.build_active_neuron_overlap_matrices_all_fish(
        active, {"right": [3, 21], "left": [7, 11]}, ["flicker", "bout"])
    outputs["empty_overlap"] = api.build_active_neuron_overlap_matrices_all_fish(
        {}, {"left": [7, 11]}, ["flicker", "bout"], empty_union_value=0.0)
    outputs["array_overlap"] = api.compute_active_neuron_jaccard_overlap(
        np.array([[1, 0, 0], [1, 1, 0]]), [2, 0, 1])
    outputs["static"] = static = api.build_static_flicker_recruitment_analysis(
        data, FISH_IDS, {"right": [3], "left": [7]}, **STATIC_SETTINGS)
    categorical = static["neuron_stimulus_metrics"].copy()
    categorical["category"] = pd.Categorical(categorical["category"], ordered=True,
        categories=["shared", "newly recruited", "static-only", "non-responsive", "unused"])
    outputs["categorical_summary"] = api.summarize_pooled_static_flicker_categories(categorical)
    outputs["statistics"] = api.compute_static_flicker_fish_level_statistics(pd.DataFrame({
        "fish_id": ["b", "a", "c"] * 2, "side": ["left"] * 6,
        "stim_id": [7] * 3 + [3] * 3, "stimulus": ["second"] * 3 + ["first"] * 3,
        "median_delta_auc": [0., np.nan, 2., 1., 3., -1.]}))
    for intersection in [False, True]:
        outputs[f"bout_{intersection}"] = api.build_bout_flicker_position_analysis(
            data, FISH_IDS, SIDES, folder, require_bout_and_flicker_active=intersection,
            position_window_offsets_s=(-1., 1.), flicker_window_offsets_s=(0., 2.),
            onset_persistence_frames=2, **TIMING)
    return outputs
