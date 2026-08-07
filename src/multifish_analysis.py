"""Pure matrix helpers for multi-fish calcium analysis notebooks."""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.analysis_tools import (
    build_active_neuron_matrix_from_trial_raster,
    classify_stimulus_specificity_neuron,
    compute_static_flicker_trial_metrics,
    compute_response_window_frames,
    compute_stimulus_selectivity_metrics,
    compute_trial_auc_by_neuron,
    compute_zscore_response_auc,
    resolve_selected_stimuli,
)


def build_bout_flicker_position_metadata(
    stimuli_path,
    stimuli_durations,
    side_conditions,
    matching_method="euclidean",
    coordinate_dot_prefix=None,
):
    """Match stationary flicker coordinates to positions in bout trajectories.

    Trajectory CSV files are the spatial authority.  The loader-derived timing
    dictionaries provide the stimulus-frame timing authority, keeping this
    helper aligned with the existing analysis pipeline.  Position indices in
    the returned table are zero-based, in trajectory visit order.
    """
    if matching_method != "euclidean":
        raise ValueError("matching_method must be 'euclidean'.")
    if set(side_conditions) != {"left", "right"}:
        raise ValueError("side_conditions must contain exactly 'left' and 'right'.")

    stimuli_path = Path(stimuli_path)
    if not stimuli_path.is_dir():
        raise FileNotFoundError(f"Stimulus trajectory folder was not found: {stimuli_path}")

    tables = []
    trajectories = {}
    for side, conditions in side_conditions.items():
        if not {"bout", "flickers"}.issubset(conditions):
            raise ValueError(f"{side} conditions need 'bout' and 'flickers' entries.")
        bout = str(conditions["bout"])
        flickers = [str(stimulus) for stimulus in conditions["flickers"]]
        if not flickers:
            raise ValueError(f"{side} conditions must include at least one flicker stimulus.")

        bout_table = _read_bout_flicker_trajectory(
            stimuli_path / f"{bout}_trajectory.csv", coordinate_dot_prefix
        )
        duration = _require_bout_flicker_timing(stimuli_durations, bout)
        positions = _extract_visible_bout_positions(bout_table, duration, bout)
        trajectories[side] = {
            "bout_stimulus": bout,
            "positions": positions,
            "motion_onset_s": float(duration["static_before_sec"]),
            "stimulus_fps": float(duration["total_frames"]) / float(duration["total_sec"]),
        }

        for flicker in flickers:
            flicker_table = _read_bout_flicker_trajectory(
                stimuli_path / f"{flicker}_trajectory.csv", coordinate_dot_prefix
            )
            _require_bout_flicker_timing(stimuli_durations, flicker)
            flicker_coordinate = _stationary_flicker_coordinate(flicker_table, flicker)
            distances = np.linalg.norm(
                positions[["x", "y"]].to_numpy(float) - flicker_coordinate[None, :], axis=1
            )
            matched_row = positions.iloc[int(np.argmin(distances))]
            tables.append({
                "hemifield": side,
                "flicker_stimulus": flicker,
                "flicker_x": float(flicker_coordinate[0]),
                "flicker_y": float(flicker_coordinate[1]),
                "nearest_bout_position_index": int(matched_row["position_index"]),
                "nearest_bout_x": float(matched_row["x"]),
                "nearest_bout_y": float(matched_row["y"]),
                "spatial_matching_error": float(np.min(distances)),
                "time_after_motion_onset_s": float(matched_row["time_after_motion_onset_s"]),
            })

    validation = pd.DataFrame(tables)
    if validation.empty:
        raise ValueError("No bout/flicker position matches were produced.")
    return {"validation_table": validation, "trajectories": trajectories}


def _read_bout_flicker_trajectory(path, coordinate_dot_prefix=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stimulus trajectory file was not found: {path}")
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f"Stimulus trajectory file is empty: {path}")

    coordinate_pairs = {}
    for column in table.columns:
        if str(column).endswith("_x"):
            prefix = str(column)[:-2]
            y_column = f"{prefix}_y"
            if y_column in table.columns:
                coordinate_pairs[prefix] = (str(column), y_column)
    if coordinate_dot_prefix is not None:
        prefix = str(coordinate_dot_prefix)
        if prefix not in coordinate_pairs:
            raise ValueError(
                f"{path.name} does not contain coordinate columns for dot prefix {prefix!r}."
            )
    elif len(coordinate_pairs) == 1:
        prefix = next(iter(coordinate_pairs))
    else:
        raise ValueError(
            f"Could not identify one coordinate pair in {path.name}; found {sorted(coordinate_pairs)}. "
            "Set coordinate_dot_prefix explicitly."
        )

    x_column, y_column = coordinate_pairs[prefix]
    radius_column = f"{prefix}_radius"
    if radius_column not in table.columns:
        raise ValueError(
            f"Could not identify the radius/visibility column {radius_column!r} in {path.name}."
        )
    result = table[[x_column, y_column, radius_column]].rename(
        columns={x_column: "x", y_column: "y", radius_column: "radius"}
    )
    result = result.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result[["x", "y"]].to_numpy(float)).any():
        raise ValueError(f"Coordinate columns in {path.name} contain no finite positions.")
    return result


def _require_bout_flicker_timing(stimuli_durations, stimulus):
    if stimulus not in stimuli_durations:
        raise ValueError(f"No timing metadata was loaded for stimulus {stimulus!r}.")
    duration = dict(stimuli_durations[stimulus])
    required = ("static_before_sec", "total_sec", "total_frames", "motion_start_frame")
    missing = [name for name in required if name not in duration]
    if missing:
        raise ValueError(
            f"Timing metadata for {stimulus!r} is missing {missing}; cannot map trajectory frames to time."
        )
    if float(duration["total_sec"]) <= 0 or int(duration["total_frames"]) <= 0:
        raise ValueError(f"Timing metadata for {stimulus!r} has a non-positive duration or frame count.")
    return duration


def _extract_visible_bout_positions(table, duration, stimulus):
    start_frame = int(duration["motion_start_frame"])
    if start_frame < 0 or start_frame >= len(table):
        raise ValueError(
            f"Motion-start frame {start_frame} for {stimulus!r} is outside its trajectory with {len(table)} rows."
        )
    stimulus_fps = float(duration["total_frames"]) / float(duration["total_sec"])
    visible = (
        np.isfinite(table[["x", "y", "radius"]].to_numpy(float)).all(axis=1)
        & (table["radius"].to_numpy(float) > 0)
    )
    positions = []
    previous = None
    for frame in range(start_frame, len(table)):
        if not visible[frame]:
            continue
        coordinate = table.loc[frame, ["x", "y"]].to_numpy(float)
        if previous is None or not np.allclose(coordinate, previous, atol=1e-9, rtol=0):
            positions.append({
                "position_index": len(positions),
                "trajectory_frame": int(frame),
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
                "time_after_motion_onset_s": (float(frame) - start_frame) / stimulus_fps,
            })
            previous = coordinate
    if not positions:
        raise ValueError(f"No visible bout positions were found after motion onset for {stimulus!r}.")
    return pd.DataFrame(positions)


def _stationary_flicker_coordinate(table, stimulus):
    visible = (
        np.isfinite(table[["x", "y", "radius"]].to_numpy(float)).all(axis=1)
        & (table["radius"].to_numpy(float) > 0)
    )
    coordinates = table.loc[visible, ["x", "y"]].to_numpy(float)
    if not len(coordinates):
        raise ValueError(f"No visible stationary coordinate was found for flicker {stimulus!r}.")
    if not np.allclose(coordinates, coordinates[0], atol=1e-9, rtol=0):
        raise ValueError(
            f"Flicker {stimulus!r} contains more than one visible spatial coordinate; it is not stationary."
        )
    return coordinates[0]


def build_bout_flicker_position_analysis(
    all_fish_data,
    fish_ids,
    side_conditions,
    stimuli_path,
    fps_2p=2.0,
    t_pre_s=5.0,
    active_fraction_threshold=0.10,
    min_epoch_s=1.0,
    min_active_reps=2,
    expected_reps=4,
    require_expected_reps=True,
    motion_onset_s=8.0,
    tau_s=6.0,
    onset_persistence_frames=1,
    onset_min_significant_trial_fraction=None,
    require_bout_and_flicker_active=False,
    matching_method="euclidean",
    coordinate_dot_prefix=None,
    position_window_offsets_s=(-1.0, 2.0),
    flicker_window_offsets_s=(0.0, 3.0),
):
    """Build pooled, bout-referenced flicker-position comparison data.

    The active-neuron decision is delegated unchanged to
    :func:`build_active_neuron_matrices_all_fish`, which is also used by the
    Exp 1 Cell 06 diagnostic.  Only the temporal onset/order and descriptive
    bout-versus-flicker summaries are added here.
    """
    fish_ids = list(dict.fromkeys(fish_ids))
    if not fish_ids:
        raise ValueError("fish_ids must contain at least one fish.")
    if onset_persistence_frames < 1:
        raise ValueError("onset_persistence_frames must be >= 1.")
    if onset_min_significant_trial_fraction is None:
        if expected_reps is None:
            raise ValueError(
                "expected_reps is required when onset_min_significant_trial_fraction is not set."
            )
        onset_min_significant_trial_fraction = float(min_active_reps) / float(expected_reps)
    if not 0 < float(onset_min_significant_trial_fraction) <= 1:
        raise ValueError("onset_min_significant_trial_fraction must be in (0, 1].")
    position_window_offsets_s = _validate_bout_flicker_window(
        position_window_offsets_s, "position_window_offsets_s"
    )
    flicker_window_offsets_s = _validate_bout_flicker_window(
        flicker_window_offsets_s, "flicker_window_offsets_s"
    )
    reference_fish = all_fish_data[fish_ids[0]]
    metadata = build_bout_flicker_position_metadata(
        stimuli_path=stimuli_path,
        stimuli_durations=reference_fish["stimuli_durations"],
        side_conditions=side_conditions,
        matching_method=matching_method,
        coordinate_dot_prefix=coordinate_dot_prefix,
    )

    all_stimuli = []
    resolved = {}
    for side, conditions in side_conditions.items():
        selection = resolve_selected_stimuli(
            [conditions["bout"], *conditions["flickers"]],
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_raster"].keys(),
        )
        resolved[side] = selection
        all_stimuli.extend(selection["stimulus_ids"])
    all_stimuli = list(dict.fromkeys(all_stimuli))
    active_matrices = build_active_neuron_matrices_all_fish(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        stim_order=all_stimuli,
        fps_2p=fps_2p,
        t_pre_s=t_pre_s,
        motion_onset_s=motion_onset_s,
        active_fraction_threshold=active_fraction_threshold,
        min_epoch_s=min_epoch_s,
        min_active_reps=min_active_reps,
        expected_reps=expected_reps,
        require_expected_reps=require_expected_reps,
        tau_s=tau_s,
    )

    sides = {}
    for side, selection in resolved.items():
        labels = selection["stimulus_labels"]
        stimulus_ids = selection["stimulus_ids"]
        bout_label = str(side_conditions[side]["bout"])
        bout_id = stimulus_ids[labels.index(bout_label)]
        flicker_labels = [str(value) for value in side_conditions[side]["flickers"]]
        flicker_ids = [stimulus_ids[labels.index(label)] for label in flicker_labels]
        sides[side] = _build_bout_flicker_side_data(
            all_fish_data=all_fish_data,
            fish_ids=fish_ids,
            active_matrices=active_matrices,
            side=side,
            bout_id=bout_id,
            bout_label=bout_label,
            flicker_ids=flicker_ids,
            flicker_labels=flicker_labels,
            validation_table=metadata["validation_table"],
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            motion_onset_s=motion_onset_s,
            tau_s=tau_s,
            onset_persistence_frames=int(onset_persistence_frames),
            onset_min_significant_trial_fraction=float(onset_min_significant_trial_fraction),
            require_bout_and_flicker_active=bool(require_bout_and_flicker_active),
            position_window_offsets_s=position_window_offsets_s,
            flicker_window_offsets_s=flicker_window_offsets_s,
        )

    return {
        "position_metadata": metadata,
        "active_matrices": active_matrices,
        "sides": sides,
        "settings": {
            "active_fraction_threshold": active_fraction_threshold,
            "min_epoch_s": min_epoch_s,
            "min_active_reps": min_active_reps,
            "expected_reps": expected_reps,
            "require_expected_reps": require_expected_reps,
            "motion_onset_s": motion_onset_s,
            "tau_s": tau_s,
            "onset_persistence_frames": onset_persistence_frames,
            "onset_min_significant_trial_fraction": onset_min_significant_trial_fraction,
            "require_bout_and_flicker_active": require_bout_and_flicker_active,
            "position_window_offsets_s": position_window_offsets_s,
            "flicker_window_offsets_s": flicker_window_offsets_s,
        },
    }


def _validate_bout_flicker_window(offsets, name):
    values = tuple(float(value) for value in offsets)
    if len(values) != 2 or not values[0] < values[1]:
        raise ValueError(f"{name} must contain exactly two increasing offsets.")
    return values


def _trace_for_bout_flicker(fish, trace_key, stimulus, expected_rows=None):
    traces = fish[trace_key]
    key = stimulus if stimulus in traces else str(stimulus)
    if key not in traces:
        raise KeyError(f"Stimulus {stimulus!r} is missing from {trace_key}.")
    values = np.asarray(traces[key], dtype=float)
    if values.ndim != 3:
        raise ValueError(
            f"Stimulus {stimulus!r} in {trace_key} has shape {values.shape}; expected (neurons, time, reps)."
        )
    if expected_rows is not None and values.shape[0] != expected_rows:
        kept = np.asarray(fish.get("kept_neuron_indices", []), dtype=int).ravel()
        if (
            kept.size == expected_rows
            and kept.size > 0
            and kept.min() >= 0
            and kept.max() < values.shape[0]
        ):
            values = values[kept]
        else:
            raise ValueError(
                f"Stimulus {stimulus!r} has {values.shape[0]} z-score rows but {expected_rows} significant-raster rows; "
                "kept_neuron_indices cannot reconcile them."
            )
    return values


def _first_persistent_significant_time(
    significant_trace,
    time_relative_s,
    persistence_frames,
    min_significant_trial_fraction,
):
    active = (
        (np.asarray(significant_trace, dtype=float) >= float(min_significant_trial_fraction))
        & (np.asarray(time_relative_s) >= 0)
    )
    if persistence_frames == 1:
        hits = np.flatnonzero(active)
        return float(time_relative_s[hits[0]]) if hits.size else np.inf
    run = 0
    for index, value in enumerate(active):
        run = run + 1 if value else 0
        if run >= persistence_frames:
            return float(time_relative_s[index - persistence_frames + 1])
    return np.inf


def _window_frames(time_relative_s, center_s, offsets_s, stimulus):
    start_s, stop_s = float(center_s) + offsets_s[0], float(center_s) + offsets_s[1]
    frames = np.flatnonzero((time_relative_s >= start_s) & (time_relative_s < stop_s))
    if not frames.size:
        raise ValueError(
            f"{stimulus!r} cannot provide the requested summary window {start_s:g}..{stop_s:g} s."
        )
    return frames


def _build_bout_flicker_side_data(
    all_fish_data,
    fish_ids,
    active_matrices,
    side,
    bout_id,
    bout_label,
    flicker_ids,
    flicker_labels,
    validation_table,
    fps_2p,
    t_pre_s,
    motion_onset_s,
    tau_s,
    onset_persistence_frames,
    onset_min_significant_trial_fraction,
    require_bout_and_flicker_active,
    position_window_offsets_s,
    flicker_window_offsets_s,
):
    stimulus_ids = [bout_id, *flicker_ids]
    stimulus_labels = [bout_label, *flicker_labels]
    pooled = {label: {"zscore": [], "significant": []} for label in stimulus_labels}
    order_rows = []
    fish_blocks = []
    fish_decisions = {}

    for fish_rank, fish_id in enumerate(fish_ids):
        fish = all_fish_data[fish_id]
        active = active_matrices[fish_id]
        active_columns = [_active_column(active, stimulus) for stimulus in stimulus_ids]
        fish_decisions[fish_id] = {
            label: active_column for label, active_column in zip(stimulus_labels, active_columns)
        }
        flicker_active = np.logical_or.reduce(active_columns[1:])
        selected_mask = (
            active_columns[0] & flicker_active
            if require_bout_and_flicker_active
            else np.logical_or(active_columns[0], flicker_active)
        )
        selected_rows = np.flatnonzero(selected_mask)
        if not selected_rows.size:
            continue

        zscore = {}
        significant = {}
        time_relative = {}
        for stimulus, label in zip(stimulus_ids, stimulus_labels):
            raster = _trace_for_bout_flicker(fish, "trial_aligned_traces_raster", stimulus)
            zvalues = _trace_for_bout_flicker(
                fish, "trial_aligned_traces_z_core", stimulus, expected_rows=raster.shape[0]
            )
            if zvalues.shape[1:] != raster.shape[1:]:
                raise ValueError(
                    f"Fish {fish_id!r}, stimulus {label!r} has incompatible z-score/raster time shapes "
                    f"{zvalues.shape} and {raster.shape}."
                )
            onset = _stimulus_motion_onset(fish["stimuli_durations"], label, motion_onset_s)
            zscore[label] = np.nanmean(zvalues, axis=2)
            significant[label] = np.nanmean(raster, axis=2)
            time_relative[label] = np.arange(raster.shape[1], dtype=float) / float(fps_2p) - float(t_pre_s) - onset

        bout_onsets = np.asarray([
            _first_persistent_significant_time(
                significant[bout_label][row], time_relative[bout_label], onset_persistence_frames,
                onset_min_significant_trial_fraction,
            )
            for row in selected_rows
        ])
        flicker_onsets = {
            label: np.asarray([
                _first_persistent_significant_time(
                    significant[label][row], time_relative[label], onset_persistence_frames,
                    onset_min_significant_trial_fraction,
                )
                for row in selected_rows
            ])
            for label in flicker_labels
        }
        original_ids = np.asarray(fish.get("kept_neuron_indices", np.arange(significant[bout_label].shape[0])), dtype=int)
        if original_ids.size != significant[bout_label].shape[0]:
            original_ids = np.arange(significant[bout_label].shape[0], dtype=int)

        for local_index, raster_row in enumerate(selected_rows):
            is_bout_active = bool(active_columns[0][raster_row])
            preferred_index = int(np.argmax([
                np.nanmean(zscore[label][raster_row, time_relative[label] >= 0])
                for label in flicker_labels
            ]))
            preferred = flicker_labels[preferred_index]
            order_rows.append({
                "fish_id": fish_id,
                "fish_rank": fish_rank,
                "raster_row": int(raster_row),
                "neuron_id": int(original_ids[raster_row]),
                "bout_active": is_bout_active,
                "bout_response_onset_s": float(bout_onsets[local_index]),
                "preferred_flicker": preferred,
                "preferred_flicker_rank": preferred_index,
                "preferred_flicker_onset_s": float(flicker_onsets[preferred][local_index]),
            })
        fish_blocks.append((fish_id, selected_rows, zscore, significant, time_relative))

    order = pd.DataFrame(order_rows)
    if order.empty:
        requirement = "for both bout and at least one flicker" if require_bout_and_flicker_active else "for bout or flicker"
        raise ValueError(f"No {side} neurons were active {requirement} under the existing Cell 06 criteria.")
    order["group"] = np.where(order["bout_active"], "bout-active", "flicker-only")
    order["group_rank"] = np.where(order["bout_active"], 0, 1)
    order = order.sort_values(
        [
            "group_rank", "bout_response_onset_s", "preferred_flicker_rank",
            "preferred_flicker_onset_s", "fish_rank", "neuron_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    order["display_row"] = np.arange(len(order), dtype=int)

    fish_values = {
        fish_id: {"zscore": zscore, "significant": significant}
        for fish_id, _, zscore, significant, _ in fish_blocks
    }
    for label in stimulus_labels:
        pooled[label]["zscore"] = np.vstack([
            fish_values[row.fish_id]["zscore"][label][int(row.raster_row)]
            for row in order.itertuples(index=False)
        ])
        pooled[label]["significant"] = np.vstack([
            fish_values[row.fish_id]["significant"][label][int(row.raster_row)]
            for row in order.itertuples(index=False)
        ])

    decision_matrix = np.column_stack([
        [
            int(fish_decisions[row.fish_id][label][int(row.raster_row)])
            for row in order.itertuples(index=False)
        ]
        for label in stimulus_labels
    ])

    time_relative = fish_blocks[0][4]
    validation = validation_table.loc[validation_table["hemifield"] == side].copy()
    summary_rows = []
    for match in validation.itertuples(index=False):
        bout_frames = _window_frames(
            time_relative[bout_label], match.time_after_motion_onset_s, position_window_offsets_s, bout_label
        )
        flicker_frames = _window_frames(
            time_relative[match.flicker_stimulus], 0.0, flicker_window_offsets_s, match.flicker_stimulus
        )
        for row in order.itertuples(index=False):
            display_row = int(row.display_row)
            bout_values = pooled[bout_label]["zscore"][display_row, bout_frames]
            flicker_values = pooled[match.flicker_stimulus]["zscore"][display_row, flicker_frames]
            summary_rows.append({
                "hemifield": side,
                "flicker_stimulus": match.flicker_stimulus,
                "nearest_bout_position_index": int(match.nearest_bout_position_index),
                "time_after_motion_onset_s": float(match.time_after_motion_onset_s),
                "fish_id": row.fish_id,
                "neuron_id": int(row.neuron_id),
                "group": row.group,
                "bout_mean_zscore": float(np.nanmean(bout_values)),
                "bout_significant_in_window": bool(np.any(pooled[bout_label]["significant"][display_row, bout_frames] > 0)),
                "flicker_mean_zscore": float(np.nanmean(flicker_values)),
                "flicker_significant_in_window": bool(np.any(pooled[match.flicker_stimulus]["significant"][display_row, flicker_frames] > 0)),
            })

    duration_by_label = {
        label: dict(all_fish_data[fish_ids[0]]["stimuli_durations"][label]) for label in stimulus_labels
    }
    panel_timing = {
        label: {
            "static_onset_relative_s": -float(duration_by_label[label].get("static_before_sec", motion_onset_s)),
            "analysis_end_relative_s": float(duration_by_label[label].get("motion_sec", 0.0)) + 2.0 * float(tau_s),
        }
        for label in stimulus_labels
    }
    bout_active_count = int(order["bout_active"].sum())
    diagnostic = {
        "trace_matrix": np.hstack([pooled[label]["significant"] for label in stimulus_labels]),
        "decision_matrix": decision_matrix,
        "stim_labels": stimulus_labels,
        "trace_block_widths": [pooled[label]["significant"].shape[1] for label in stimulus_labels],
        "trace_block_timepoints": [pooled[label]["significant"].shape[1] for label in stimulus_labels],
        "trace_block_reps": [1 for _ in stimulus_labels],
        "combine_mode": "mean",
    }
    return {
        "side": side,
        "bout_stimulus": bout_label,
        "flicker_stimuli": flicker_labels,
        "stimulus_labels": stimulus_labels,
        "order_table": order,
        "n_bout_active": bout_active_count,
        "pooled_traces": pooled,
        "time_relative_s": time_relative,
        "panel_timing": panel_timing,
        "stimuli_durations": duration_by_label,
        "position_matches": validation,
        "summary_table": pd.DataFrame(summary_rows),
        "cell06_style_diagnostic": diagnostic,
    }


def _active_column(active_matrix, stimulus):
    if stimulus in active_matrix.columns:
        return active_matrix[stimulus].to_numpy(dtype=bool)
    if str(stimulus) in active_matrix.columns:
        return active_matrix[str(stimulus)].to_numpy(dtype=bool)
    raise KeyError(f"Stimulus {stimulus!r} is missing from the active-neuron matrix.")


def _stimulus_motion_onset(stimuli_durations, stimulus, fallback):
    duration = stimuli_durations.get(stimulus, {})
    return float(duration.get("static_before_sec", fallback))


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


def combine_reps_one_stim(arr, mode="concat"):
    """
    Combine repetitions for one stimulus.

    Parameters
    ----------
    arr : np.ndarray
        Trial-aligned traces with shape (n_neurons, n_time, n_reps).
    mode : {"concat", "mean"}
        If "concat", repetitions are laid out along time. If "mean",
        repetitions are averaged.

    Returns
    -------
    np.ndarray
        Shape (n_neurons, n_time * n_reps) for "concat", or
        (n_neurons, n_time) for "mean".
    """
    assert arr.ndim == 3, "Expected 3D array (neurons, time, reps)"
    n_neurons, n_time, n_reps = arr.shape

    if mode == "concat":
        arr_swapped = np.swapaxes(arr, 1, 2)
        out = arr_swapped.reshape(n_neurons, n_reps * n_time)
        return out

    elif mode == "mean":
        out = np.nanmean(arr, axis=2)
        return out

    else:
        raise ValueError(f"Unknown mode '{mode}', use 'concat' or 'mean'")


def build_matrix_for_fish(
    trial_aligned_traces,
    stim_order,
    kept_neuron_indices=None,
    combine_mode="concat",
):
    """
    Build one fish matrix by concatenating stimulus blocks along time.

    Parameters
    ----------
    trial_aligned_traces : dict
        Mapping stim_id -> array with shape (n_neurons, n_time, n_reps).
        Integer stimulus keys are tried first, then string keys.
    stim_order : list
        Stimulus IDs in the order to concatenate.
    kept_neuron_indices : array-like or None
        Optional neuron indices to keep before combining repetitions.
    combine_mode : {"concat", "mean"}
        Passed to `combine_reps_one_stim`.

    Returns
    -------
    np.ndarray
        Shape (n_selected_neurons, sum_time_across_stimuli).
    """
    blocks = []

    for stim in stim_order:
        if stim in trial_aligned_traces:
            arr = trial_aligned_traces[stim]
        else:
            arr = trial_aligned_traces[str(stim)]

        if kept_neuron_indices is not None:
            arr = arr[kept_neuron_indices, :, :]

        arr_block = combine_reps_one_stim(arr, mode=combine_mode)
        blocks.append(arr_block)

    fish_mat = np.concatenate(blocks, axis=1)
    return fish_mat


def _stim_key(trial_aligned_traces, stim):
    if stim in trial_aligned_traces:
        return stim
    str_stim = str(stim)
    if str_stim in trial_aligned_traces:
        return str_stim
    raise KeyError(stim)


def describe_matrix_widths(
    all_fish_data,
    stim_order,
    fish_ids=None,
    combine_mode="concat",
    trace_type="dfof",
):
    """Return per-fish matrix widths and per-stimulus source shapes."""
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())

    rows = []
    for fid in fish_ids:
        if trace_type == "raster":
            trial_key = "trial_aligned_traces_raster"
        elif trace_type == "zscore":
            trial_key = "trial_aligned_traces_z_core"
        elif trace_type == "dfof":
            trial_key = "trial_aligned_traces"
        else:
            raise ValueError(
                f"Unknown trace_type={trace_type!r}. "
                "Use 'dfof', 'raster', or 'zscore'."
            )

        trial_aligned_traces = all_fish_data[fid][trial_key]
        total_width = 0
        stim_shapes = {}
        for stim in stim_order:
            key = _stim_key(trial_aligned_traces, stim)
            arr = trial_aligned_traces[key]
            if arr.ndim != 3:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} has shape {arr.shape}; "
                    "expected (neurons, time, reps)."
                )
            _, n_time, n_reps = arr.shape
            width = n_time * n_reps if combine_mode == "concat" else n_time
            total_width += width
            stim_shapes[stim] = arr.shape

        rows.append(
            {
                "fish_id": fid,
                "matrix_width": total_width,
                "stim_shapes": stim_shapes,
            }
        )

    return rows


def build_matrix_all_fish(
    all_fish_data,
    stim_order,
    fish_ids=None,
    combine_mode="concat",
    trace_type="dfof",
):
    """
    Stack per-fish matrices into one all-fish matrix.

    Parameters
    ----------
    all_fish_data : dict
        Per-fish data container from the several-fish notebooks.
    stim_order : list
        Stimulus IDs in the order to concatenate.
    fish_ids : list or None
        Fish IDs to include. If None, uses all keys in `all_fish_data`.
    combine_mode : {"concat", "mean"}
        Passed to `build_matrix_for_fish`.
    trace_type : {"dfof", "raster", "zscore"}
        Selects the per-fish trial-aligned trace key. "dfof" and "zscore"
        apply `kept_neuron_indices`; "raster" uses the raster traces as-is.

    Returns
    -------
    np.ndarray
        Shape (sum_selected_neurons_all_fish, sum_time_across_stimuli).
    """
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())

    mats = []
    fish_widths = []
    for fid in fish_ids:
        if trace_type == "raster":
            trial_key = "trial_aligned_traces_raster"
            kept_idx = None

        elif trace_type == "zscore":
            trial_key = "trial_aligned_traces_z_core"
            kept_idx = all_fish_data[fid]["kept_neuron_indices"]

        elif trace_type == "dfof":
            trial_key = "trial_aligned_traces"
            kept_idx = all_fish_data[fid]["kept_neuron_indices"]

        else:
            raise ValueError(
                f"Unknown trace_type={trace_type!r}. "
                "Use 'dfof', 'raster', or 'zscore'."
            )

        trial_aligned_traces = all_fish_data[fid][trial_key]

        M_fish = build_matrix_for_fish(
            trial_aligned_traces=trial_aligned_traces,
            stim_order=stim_order,
            kept_neuron_indices=kept_idx,
            combine_mode=combine_mode,
        )
        mats.append(M_fish)
        fish_widths.append((fid, M_fish.shape[1]))

    widths = {width for _, width in fish_widths}
    if len(widths) != 1:
        details = describe_matrix_widths(
            all_fish_data=all_fish_data,
            stim_order=stim_order,
            fish_ids=fish_ids,
            combine_mode=combine_mode,
            trace_type=trace_type,
        )
        detail_lines = [
            f"{row['fish_id']}: width={row['matrix_width']}, "
            f"stim_shapes={row['stim_shapes']}"
            for row in details
        ]
        raise ValueError(
            "Cannot stack fish matrices because their time widths differ. "
            "All fish must have the same aligned window and repetition count "
            f"for stim_order={stim_order!r}, combine_mode={combine_mode!r}, "
            f"trace_type={trace_type!r}.\n"
            + "\n".join(detail_lines)
        )

    M_all = np.vstack(mats)

    return M_all


def build_zscore_response_matrix_for_fish(
    trial_aligned_traces_z_core,
    selected_stimuli,
    stimuli_durations,
    stimuli_id_map=None,
    kept_neuron_indices=None,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """
    Build one neuron-by-stimulus z-score AUC response matrix for one fish.

    Z-score traces are subset with ``kept_neuron_indices`` before response
    extraction so matrix rows match downstream filtered-neuron rasters and
    active-neuron matrices.
    """
    selection = resolve_selected_stimuli(
        selected_stimuli,
        stimuli_id_map=stimuli_id_map,
        available_stimuli=trial_aligned_traces_z_core.keys(),
    )
    stimulus_ids = selection["stimulus_ids"]
    stimulus_labels = selection["stimulus_labels"]

    kept_idx = None
    if kept_neuron_indices is not None:
        kept_idx = np.asarray(kept_neuron_indices, dtype=int).ravel()
        if kept_idx.size == 0:
            raise ValueError("kept_neuron_indices is empty.")

    response_columns = {}
    n_neurons_expected = None

    for stim_id, stim_label in zip(stimulus_ids, stimulus_labels):
        key = _stim_key(trial_aligned_traces_z_core, stim_id)
        arr = np.asarray(trial_aligned_traces_z_core[key], dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"Stimulus {stim_id!r} has shape {arr.shape}; "
                "expected (n_neurons, n_time, n_reps)."
            )

        if kept_idx is not None:
            if np.any(kept_idx < 0) or np.any(kept_idx >= arr.shape[0]):
                raise IndexError(
                    f"kept_neuron_indices contains values outside stimulus "
                    f"{stim_id!r} neuron axis with n_neurons={arr.shape[0]}."
                )
            arr = arr[kept_idx, :, :]

        if n_neurons_expected is None:
            n_neurons_expected = arr.shape[0]
        elif arr.shape[0] != n_neurons_expected:
            raise ValueError(
                f"Stimulus {stim_id!r} has {arr.shape[0]} selected neurons; "
                f"expected {n_neurons_expected} to preserve row alignment."
            )

        window = compute_response_window_frames(
            n_time=arr.shape[1],
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            motion_onset_s=motion_onset_s,
            stimulus=stim_id,
            stimuli_durations=stimuli_durations,
            stimuli_id_map=stimuli_id_map,
            tau_s=tau_s,
            motion_duration_key=motion_duration_key,
        )
        response_columns[stim_label] = compute_zscore_response_auc(
            arr,
            frame_indices=window["frame_indices"],
            time_s=window["time_s"],
            fps_2p=fps_2p,
        )

    response_matrix = pd.DataFrame(response_columns, columns=stimulus_labels)
    response_matrix.index.name = "neuron_id"
    return response_matrix


def build_zscore_response_matrices_all_fish(
    all_fish_data,
    selected_stimuli,
    fish_ids=None,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """
    Build per-fish and pooled z-score AUC response matrices.

    Returns per-fish matrices, a pooled matrix, and row metadata describing the
    pooled row order. Identity columns remain metadata here; Slice 3 promotes
    them into the neuron summary table.
    """
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())
    fish_ids = list(dict.fromkeys(fish_ids))

    response_matrices = {}
    pooled_matrices = []
    metadata_rows = []
    selected_stimulus_ids = None
    selected_stimulus_labels = None
    global_neuron_id = 0

    for fid in fish_ids:
        fish = all_fish_data[fid]
        response_matrix = build_zscore_response_matrix_for_fish(
            trial_aligned_traces_z_core=fish["trial_aligned_traces_z_core"],
            selected_stimuli=selected_stimuli,
            stimuli_durations=fish["stimuli_durations"],
            stimuli_id_map=fish.get("stimuli_id_map"),
            kept_neuron_indices=fish.get("kept_neuron_indices"),
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            motion_onset_s=motion_onset_s,
            tau_s=tau_s,
            motion_duration_key=motion_duration_key,
        )

        selection = resolve_selected_stimuli(
            selected_stimuli,
            stimuli_id_map=fish.get("stimuli_id_map"),
            available_stimuli=fish["trial_aligned_traces_z_core"].keys(),
        )
        if selected_stimulus_ids is None:
            selected_stimulus_ids = selection["stimulus_ids"]
            selected_stimulus_labels = selection["stimulus_labels"]
        elif response_matrix.columns.tolist() != selected_stimulus_labels:
            raise ValueError(
                f"Fish {fid!r} resolved selected stimulus labels "
                f"{response_matrix.columns.tolist()} but expected "
                f"{selected_stimulus_labels}."
            )

        response_matrices[fid] = response_matrix
        pooled_matrices.append(response_matrix)

        kept_idx = fish.get("kept_neuron_indices")
        if kept_idx is None:
            source_neuron_ids = np.arange(response_matrix.shape[0])
        else:
            source_neuron_ids = np.asarray(kept_idx, dtype=int).ravel()

        if source_neuron_ids.shape[0] != response_matrix.shape[0]:
            raise ValueError(
                f"Fish {fid!r} kept_neuron_indices length "
                f"{source_neuron_ids.shape[0]} does not match response rows "
                f"{response_matrix.shape[0]}."
            )

        for neuron_id, source_neuron_id in enumerate(source_neuron_ids):
            metadata_rows.append(
                {
                    "fish_id": fid,
                    "neuron_id": neuron_id,
                    "source_neuron_id": int(source_neuron_id),
                    "global_neuron_id": global_neuron_id,
                }
            )
            global_neuron_id += 1

    if pooled_matrices:
        pooled_response_matrix = pd.concat(
            pooled_matrices,
            axis=0,
            ignore_index=True,
        )
    else:
        pooled_response_matrix = pd.DataFrame(columns=selected_stimulus_labels or [])

    pooled_response_matrix.index.name = "global_neuron_id"
    row_metadata = pd.DataFrame(
        metadata_rows,
        columns=["fish_id", "neuron_id", "source_neuron_id", "global_neuron_id"],
    )

    return {
        "response_matrices": response_matrices,
        "pooled_response_matrix": pooled_response_matrix,
        "row_metadata": row_metadata,
        "selected_stimulus_ids": selected_stimulus_ids or [],
        "selected_stimulus_labels": selected_stimulus_labels or [],
    }


def _active_column_values(active_matrix, stim):
    if hasattr(active_matrix, "columns"):
        if stim in active_matrix.columns:
            key = stim
        elif str(stim) in active_matrix.columns:
            key = str(stim)
        else:
            raise KeyError(f"Stimulus {stim!r} not found in active matrix.")
        return np.asarray(active_matrix[key]).astype(int)

    arr = np.asarray(active_matrix)
    if arr.ndim != 2:
        raise ValueError("active_matrix must be 2D.")
    return arr[:, int(stim)].astype(int)


def build_neuron_stimulus_summary_table(
    response_matrices,
    active_matrices,
    row_metadata,
    selected_stimulus_ids,
    selected_stimulus_labels,
    analysis_label,
):
    """
    Join z-score responses, binary active decisions, and neuron identity rows.

    This is the Slice 3 table scaffold: one row per filtered neuron, identity
    columns first, then one raw response and one active-decision column per
    selected stimulus. Selectivity metrics and classes are intentionally added
    in later slices.
    """
    selected_stimulus_ids = list(selected_stimulus_ids)
    selected_stimulus_labels = list(selected_stimulus_labels)
    if len(selected_stimulus_ids) != len(selected_stimulus_labels):
        raise ValueError(
            "selected_stimulus_ids and selected_stimulus_labels must have "
            "the same length."
        )
    if not selected_stimulus_ids:
        raise ValueError("At least one selected stimulus is required.")

    required_metadata = [
        "fish_id",
        "neuron_id",
        "source_neuron_id",
        "global_neuron_id",
    ]
    row_metadata = row_metadata.copy()
    missing_metadata = [
        column for column in required_metadata if column not in row_metadata.columns
    ]
    if missing_metadata:
        raise ValueError(
            "row_metadata is missing required column(s): "
            + ", ".join(missing_metadata)
        )

    rows = []
    for fid, fish_metadata in row_metadata.groupby("fish_id", sort=False):
        if fid not in response_matrices:
            raise KeyError(f"Fish {fid!r} not found in response_matrices.")
        if fid not in active_matrices:
            raise KeyError(f"Fish {fid!r} not found in active_matrices.")

        response_matrix = response_matrices[fid]
        active_matrix = active_matrices[fid]
        if response_matrix.shape[0] != fish_metadata.shape[0]:
            raise ValueError(
                f"Fish {fid!r} response rows {response_matrix.shape[0]} do not "
                f"match metadata rows {fish_metadata.shape[0]}."
            )

        active_columns = {
            label: _active_column_values(active_matrix, stim_id)
            for stim_id, label in zip(selected_stimulus_ids, selected_stimulus_labels)
        }
        for label, values in active_columns.items():
            if values.shape[0] != fish_metadata.shape[0]:
                raise ValueError(
                    f"Fish {fid!r} active column {label!r} has {values.shape[0]} "
                    f"rows; expected {fish_metadata.shape[0]}."
                )

        for local_row, (_, meta_row) in enumerate(fish_metadata.iterrows()):
            row = {column: meta_row[column] for column in required_metadata}
            row["analysis_label"] = analysis_label
            row["selected_stimuli"] = list(selected_stimulus_labels)

            for stim_label in selected_stimulus_labels:
                if stim_label not in response_matrix.columns:
                    raise KeyError(
                        f"Response matrix for fish {fid!r} is missing column "
                        f"{stim_label!r}."
                    )
                row[f"response_{stim_label}"] = response_matrix.iloc[
                    local_row
                ][stim_label]
                row[f"active_{stim_label}"] = int(active_columns[stim_label][local_row])

            rows.append(row)

    columns = (
        required_metadata
        + ["analysis_label", "selected_stimuli"]
        + [f"response_{label}" for label in selected_stimulus_labels]
        + [f"active_{label}" for label in selected_stimulus_labels]
    )
    summary_table = pd.DataFrame(rows, columns=columns)
    if not summary_table.empty:
        summary_table = summary_table.sort_values("global_neuron_id").reset_index(drop=True)

    return summary_table


def add_selectivity_metrics_to_summary_table(summary_table, selected_stimulus_labels):
    """
    Add preference, selectivity, and binary breadth metrics to a summary table.

    Expects Slice 3 columns named ``response_<stimulus>`` and
    ``active_<stimulus>`` for every selected stimulus label.
    """
    selected_stimulus_labels = list(selected_stimulus_labels)
    if len(selected_stimulus_labels) < 2:
        raise ValueError("At least two selected stimuli are required.")

    response_columns = [f"response_{label}" for label in selected_stimulus_labels]
    active_columns = [f"active_{label}" for label in selected_stimulus_labels]
    missing_columns = [
        column
        for column in response_columns + active_columns
        if column not in summary_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "summary_table is missing required column(s): "
            + ", ".join(missing_columns)
        )

    metrics_rows = []
    for _, row in summary_table.iterrows():
        response_values = row[response_columns].to_numpy(dtype=float)
        metrics = compute_stimulus_selectivity_metrics(
            response_values,
            stimulus_labels=selected_stimulus_labels,
        )

        active_values = row[active_columns].to_numpy(dtype=int)
        n_active_stimuli = int(np.count_nonzero(active_values))
        metrics["n_active_stimuli"] = n_active_stimuli
        metrics["response_breadth"] = float(n_active_stimuli / len(selected_stimulus_labels))
        metrics_rows.append(metrics)

    metrics_table = pd.DataFrame(metrics_rows)
    result = summary_table.copy().reset_index(drop=True)

    metric_columns = [
        "preferred_stimulus",
        "max_response",
        "mean_response",
        "simple_selectivity",
        "selectivity_index",
        "lifetime_sparseness",
        "n_active_stimuli",
        "response_breadth",
    ]
    for column in metric_columns:
        result[column] = metrics_table[column]

    preferred_order = [
        "fish_id",
        "neuron_id",
        "source_neuron_id",
        "global_neuron_id",
        "preferred_stimulus",
        "max_response",
        "mean_response",
        "simple_selectivity",
        "selectivity_index",
        "lifetime_sparseness",
        "n_active_stimuli",
        "response_breadth",
        "analysis_label",
        "selected_stimuli",
    ]
    remaining_columns = [column for column in result.columns if column not in preferred_order]
    return result[[column for column in preferred_order if column in result.columns] + remaining_columns]


def classify_stimulus_specificity_summary_table(
    summary_table,
    selected_stimulus_labels,
    classification_thresholds=None,
    return_thresholds=False,
):
    """
    Add adaptive neuron classes to a metric-enriched summary table.

    Strong and weak response thresholds are derived from the current table's
    positive/clipped max responses using the configured quantiles.
    """
    defaults = {
        "high_lifetime_sparseness": 0.70,
        "intermediate_lifetime_sparseness": 0.40,
        "broad_breadth_threshold": 0.80,
        "strong_response_quantile": 0.75,
        "weak_response_quantile": 0.25,
    }
    thresholds = defaults.copy()
    if classification_thresholds is not None:
        thresholds.update(classification_thresholds)

    selected_stimulus_labels = list(selected_stimulus_labels)
    response_columns = [f"response_{label}" for label in selected_stimulus_labels]
    required_columns = response_columns + [
        "lifetime_sparseness",
        "n_active_stimuli",
        "response_breadth",
    ]
    missing_columns = [
        column for column in required_columns if column not in summary_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "summary_table is missing required column(s): "
            + ", ".join(missing_columns)
        )

    strong_quantile = float(thresholds["strong_response_quantile"])
    weak_quantile = float(thresholds["weak_response_quantile"])
    for name, value in (
        ("strong_response_quantile", strong_quantile),
        ("weak_response_quantile", weak_quantile),
    ):
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")

    result = summary_table.copy().reset_index(drop=True)
    response_values = result[response_columns].to_numpy(dtype=float)
    positive_responses = np.where(
        np.isfinite(response_values),
        np.maximum(response_values, 0.0),
        np.nan,
    )
    all_nan_rows = np.all(np.isnan(positive_responses), axis=1)
    max_positive_response = np.full(positive_responses.shape[0], np.nan, dtype=float)
    valid_rows = ~all_nan_rows
    if np.any(valid_rows):
        max_positive_response[valid_rows] = np.nanmax(
            positive_responses[valid_rows],
            axis=1,
        )

    finite_max_positive = max_positive_response[np.isfinite(max_positive_response)]
    if finite_max_positive.size == 0:
        strong_response_threshold = np.nan
        weak_response_threshold = np.nan
    else:
        strong_response_threshold = float(
            np.nanquantile(finite_max_positive, strong_quantile)
        )
        weak_response_threshold = float(
            np.nanquantile(finite_max_positive, weak_quantile)
        )

    neuron_classes = [
        classify_stimulus_specificity_neuron(
            lifetime_sparseness=row["lifetime_sparseness"],
            n_active_stimuli=int(row["n_active_stimuli"]),
            response_breadth=float(row["response_breadth"]),
            max_positive_response=max_positive_response[row_idx],
            high_lifetime_sparseness=float(thresholds["high_lifetime_sparseness"]),
            intermediate_lifetime_sparseness=float(
                thresholds["intermediate_lifetime_sparseness"]
            ),
            broad_breadth_threshold=float(thresholds["broad_breadth_threshold"]),
            strong_response_threshold=strong_response_threshold,
            weak_response_threshold=weak_response_threshold,
        )
        for row_idx, row in result.iterrows()
    ]

    result["neuron_class"] = neuron_classes

    applied_thresholds = thresholds.copy()
    applied_thresholds["strong_response_threshold"] = strong_response_threshold
    applied_thresholds["weak_response_threshold"] = weak_response_threshold

    preferred_order = [
        "fish_id",
        "neuron_id",
        "source_neuron_id",
        "global_neuron_id",
        "preferred_stimulus",
        "max_response",
        "mean_response",
        "simple_selectivity",
        "selectivity_index",
        "lifetime_sparseness",
        "n_active_stimuli",
        "response_breadth",
        "neuron_class",
        "analysis_label",
        "selected_stimuli",
    ]
    remaining_columns = [column for column in result.columns if column not in preferred_order]
    result = result[[column for column in preferred_order if column in result.columns] + remaining_columns]

    if return_thresholds:
        return result, applied_thresholds
    return result


def build_stimulus_specificity_neuron_order(
    summary_table,
    selected_stimulus_labels,
    diagnostic_row_metadata=None,
):
    """
    Build a pooled raster row order from stimulus-specificity metrics.

    Rows are sorted by selected-stimulus preference order, descending lifetime
    sparseness, then descending max response. If diagnostic row metadata is
    supplied, the returned indices are in diagnostic row coordinates.
    """
    selected_stimulus_labels = list(selected_stimulus_labels)
    required_columns = [
        "fish_id",
        "neuron_id",
        "preferred_stimulus",
        "lifetime_sparseness",
        "max_response",
    ]
    missing_columns = [
        column for column in required_columns if column not in summary_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "summary_table is missing required column(s): "
            + ", ".join(missing_columns)
        )

    order_source = summary_table.copy().reset_index(drop=True)
    order_source["_summary_row"] = np.arange(order_source.shape[0])

    if diagnostic_row_metadata is not None:
        diagnostic_row_metadata = diagnostic_row_metadata.copy().reset_index(drop=True)
        missing_metadata = [
            column
            for column in ("fish_id", "neuron_id")
            if column not in diagnostic_row_metadata.columns
        ]
        if missing_metadata:
            raise ValueError(
                "diagnostic_row_metadata is missing required column(s): "
                + ", ".join(missing_metadata)
            )

        order_source = diagnostic_row_metadata.reset_index(names="_diagnostic_row").merge(
            order_source,
            on=["fish_id", "neuron_id"],
            how="left",
            validate="one_to_one",
        )
        if order_source["preferred_stimulus"].isna().any():
            missing = order_source.loc[
                order_source["preferred_stimulus"].isna(),
                ["fish_id", "neuron_id"],
            ]
            raise ValueError(
                "Some diagnostic rows were not found in the summary table: "
                f"{missing.head().to_dict('records')}"
            )
        row_index_column = "_diagnostic_row"
    else:
        row_index_column = "_summary_row"

    preference_rank = {
        label: rank for rank, label in enumerate(selected_stimulus_labels)
    }
    fallback_rank = len(selected_stimulus_labels)
    order_source["_preference_rank"] = (
        order_source["preferred_stimulus"].map(preference_rank).fillna(fallback_rank)
    )
    order_source["_sparseness_sort"] = order_source["lifetime_sparseness"].fillna(-np.inf)
    order_source["_max_response_sort"] = order_source["max_response"].fillna(-np.inf)

    sorted_rows = order_source.sort_values(
        by=[
            "_preference_rank",
            "_sparseness_sort",
            "_max_response_sort",
            row_index_column,
        ],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    return sorted_rows[row_index_column].to_numpy(dtype=int)


def build_stimulus_vector_similarity(response_matrix, selected_stimuli=None):
    """
    Compute stimulus-vector similarity from a neuron-by-stimulus response matrix.

    Returns Pearson and cosine similarity matrices plus a compact pair table
    with the positional distance between selected stimuli.
    """
    response_matrix = pd.DataFrame(response_matrix).copy()
    if selected_stimuli is None:
        selected_stimuli = list(response_matrix.columns)
    selected_stimuli = list(selected_stimuli)
    if not selected_stimuli:
        raise ValueError("selected_stimuli must contain at least one stimulus.")

    missing = [
        stimulus for stimulus in selected_stimuli
        if stimulus not in response_matrix.columns
    ]
    if missing:
        raise KeyError(
            "selected_stimuli contains label(s) missing from response_matrix: "
            + ", ".join(map(str, missing))
        )

    selected_response_matrix = response_matrix.loc[:, selected_stimuli]
    pearson_similarity_matrix = selected_response_matrix.corr(method="pearson")
    cosine_similarity_matrix = pd.DataFrame(
        np.nan,
        index=selected_stimuli,
        columns=selected_stimuli,
        dtype=float,
    )

    for stimulus_a in selected_stimuli:
        vector_a = selected_response_matrix[stimulus_a].to_numpy(dtype=float)
        for stimulus_b in selected_stimuli:
            vector_b = selected_response_matrix[stimulus_b].to_numpy(dtype=float)
            finite_mask = np.isfinite(vector_a) & np.isfinite(vector_b)
            if not np.any(finite_mask):
                continue
            a = vector_a[finite_mask]
            b = vector_b[finite_mask]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator > 0:
                cosine_similarity_matrix.loc[stimulus_a, stimulus_b] = (
                    np.dot(a, b) / denominator
                )

    pair_rows = []
    for index_a, stimulus_a in enumerate(selected_stimuli):
        for index_b in range(index_a + 1, len(selected_stimuli)):
            stimulus_b = selected_stimuli[index_b]
            pair_rows.append(
                {
                    "stimulus_a": stimulus_a,
                    "stimulus_b": stimulus_b,
                    "segment_distance": abs(index_b - index_a),
                    "pearson_similarity": pearson_similarity_matrix.loc[
                        stimulus_a, stimulus_b
                    ],
                    "cosine_similarity": cosine_similarity_matrix.loc[
                        stimulus_a, stimulus_b
                    ],
                }
            )

    pair_similarity = pd.DataFrame(
        pair_rows,
        columns=[
            "stimulus_a",
            "stimulus_b",
            "segment_distance",
            "pearson_similarity",
            "cosine_similarity",
        ],
    )

    return {
        "selected_response_matrix": selected_response_matrix,
        "pearson_similarity_matrix": pearson_similarity_matrix,
        "cosine_similarity_matrix": cosine_similarity_matrix,
        "pair_similarity": pair_similarity,
    }


def resolve_segment_labels(segments, available_labels):
    """Resolve editable segment names against selected stimulus labels."""
    available_labels = list(available_labels)
    resolved = []
    for segment in segments:
        exact_matches = [label for label in available_labels if label == segment]
        if len(exact_matches) == 1:
            resolved.append(exact_matches[0])
            continue

        suffix_matches = [
            label for label in available_labels
            if str(label).endswith(str(segment))
        ]
        if len(suffix_matches) == 1:
            resolved.append(suffix_matches[0])
        elif len(suffix_matches) == 0:
            raise KeyError(
                f"Segment {segment!r} was not found in available labels "
                f"{available_labels!r}."
            )
        else:
            raise ValueError(
                f"Segment {segment!r} matched multiple labels: "
                f"{suffix_matches!r}. Use an exact label."
            )

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Resolved segment labels are not unique: {resolved!r}")
    return resolved


def _compute_segment_selectivity_index(segment_to_trials):
    means = {}
    for segment, values in segment_to_trials.items():
        values = np.asarray(values, dtype=float).ravel()
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.nan, np.nan, "missing_or_empty_trials"
        means[segment] = float(np.mean(finite_values))

    mean_values = np.asarray(list(means.values()), dtype=float)
    if not np.all(np.isfinite(mean_values)):
        return np.nan, np.nan, "invalid_segment_mean"

    preferred_idx = int(np.argmax(mean_values))
    preferred = list(means.keys())[preferred_idx]
    r_preferred = mean_values[preferred_idx]
    r_others = float(np.mean(np.delete(mean_values, preferred_idx)))
    denominator = r_preferred + r_others
    if r_preferred <= 0.0 or r_others < 0.0:
        return np.nan, preferred, "negative_or_nonpositive_response"
    if not np.isfinite(denominator) or denominator <= 0.0 or np.isclose(denominator, 0.0):
        return np.nan, preferred, "zero_or_invalid_denominator"

    return float((r_preferred - r_others) / denominator), preferred, None


def _shuffle_segment_trials(segment_to_trials, rng):
    segment_counts = {
        segment: np.asarray(values, dtype=float).ravel().size
        for segment, values in segment_to_trials.items()
    }
    pooled = np.concatenate(
        [np.asarray(values, dtype=float).ravel() for values in segment_to_trials.values()]
    )
    shuffled = rng.permutation(pooled)

    shuffled_segments = {}
    start = 0
    for segment, count in segment_counts.items():
        stop = start + count
        shuffled_segments[segment] = shuffled[start:stop]
        start = stop
    return shuffled_segments


def _run_segment_selectivity_permutations(neuron_segment_trials, n_permutations, rng):
    real_si, real_preferred, skip_reason = _compute_segment_selectivity_index(
        neuron_segment_trials
    )
    shuffled_si = np.full(int(n_permutations), np.nan, dtype=float)
    if skip_reason is not None:
        return real_si, real_preferred, shuffled_si, skip_reason

    for perm_idx in range(int(n_permutations)):
        shuffled_trials = _shuffle_segment_trials(neuron_segment_trials, rng)
        shuffled_si[perm_idx], _, _ = _compute_segment_selectivity_index(
            shuffled_trials
        )
    return real_si, real_preferred, shuffled_si, None


def build_segment_selectivity_permutation_summary(
    all_fish_data,
    fish_ids,
    selected_stimulus_ids,
    selected_stimulus_labels,
    segments_to_compare,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
    n_permutations=1000,
    random_seed=42,
    alpha_percentile=95,
):
    """
    Run segment-selectivity permutations for selected z-score AUC responses.

    The summary table has one row per pooled neuron and can be joined to the
    neuron summary table on ``global_neuron_id``.
    """
    fish_ids = list(fish_ids)
    selected_stimulus_ids = list(selected_stimulus_ids)
    selected_stimulus_labels = list(selected_stimulus_labels)
    resolved_segment_labels = resolve_segment_labels(
        segments_to_compare,
        selected_stimulus_labels,
    )
    segment_display_labels = list(segments_to_compare)
    segment_label_to_display = dict(
        zip(resolved_segment_labels, segment_display_labels)
    )
    selected_label_to_id = dict(zip(selected_stimulus_labels, selected_stimulus_ids))
    segment_label_to_id = {
        label: selected_label_to_id[label] for label in resolved_segment_labels
    }

    rng = np.random.default_rng(random_seed)
    trial_auc_by_fish = {}
    trial_count_rows = []
    warning_rows = []

    for fid in fish_ids:
        fish = all_fish_data[fid]
        trial_aligned_z = fish["trial_aligned_traces_z_core"]
        kept_idx = np.asarray(fish["kept_neuron_indices"], dtype=int).ravel()
        if kept_idx.size == 0:
            raise ValueError(f"Fish {fid!r} has empty kept_neuron_indices.")

        trial_auc_by_fish[fid] = {}
        n_kept_expected = None
        fish_trial_counts = []

        for resolved_label in resolved_segment_labels:
            display_label = segment_label_to_display[resolved_label]
            stim_id = segment_label_to_id[resolved_label]
            stim_key = _stim_key(trial_aligned_z, stim_id)
            arr = np.asarray(trial_aligned_z[stim_key], dtype=float)
            if arr.ndim != 3:
                raise ValueError(
                    f"Fish {fid!r}, segment {resolved_label!r} has shape "
                    f"{arr.shape}; expected (n_neurons, n_time, n_reps)."
                )
            if np.any(kept_idx < 0) or np.any(kept_idx >= arr.shape[0]):
                raise IndexError(
                    f"Fish {fid!r} kept_neuron_indices contains values outside "
                    f"segment {resolved_label!r} neuron axis with n_neurons={arr.shape[0]}."
                )

            arr_kept = arr[kept_idx, :, :]
            if n_kept_expected is None:
                n_kept_expected = arr_kept.shape[0]
            elif arr_kept.shape[0] != n_kept_expected:
                raise ValueError(
                    f"Fish {fid!r}, segment {resolved_label!r} has "
                    f"{arr_kept.shape[0]} kept neurons; expected {n_kept_expected}."
                )

            if arr_kept.shape[2] == 0:
                warning_rows.append(
                    {
                        "fish_id": fid,
                        "segment": display_label,
                        "warning": "empty_response_array",
                    }
                )

            window = compute_response_window_frames(
                n_time=arr_kept.shape[1],
                fps_2p=fps_2p,
                t_pre_s=t_pre_s,
                motion_onset_s=motion_onset_s,
                stimulus=stim_id,
                stimuli_durations=fish["stimuli_durations"],
                stimuli_id_map=fish["stimuli_id_map"],
                tau_s=tau_s,
                motion_duration_key=motion_duration_key,
            )
            trial_auc_by_fish[fid][display_label] = compute_trial_auc_by_neuron(
                arr_kept,
                frame_indices=window["frame_indices"],
                time_s=window["time_s"],
                fps_2p=fps_2p,
            )
            fish_trial_counts.append(arr_kept.shape[2])
            trial_count_rows.append(
                {
                    "fish_id": fid,
                    "segment": display_label,
                    "resolved_label": resolved_label,
                    "stimulus_id": stim_id,
                    "n_kept_neurons": arr_kept.shape[0],
                    "n_trials": arr_kept.shape[2],
                    "n_response_frames": window["n_frames"],
                    "start_frame": window["start_frame"],
                    "stop_frame": window["stop_frame"],
                }
            )

        if len(set(fish_trial_counts)) > 1:
            warning_rows.append(
                {
                    "fish_id": fid,
                    "segment": "all",
                    "warning": f"unequal_trial_counts:{fish_trial_counts}",
                }
            )

    summary_rows = []
    si_shuffle_rows = []
    global_row = 0
    skip_reason_counts = {}

    for fid in fish_ids:
        kept_idx = np.asarray(all_fish_data[fid]["kept_neuron_indices"], dtype=int).ravel()
        for neuron_id in range(kept_idx.size):
            neuron_segment_trials = {
                segment: trial_auc_by_fish[fid][segment][neuron_id, :]
                for segment in segment_display_labels
            }
            real_si, real_preferred, shuffled_si, skip_reason = (
                _run_segment_selectivity_permutations(
                    neuron_segment_trials,
                    n_permutations=n_permutations,
                    rng=rng,
                )
            )
            if skip_reason is not None:
                skip_reason_counts[skip_reason] = skip_reason_counts.get(skip_reason, 0) + 1

            finite_shuffle = shuffled_si[np.isfinite(shuffled_si)]
            if finite_shuffle.size == 0:
                threshold = np.nan
                is_significant = False
            else:
                threshold = float(np.nanpercentile(finite_shuffle, alpha_percentile))
                is_significant = bool(np.isfinite(real_si) and real_si > threshold)

            summary_rows.append(
                {
                    "fish_id": fid,
                    "neuron_id": neuron_id,
                    "kept_neuron_index": int(kept_idx[neuron_id]),
                    "global_neuron_id": global_row,
                    "segment_selectivity_index": real_si,
                    "preferred_segment": real_preferred,
                    "segment_shuffle_threshold": threshold,
                    "segment_selective": is_significant,
                    "segment_skip_reason": skip_reason,
                }
            )
            si_shuffle_rows.append(shuffled_si)
            global_row += 1

    summary_df = pd.DataFrame(summary_rows)
    si_shuffle = (
        np.vstack(si_shuffle_rows)
        if si_shuffle_rows
        else np.empty((0, int(n_permutations)))
    )

    return {
        "segments_requested": segment_display_labels,
        "segments_resolved": resolved_segment_labels,
        "segment_label_to_id": segment_label_to_id,
        "trial_auc_by_fish": trial_auc_by_fish,
        "trial_counts": pd.DataFrame(trial_count_rows),
        "warnings": pd.DataFrame(warning_rows),
        "summary_df": summary_df,
        "si_real": summary_df["segment_selectivity_index"].to_numpy(dtype=float),
        "si_shuffle": si_shuffle,
        "preferred_segment": summary_df["preferred_segment"].to_numpy(dtype=object),
        "significant_segment_selective": summary_df["segment_selective"].to_numpy(dtype=bool),
        "skip_reason_counts": skip_reason_counts,
        "alpha_percentile": alpha_percentile,
        "n_permutations": n_permutations,
        "random_seed": random_seed,
    }


def build_active_neuron_matrices_all_fish(
    all_fish_data,
    fish_ids=None,
    stim_order=None,
    fps_2p=2.0,
    t_pre_s=5.0,
    motion_onset_s=8.0,
    active_fraction_threshold=0.30,
    min_epoch_s=2.0,
    min_active_reps=2,
    expected_reps=4,
    require_expected_reps=True,
    tau_s=6.0,
    motion_duration_key="motion_sec",
    return_counts=False,
):
    """
    Build one binary neuron-by-stimulus active matrix per fish.

    Parameters controlling inclusion are exposed so notebooks can tune the
    active-neuron definition without changing the helper.
    """
    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())
    fish_ids = list(dict.fromkeys(fish_ids))

    if stim_order is None:
        if not fish_ids:
            return ({}, {}) if return_counts else {}
        stim_order = all_fish_data[fish_ids[0]]["stimuli_ids"]

    active_matrices = {}
    active_counts = {}

    for fid in fish_ids:
        fish = all_fish_data[fid]
        result = build_active_neuron_matrix_from_trial_raster(
            trial_aligned_traces_raster=fish["trial_aligned_traces_raster"],
            stimuli_durations=fish["stimuli_durations"],
            stimuli_id_map=fish.get("stimuli_id_map"),
            stim_order=stim_order,
            fps_2p=fps_2p,
            t_pre_s=t_pre_s,
            motion_onset_s=motion_onset_s,
            active_fraction_threshold=active_fraction_threshold,
            min_epoch_s=min_epoch_s,
            min_active_reps=min_active_reps,
            expected_reps=expected_reps,
            require_expected_reps=require_expected_reps,
            tau_s=tau_s,
            motion_duration_key=motion_duration_key,
            return_counts=return_counts,
        )

        if return_counts:
            active_matrices[fid], active_counts[fid] = result
        else:
            active_matrices[fid] = result

    if return_counts:
        return active_matrices, active_counts
    return active_matrices


def _active_columns_as_bool(active_matrix, stim_order, labels):
    if len(stim_order) != len(labels):
        raise ValueError("stim_order and labels must have the same length.")

    columns = {}
    if hasattr(active_matrix, "columns"):
        for stim, label in zip(stim_order, labels):
            if stim in active_matrix.columns:
                key = stim
            elif str(stim) in active_matrix.columns:
                key = str(stim)
            else:
                raise KeyError(f"Stimulus {stim!r} not found in active matrix.")
            columns[label] = np.asarray(active_matrix[key]).astype(bool)
    else:
        arr = np.asarray(active_matrix)
        if arr.ndim != 2:
            raise ValueError("active_matrix must be 2D.")
        for stim, label in zip(stim_order, labels):
            columns[label] = arr[:, int(stim)].astype(bool)

    return pd.DataFrame(columns)


def compute_active_neuron_jaccard_overlap(
    active_matrix,
    stim_order,
    labels=None,
    empty_union_value=np.nan,
):
    """
    Compute pairwise Jaccard overlap between active-neuron stimulus sets.

    Nonzero values in active_matrix are treated as active. The returned
    DataFrame is indexed and columned by labels, or by stim_order if labels is
    not provided.
    """
    if labels is None:
        labels = list(stim_order)
    labels = list(labels)

    active = _active_columns_as_bool(active_matrix, list(stim_order), labels)
    values = np.empty((len(labels), len(labels)), dtype=float)

    for i, label_i in enumerate(labels):
        set_i = active[label_i].to_numpy(dtype=bool)
        for j, label_j in enumerate(labels):
            set_j = active[label_j].to_numpy(dtype=bool)
            union = np.count_nonzero(set_i | set_j)
            if union == 0:
                values[i, j] = empty_union_value
            else:
                intersection = np.count_nonzero(set_i & set_j)
                values[i, j] = intersection / union

    return pd.DataFrame(values, index=labels, columns=labels)


def build_active_neuron_overlap_matrices_all_fish(
    active_matrices,
    side_stimuli,
    condition_labels,
    empty_union_value=np.nan,
):
    """
    Build left/right active-neuron overlap matrices from per-fish matrices.

    Returns both pooled all-neuron overlaps and the mean of per-fish overlaps:
    result["pooled"][side] and result["mean_per_fish"][side].
    """
    condition_labels = list(condition_labels)
    results = {"pooled": {}, "mean_per_fish": {}}

    for side, stim_order in side_stimuli.items():
        stim_order = list(stim_order)
        per_fish = [
            compute_active_neuron_jaccard_overlap(
                active_matrix=active_matrix,
                stim_order=stim_order,
                labels=condition_labels,
                empty_union_value=empty_union_value,
            )
            for active_matrix in active_matrices.values()
        ]

        if per_fish:
            stacked = np.stack([matrix.to_numpy(dtype=float) for matrix in per_fish])
            valid = np.isfinite(stacked)
            sums = np.where(valid, stacked, 0.0).sum(axis=0)
            counts = valid.sum(axis=0)
            mean_values = np.full(sums.shape, np.nan, dtype=float)
            np.divide(sums, counts, out=mean_values, where=counts > 0)
            results["mean_per_fish"][side] = pd.DataFrame(
                mean_values,
                index=condition_labels,
                columns=condition_labels,
            )

            pooled_input = pd.concat(
                [
                    _active_columns_as_bool(active_matrix, stim_order, condition_labels)
                    for active_matrix in active_matrices.values()
                ],
                axis=0,
                ignore_index=True,
            )
        else:
            results["mean_per_fish"][side] = pd.DataFrame(
                np.nan,
                index=condition_labels,
                columns=condition_labels,
            )
            pooled_input = pd.DataFrame(columns=condition_labels, dtype=bool)

        results["pooled"][side] = compute_active_neuron_jaccard_overlap(
            active_matrix=pooled_input,
            stim_order=condition_labels,
            labels=condition_labels,
            empty_union_value=empty_union_value,
        )

    return results


def _active_column_as_bool(active_matrix, stim):
    if hasattr(active_matrix, "columns"):
        if stim in active_matrix.columns:
            key = stim
        elif str(stim) in active_matrix.columns:
            key = str(stim)
        else:
            raise KeyError(f"Stimulus {stim!r} not found in active matrix.")
        return np.asarray(active_matrix[key]).astype(bool)

    arr = np.asarray(active_matrix)
    if arr.ndim != 2:
        raise ValueError("active_matrix must be 2D.")
    return arr[:, int(stim)].astype(bool)


def _trial_trace_key(trial_aligned_traces, stim):
    if stim in trial_aligned_traces:
        return stim
    str_stim = str(stim)
    if str_stim in trial_aligned_traces:
        return str_stim
    raise KeyError(f"Stimulus {stim!r} not found in trial_aligned_traces.")


def _stimulus_label(stim, stimuli_id_map=None):
    if stimuli_id_map:
        for name, stim_id in stimuli_id_map.items():
            if stim_id == stim or str(stim_id) == str(stim):
                return name
    return str(stim)


def build_pooled_active_trace_diagnostic(
    all_fish_data,
    active_matrices,
    stim_order,
    fish_ids=None,
    combine_mode="mean",
    trial_key="trial_aligned_traces_raster",
):
    """
    Build pooled trace and final-decision matrices for active-neuron diagnostics.

    The trace matrix pools neurons across fish and concatenates selected
    stimulus blocks along time. The decision matrix uses the same pooled row
    order and has one strict active-neuron decision column per stimulus.
    """
    if combine_mode not in {"mean", "concat"}:
        raise ValueError("combine_mode must be 'mean' or 'concat'.")

    if fish_ids is None:
        fish_ids = list(all_fish_data.keys())
    fish_ids = list(dict.fromkeys(fish_ids))
    stim_order = list(stim_order)

    trace_mats = []
    decision_mats = []
    row_metadata = []
    block_widths = None
    block_timepoints = None
    block_reps = None
    stim_labels = None

    for fid in fish_ids:
        if fid not in active_matrices:
            raise KeyError(f"Fish {fid!r} not found in active_matrices.")

        fish = all_fish_data[fid]
        trial_aligned_traces = fish[trial_key]
        active_matrix = active_matrices[fid]

        if stim_labels is None:
            stim_labels = [
                _stimulus_label(stim, fish.get("stimuli_id_map"))
                for stim in stim_order
            ]

        fish_blocks = []
        fish_decisions = []
        n_neurons_expected = None
        fish_widths = []
        fish_timepoints = []
        fish_reps = []

        for stim in stim_order:
            key = _trial_trace_key(trial_aligned_traces, stim)
            arr = np.asarray(trial_aligned_traces[key])
            if arr.ndim != 3:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} has shape {arr.shape}; "
                    "expected (neurons, time, reps)."
                )

            n_neurons = arr.shape[0]
            if n_neurons_expected is None:
                n_neurons_expected = n_neurons
            elif n_neurons != n_neurons_expected:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} has {n_neurons} neurons; "
                    f"expected {n_neurons_expected} to preserve row alignment."
                )

            block = combine_reps_one_stim(arr, mode=combine_mode)
            fish_blocks.append(block)
            fish_widths.append(block.shape[1])
            fish_timepoints.append(arr.shape[1])
            fish_reps.append(arr.shape[2])

            decision = _active_column_as_bool(active_matrix, stim)
            if decision.shape[0] != n_neurons:
                raise ValueError(
                    f"Fish {fid!r}, stimulus {stim!r} active decision length "
                    f"{decision.shape[0]} does not match trace neurons {n_neurons}."
                )
            fish_decisions.append(decision.astype(int))

        if block_widths is None:
            block_widths = fish_widths
            block_timepoints = fish_timepoints
            block_reps = fish_reps
        elif fish_widths != block_widths:
            raise ValueError(
                f"Fish {fid!r} block widths {fish_widths} do not match expected "
                f"{block_widths}. All fish need matching time/repetition widths."
            )
        elif fish_timepoints != block_timepoints or fish_reps != block_reps:
            raise ValueError(
                f"Fish {fid!r} time/repetition shapes do not match expected values."
            )

        trace_mats.append(np.concatenate(fish_blocks, axis=1))
        decision_mats.append(np.column_stack(fish_decisions))
        row_metadata.extend(
            {"fish_id": fid, "neuron_id": neuron_id}
            for neuron_id in range(n_neurons_expected or 0)
        )

    if trace_mats:
        trace_matrix = np.vstack(trace_mats)
        decision_matrix = np.vstack(decision_mats).astype(int)
    else:
        block_widths = [0 for _ in stim_order]
        block_timepoints = [0 for _ in stim_order]
        block_reps = [0 for _ in stim_order]
        stim_labels = [str(stim) for stim in stim_order]
        trace_matrix = np.empty((0, 0), dtype=float)
        decision_matrix = np.empty((0, len(stim_order)), dtype=int)

    return {
        "trace_matrix": trace_matrix,
        "decision_matrix": decision_matrix,
        "row_metadata": pd.DataFrame(row_metadata),
        "stim_order": stim_order,
        "stim_labels": stim_labels,
        "trace_block_widths": block_widths,
        "trace_block_timepoints": block_timepoints,
        "trace_block_reps": block_reps,
        "combine_mode": combine_mode,
    }
