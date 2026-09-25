"""Trajectory-based bout/flicker position matching and pooled response ordering."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.active_neuron_analysis import build_active_neuron_matrices_all_fish
from src.trial_alignment import resolve_selected_stimuli


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
