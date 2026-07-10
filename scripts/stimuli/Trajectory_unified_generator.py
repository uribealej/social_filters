import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG_FILENAME = "trajectory_exp_8_left.json"

DEFAULT_EXPERIMENT_CONFIG = {
    "output_path": "scripts/stimuli/unified_example_output",
    "angle_ranges": [[30, 160]],
    "radius_cm": 1.8,
    "speed_cm_sec": 0.497,
    "framerate": 60,
    "update_interval_ms": 600,
    "static_period_sec": 8,
    "flicker_interval_ms": 300,
    "dot_size_on": 0.2,
    "rotation_angle_deg": 45,
    "pause_before_sec": 12.5,
    "pause_after_sec": 12.5,
}


def _frames_from_ms(ms, framerate):
    return max(1, int(round((float(ms) / 1000.0) * framerate)))


def _frames_from_sec(sec, framerate):
    return max(0, int(round(float(sec) * framerate)))


def _resolve_path(path_value, config_path):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def build_base_position_table(experiment_config):
    angle_range = experiment_config["angle_ranges"][0]
    radius = float(experiment_config["radius_cm"])
    speed = float(experiment_config["speed_cm_sec"])
    update_interval_ms = float(experiment_config["update_interval_ms"])
    rotation_angle = float(experiment_config["rotation_angle_deg"])

    start_angle, end_angle = np.deg2rad(angle_range)
    arc_length = radius * abs(end_angle - start_angle)
    total_time = round(arc_length / speed, 4)
    step_size = update_interval_ms / 1000.0
    frame_times = np.round(np.arange(0, total_time + step_size, step_size), 4)

    sign = np.sign(end_angle - start_angle) if end_angle != start_angle else 1.0
    angles = start_angle + sign * (speed / radius) * frame_times
    angles_deg = np.rad2deg(angles)

    x_values = radius * np.sin(angles)
    y_values = radius * np.cos(angles)
    theta = np.deg2rad(rotation_angle)
    x_rotated = np.round(x_values * np.cos(theta) - y_values * np.sin(theta), 3)
    y_rotated = np.round(x_values * np.sin(theta) + y_values * np.cos(theta), 3)

    table = pd.DataFrame(
        {
            "position_id": np.arange(1, len(angles_deg) + 1, dtype=int),
            "angle_deg": np.round(angles_deg, 6),
            "x": x_rotated,
            "y": y_rotated,
        }
    )
    return table, total_time


def base_control_steps(base_table):
    return len(base_table)


def base_control_duration_sec(base_table, experiment_config):
    framerate = float(experiment_config["framerate"])
    hold_frames = _frames_from_ms(experiment_config["update_interval_ms"], framerate)
    return (base_control_steps(base_table) * hold_frames) / framerate


def parse_positions(value, base_table, field_name="positions"):
    max_position = int(base_table["position_id"].max())
    if value == "all":
        positions = list(range(1, max_position + 1))
    elif isinstance(value, str):
        text = value.strip()
        if "," in text:
            positions = [int(part.strip()) for part in text.split(",") if part.strip()]
        elif "-" in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            positions = list(range(start, end + step, step))
        else:
            positions = [int(text)]
    elif isinstance(value, int):
        positions = [int(value)]
    elif isinstance(value, list):
        positions = [int(item) for item in value]
    else:
        raise ValueError(
            f"{field_name} must be 'all', a range string like '1-5', "
            "a comma string like '1,3,5', an int, or a list of ints."
        )

    invalid = [pos for pos in positions if pos < 1 or pos > max_position]
    if invalid:
        raise ValueError(f"{field_name} includes invalid position IDs {invalid}; valid range is 1-{max_position}.")
    if not positions:
        raise ValueError(f"{field_name} must select at least one position.")
    return positions


def get_repeats(params):
    repeat_keys = [key for key in ("repeats", "repetitions", "n_repetitions") if key in params]
    if len(repeat_keys) > 1:
        raise ValueError(f"Use only one repeat key per stimulus, not {repeat_keys}.")
    if not repeat_keys:
        return None
    return int(params[repeat_keys[0]])


def get_single_flicker_position(params, base_table):
    if "position" in params and "positions" in params:
        raise ValueError("single_flicker accepts either 'position' or 'positions', not both.")
    if "position" in params:
        positions = parse_positions(params["position"], base_table, field_name="position")
    elif "positions" in params:
        positions = parse_positions(params["positions"], base_table, field_name="positions")
    else:
        raise ValueError("single_flicker requires 'position' or 'positions' with exactly one position ID.")
    if len(positions) != 1:
        raise ValueError(f"single_flicker requires exactly one position, but got {positions}.")
    return positions[0]


def position_lookup(base_table):
    return {
        int(row.position_id): (float(row.x), float(row.y))
        for row in base_table.itertuples(index=False)
    }


def flicker_radius_for_frame(frame_idx, on_frames, off_frames, dot_size_on):
    cycle = on_frames + off_frames
    if cycle <= 0:
        return dot_size_on
    return dot_size_on if (frame_idx % cycle) < on_frames else 0.0


def append_static(rows, first_position, lookup, experiment_config):
    static_frames = _frames_from_sec(experiment_config["static_period_sec"], experiment_config["framerate"])
    dot_size_on = float(experiment_config["dot_size_on"])
    x, y = lookup[first_position]
    for _ in range(static_frames):
        rows.append({"x": x, "y": y, "radius": dot_size_on})


def build_single_dot_sequence(sequence, base_table, experiment_config, flickering=False, params=None):
    if not sequence:
        raise ValueError("Cannot build a trajectory from an empty position sequence.")

    params = params or {}
    lookup = position_lookup(base_table)
    framerate = float(experiment_config["framerate"])
    hold_frames = _frames_from_ms(experiment_config["update_interval_ms"], framerate)
    dot_size_on = float(experiment_config["dot_size_on"])
    on_frames = _frames_from_ms(params.get("flicker_on_ms", experiment_config["flicker_interval_ms"]), framerate)
    off_frames = _frames_from_ms(params.get("flicker_off_ms", experiment_config["flicker_interval_ms"]), framerate)

    rows = []
    append_static(rows, sequence[0], lookup, experiment_config)

    for position in sequence:
        x, y = lookup[position]
        for frame_idx in range(hold_frames):
            radius = flicker_radius_for_frame(frame_idx, on_frames, off_frames, dot_size_on) if flickering else dot_size_on
            rows.append({"x": x, "y": y, "radius": radius})

    return pd.DataFrame(rows)


def build_fixed_flicker(position, base_table, experiment_config, params):
    lookup = position_lookup(base_table)
    framerate = float(experiment_config["framerate"])
    dot_size_on = float(experiment_config["dot_size_on"])
    duration_sec = float(params["duration_sec"])
    on_frames = _frames_from_ms(params.get("flicker_on_ms", experiment_config["flicker_interval_ms"]), framerate)
    off_frames = _frames_from_ms(params.get("flicker_off_ms", experiment_config["flicker_interval_ms"]), framerate)
    flicker_frames = _frames_from_sec(duration_sec, framerate)

    rows = []
    append_static(rows, position, lookup, experiment_config)
    x, y = lookup[position]
    for frame_idx in range(flicker_frames):
        rows.append(
            {
                "x": x,
                "y": y,
                "radius": flicker_radius_for_frame(frame_idx, on_frames, off_frames, dot_size_on),
            }
        )
    return pd.DataFrame(rows)


def build_multi_flicker(positions, base_table, experiment_config, params):
    lookup = position_lookup(base_table)
    framerate = float(experiment_config["framerate"])
    dot_size_on = float(experiment_config["dot_size_on"])
    duration_sec = float(params["duration_sec"])
    on_frames = _frames_from_ms(params.get("flicker_on_ms", experiment_config["flicker_interval_ms"]), framerate)
    off_frames = _frames_from_ms(params.get("flicker_off_ms", experiment_config["flicker_interval_ms"]), framerate)
    static_frames = _frames_from_sec(experiment_config["static_period_sec"], framerate)
    flicker_frames = _frames_from_sec(duration_sec, framerate)

    rows = []
    for frame_idx in range(static_frames + flicker_frames):
        flicker_idx = max(0, frame_idx - static_frames)
        is_static = frame_idx < static_frames
        radius = dot_size_on if is_static else flicker_radius_for_frame(flicker_idx, on_frames, off_frames, dot_size_on)
        row = {}
        for dot_idx, position in enumerate(positions):
            x, y = lookup[position]
            row[f"dot{dot_idx}_x"] = x
            row[f"dot{dot_idx}_y"] = y
            row[f"dot{dot_idx}_radius"] = radius
        rows.append(row)
    return pd.DataFrame(rows)


def build_rocking_sequence(positions, params, experiment_config, default_steps):
    if len(positions) != 2:
        raise ValueError("rocking mode requires exactly two positions.")
    repeats = get_repeats(params)
    has_repeats = repeats is not None
    has_duration = "duration_sec" in params
    if has_repeats and has_duration:
        raise ValueError("rocking mode accepts repeats/repetitions or duration_sec, not both.")
    if has_duration:
        framerate = float(experiment_config["framerate"])
        hold_frames = _frames_from_ms(experiment_config["update_interval_ms"], framerate)
        target_frames = _frames_from_sec(params["duration_sec"], framerate)
        steps = max(1, int(np.ceil(target_frames / hold_frames)))
    elif has_repeats:
        steps = repeats * 2
    else:
        steps = default_steps
    return [positions[idx % 2] for idx in range(steps)]


def sequence_is_valid(sequence, max_jump, max_same_position_consecutive, previous_tail=None):
    combined = ([] if previous_tail is None else list(previous_tail)) + list(sequence)
    if not combined:
        return True
    for idx in range(1, len(combined)):
        if abs(combined[idx] - combined[idx - 1]) > max_jump:
            return False

    run_value = combined[0]
    run_length = 1
    for value in combined[1:]:
        if value == run_value:
            run_length += 1
            if run_length > max_same_position_consecutive:
                return False
        else:
            run_value = value
            run_length = 1
    return True


def make_balanced_pool(positions):
    return list(positions)


def make_balanced_pool_for_steps(positions, steps):
    base_count = steps // len(positions)
    extra_count = steps % len(positions)
    pool = []
    for position in positions:
        pool.extend([position] * base_count)
    pool.extend(positions[:extra_count])
    return pool


def generate_random_target_sequence(
    positions,
    rng,
    target_steps,
    max_jump,
    max_same_position_consecutive,
    retry_limit,
    first_position=None,
):
    if first_position is None:
        first_position = positions[0]
    pool = make_balanced_pool_for_steps(positions, target_steps)
    if first_position not in pool:
        raise ValueError(
            f"random_motion must include position {first_position} because random stimuli "
            "begin at the first selected position."
        )

    for _ in range(retry_limit):
        remaining = list(pool)
        remaining.remove(first_position)
        candidate = [int(first_position)] + [int(pos) for pos in rng.permutation(remaining)]
        if sequence_is_valid(candidate, max_jump, max_same_position_consecutive):
            return candidate

    raise ValueError(
        "Could not generate a valid default-length random sequence. "
        "Try increasing max_jump, reducing selected positions, adding repeats, or changing the seed."
    )


def generate_random_repeat(
    positions,
    rng,
    max_jump,
    max_same_position_consecutive,
    previous_tail,
    retry_limit,
    required_first_position=None,
):
    pool = make_balanced_pool(positions)
    if required_first_position is not None and required_first_position not in pool:
        raise ValueError(
            f"random_motion requires position {required_first_position} to be selected "
            "because random stimuli start at position 1."
        )
    for _ in range(retry_limit):
        if required_first_position is None:
            candidate = [int(pos) for pos in rng.permutation(pool)]
        else:
            remaining = [pos for pos in pool if pos != required_first_position]
            candidate = [int(required_first_position)] + [int(pos) for pos in rng.permutation(remaining)]
        if sequence_is_valid(candidate, max_jump, max_same_position_consecutive, previous_tail):
            return candidate
    raise ValueError(
        "Could not generate a valid balanced random sequence. "
        "Try increasing max_jump, reducing selected positions, or changing the seed."
    )


def build_random_sequence(positions, params, default_steps):
    explicit_repeats = get_repeats(params)
    has_repeats = explicit_repeats is not None
    repeats = explicit_repeats if has_repeats else None
    max_jump = int(params.get("max_jump", 2))
    max_same = int(params.get("max_same_position_consecutive", 2))
    balanced = bool(params.get("balanced_visits", True))
    randomize_each_repeat = bool(params.get("randomize_each_repeat", True))
    retry_limit = int(params.get("retry_limit", 10000))
    rng = np.random.default_rng(params.get("seed"))

    full_sequence = []
    previous_tail = []
    first_position = positions[0]
    if first_position not in positions:
        raise ValueError(
            f"random_motion must include position {first_position} because random stimuli "
            "begin at the first selected position."
        )

    if balanced and not has_repeats:
        return generate_random_target_sequence(
            positions,
            rng,
            default_steps,
            max_jump,
            max_same,
            retry_limit,
            first_position=first_position,
        )

    if not balanced:
        target_steps = repeats * len(positions) if has_repeats else default_steps
        for step_idx in range(target_steps):
            if step_idx == 0:
                if not sequence_is_valid([first_position], max_jump, max_same, previous_tail):
                    raise ValueError("random_motion cannot start at position 1 with the requested constraints.")
                full_sequence.append(first_position)
                previous_tail = full_sequence[-max_same:]
                continue

            allowed = [
                pos for pos in positions
                if sequence_is_valid([pos], max_jump, max_same, previous_tail)
            ]
            if not allowed:
                raise ValueError("Could not generate a valid random sequence with the requested constraints.")
            choice = int(rng.choice(allowed))
            full_sequence.append(choice)
            previous_tail = full_sequence[-max_same:]
        return full_sequence

    cached_repeat = None
    for repeat_idx in range(repeats):
        required_first = first_position if repeat_idx == 0 else None
        if randomize_each_repeat or cached_repeat is None:
            repeat_sequence = generate_random_repeat(
                positions,
                rng,
                max_jump,
                max_same,
                previous_tail,
                retry_limit,
                required_first_position=required_first,
            )
            if not randomize_each_repeat:
                cached_repeat = list(repeat_sequence)
        else:
            repeat_sequence = list(cached_repeat)
            if not sequence_is_valid(repeat_sequence, max_jump, max_same, previous_tail):
                repeat_sequence = generate_random_repeat(
                    positions,
                    rng,
                    max_jump,
                    max_same,
                    previous_tail,
                    retry_limit,
                    required_first_position=required_first,
                )

        full_sequence.extend(repeat_sequence)
        previous_tail = full_sequence[-max_same:]

    return full_sequence


def add_default_duration(params, base_motion_duration):
    if "duration_sec" not in params:
        params = dict(params)
        params["duration_sec"] = base_motion_duration
    return params


def generate_stimulus(stimulus_key, params, base_table, experiment_config, base_motion_duration):
    mode = params["type"]
    default_steps = base_control_steps(base_table)
    if mode == "ordered_motion":
        positions = parse_positions(params.get("positions", "all"), base_table)
        repeats = get_repeats(params)
        if repeats is not None:
            sequence = positions * repeats
        else:
            sequence = [positions[idx % len(positions)] for idx in range(default_steps)]
        return build_single_dot_sequence(
            sequence,
            base_table,
            experiment_config,
            flickering=bool(params.get("flickering", False)),
            params=params,
        )

    if mode == "random_motion":
        positions = parse_positions(params["positions"], base_table)
        sequence = build_random_sequence(positions, params, default_steps)
        return build_single_dot_sequence(
            sequence,
            base_table,
            experiment_config,
            flickering=bool(params.get("flickering", False)),
            params=params,
        )

    if mode == "rocking":
        positions = parse_positions(params["positions"], base_table)
        sequence = build_rocking_sequence(positions, params, experiment_config, default_steps)
        return build_single_dot_sequence(
            sequence,
            base_table,
            experiment_config,
            flickering=bool(params.get("flickering", False)),
            params=params,
        )

    if mode == "single_flicker":
        position = get_single_flicker_position(params, base_table)
        flicker_params = add_default_duration(params, base_motion_duration)
        return build_fixed_flicker(position, base_table, experiment_config, flicker_params)

    if mode == "multi_flicker":
        positions = parse_positions(params["positions"], base_table)
        flicker_params = add_default_duration(params, base_motion_duration)
        return build_multi_flicker(positions, base_table, experiment_config, flicker_params)

    raise ValueError(f"Unknown stimulus type for {stimulus_key}: {mode}")


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)
    experiment_config = {
        **DEFAULT_EXPERIMENT_CONFIG,
        **raw_config.get("_experiment", {}),
    }
    stimulus_config = {
        key: value for key, value in raw_config.items()
        if key != "_experiment"
    }
    return experiment_config, stimulus_config


def save_outputs(config_path, experiment_config, stimulus_config):
    output_path = _resolve_path(experiment_config["output_path"], config_path)
    output_path.mkdir(parents=True, exist_ok=True)
    parameters_path = output_path / "parameters"
    parameters_path.mkdir(parents=True, exist_ok=True)

    base_table, _base_arc_duration = build_base_position_table(experiment_config)
    base_motion_duration = base_control_duration_sec(base_table, experiment_config)
    base_table.to_csv(parameters_path / "base_position_table.csv", index=False)

    experiment_rows = []
    summary_rows = []
    generated = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for stimulus_key, params in stimulus_config.items():
        print(f"Generating {stimulus_key} ({params['type']})")
        df = generate_stimulus(stimulus_key, params, base_table, experiment_config, base_motion_duration)
        csv_path = output_path / f"{stimulus_key}_trajectory.csv"
        df.to_csv(csv_path, index=False)

        csv_duration_sec = len(df) / float(experiment_config["framerate"])
        pause_before = float(experiment_config.get("pause_before_sec", 0.0))
        pause_after = float(experiment_config.get("pause_after_sec", 0.0))
        total_with_pauses = csv_duration_sec + pause_before + pause_after

        generated.append((stimulus_key, csv_duration_sec, total_with_pauses))
        summary_rows.append(
            {
                "stimulus_key": stimulus_key,
                "type": params["type"],
                "n_frames": len(df),
                "csv_duration_sec": round(csv_duration_sec, 6),
                "pause_before_sec": pause_before,
                "pause_after_sec": pause_after,
                "duration_with_pauses_sec": round(total_with_pauses, 6),
            }
        )
        experiment_rows.append(
            {
                "stimulus_key": stimulus_key,
                **params,
                **{
                    key: value for key, value in experiment_config.items()
                    if key != "output_path"
                },
                "base_motion_duration_sec": base_motion_duration,
                "date": now,
            }
        )

    pd.DataFrame(experiment_rows).to_csv(parameters_path / "experiment_parameters.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(parameters_path / "stimulus_time_summary.csv", index=False)
    total_time_sec = sum(row["duration_with_pauses_sec"] for row in summary_rows)
    pd.DataFrame([{"total_experiment_duration_sec": round(total_time_sec, 6)}]).to_csv(
        parameters_path / "total_time_sec.csv",
        index=False,
    )

    print("\nStimulus durations:")
    for stimulus_key, csv_duration_sec, total_with_pauses in generated:
        print(
            f"{stimulus_key}: {csv_duration_sec:.3f} s CSV "
            f"({total_with_pauses:.3f} s with pauses)"
        )
    print(f"\nTotal experiment time with pauses: {total_time_sec:.2f} sec ({total_time_sec / 60:.2f} min)")
    print("All stimuli generated and saved.")

    return output_path


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate unified social-filter stimulus trajectories.")
    parser.add_argument(
        "--config",
        default=str(script_dir / DEFAULT_CONFIG_FILENAME),
        help="Path to a unified trajectory JSON config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    experiment_config, stimulus_config = load_config(config_path)
    save_outputs(config_path, experiment_config, stimulus_config)


if __name__ == "__main__":
    main()
