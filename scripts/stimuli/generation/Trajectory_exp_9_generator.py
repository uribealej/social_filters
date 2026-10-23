"""Generate RF-memory preconditioning/delay/bout assets and their run records.

Run directly, or with ``python -m scripts.stimuli.generation.Trajectory_exp_9_generator``.
Experiment 9 positions are zero-based; the unified geometry remains unchanged.
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

if __package__:
    from .Trajectory_unified_generator import DEFAULT_EXPERIMENT_CONFIG, build_base_position_table
else:
    from Trajectory_unified_generator import DEFAULT_EXPERIMENT_CONFIG, build_base_position_table


DEFAULTS = {
    **DEFAULT_EXPERIMENT_CONFIG,
    "static_period_sec": 0,
    "bout_positions": [7, 8, 9, 10, 11, 12],
    "distant_positions": [0, 1, 2, 3, 4],
    "bout_on_ms": 300,
    "bout_off_ms": 300,
    "preconditioning_duration_sec": 8,
    "distant_delay_sec": 0.3,
    "overlap_delays_sec": [0.3, 0.6, 2, 4, 8],
}
DEFAULT_CONFIGS = ("trajectory_exp_9_left.json", "trajectory_exp_9_right.json")


def _number(value, name, *, positive=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    return result


def _frames(seconds, fps, *, positive=False):
    frames = int(math.floor(seconds * fps + 0.5))
    if positive and frames < 1:
        raise ValueError("An ON/OFF interval must occupy at least one frame.")
    return frames


def _positions(values, count, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list of zero-based integers.")
    if any(type(value) is not int or not 0 <= value < count for value in values):
        raise ValueError(f"{name} must contain zero-based integers in 0-{count - 1}.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate positions.")
    return list(values)


def load_experiment(config_path):
    """Resolve one side's JSON into validated settings and its geometry table."""
    path = Path(config_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"_experiment"}:
        raise ValueError("Experiment 9 configs must contain one '_experiment' object.")
    config = {**DEFAULTS, **raw["_experiment"]}
    if config.get("hemifield") not in {"left", "right"}:
        raise ValueError("hemifield must be 'left' or 'right'.")
    if config["static_period_sec"] != 0:
        raise ValueError("Experiment 9 has no initial static period; static_period_sec must be 0.")
    for key in ("radius_cm", "speed_cm_sec", "framerate", "update_interval_ms", "dot_size_on",
                "flicker_interval_ms", "bout_on_ms", "bout_off_ms", "preconditioning_duration_sec"):
        config[key] = _number(config[key], key, positive=True)
    for key in ("pause_before_sec", "pause_after_sec", "distant_delay_sec"):
        config[key] = _number(config[key], key)
    for key in ("flicker_on_ms", "flicker_off_ms"):
        config[key] = _number(config.get(key, config["flicker_interval_ms"]), key, positive=True)
    if not np.isfinite(float(config["rotation_angle_deg"])):
        raise ValueError("rotation_angle_deg must be finite.")
    angles = np.asarray(config["angle_ranges"], dtype=float)
    if angles.shape != (1, 2) or not np.isfinite(angles).all():
        raise ValueError("angle_ranges must contain one finite start/end angle pair.")
    expected_sign = -1 if config["hemifield"] == "left" else 1
    if np.any(expected_sign * angles <= 0) or np.any(np.abs(angles) > 180):
        raise ValueError("angle_ranges must stay within the configured left/right hemifield.")
    base, _ = build_base_position_table(config)
    base.insert(0, "position_index", base["position_id"] - 1)
    for key in ("bout_positions", "distant_positions"):
        config[key] = _positions(config[key], len(base), key)
    if set(config["bout_positions"]) & set(config["distant_positions"]):
        raise ValueError("distant_positions must not overlap bout_positions.")
    if not isinstance(config["overlap_delays_sec"], list) or not config["overlap_delays_sec"]:
        raise ValueError("overlap_delays_sec must be a nonempty list.")
    config["overlap_delays_sec"] = [_number(v, "overlap_delays_sec") for v in config["overlap_delays_sec"]]
    destination = Path(config["output_path"])
    config["output_path"] = str(destination if destination.is_absolute() else (path.parent / destination).resolve())
    return {"path": path, "raw": raw, "config": config, "base": base}


def preconditioning_frames(target_sec, on_frames, off_frames, fps):
    """Nearest positive pulse count; full ON pulses, no trailing OFF, ties longer."""
    count = max(1, int(math.floor((target_sec * fps + off_frames) / (on_frames + off_frames) + 0.5)))
    return count, count * on_frames + (count - 1) * off_frames


def condition_specs(config):
    """Return conditions in experimental order, independently of playback ordering."""
    prefix = "Le" if config["hemifield"] == "left" else "Ri"
    specs = [("C1_Bout", "Control 1", "none", 0),
             ("C2_Far", "Control 2", "distant", config["distant_delay_sec"])]
    specs += [(f"S{i}_Overlap", f"Stimulus {i}", "overlap", delay)
              for i, delay in enumerate(config["overlap_delays_sec"], 1)]
    return [{"stimulus_key": prefix + key, "condition": label, "preconditioning": kind,
             "delay_requested_sec": delay} for key, label, kind, delay in specs]


def build_stimulus(config, base, spec):
    """Return a frame-by-column trajectory and explicit, exclusive phase boundaries."""
    fps = config["framerate"]
    bout_on = _frames(config["bout_on_ms"] / 1000, fps, positive=True)
    bout_off = _frames(config["bout_off_ms"] / 1000, fps, positive=True)
    flicker_on = _frames(config["flicker_on_ms"] / 1000, fps, positive=True)
    flicker_off = _frames(config["flicker_off_ms"] / 1000, fps, positive=True)
    kind = spec["preconditioning"]
    pre_positions = ([] if kind == "none" else config["distant_positions"] if kind == "distant"
                     else config["bout_positions"])
    pulses, pre_frames = (preconditioning_frames(config["preconditioning_duration_sec"], flicker_on,
                                                flicker_off, fps) if pre_positions else (0, 0))
    delay_frames = _frames(spec["delay_requested_sec"], fps)
    bout_frames = len(config["bout_positions"]) * (bout_on + bout_off)
    bout_start = pre_frames + delay_frames
    count = bout_start + bout_frames
    dot_count = max(1, len(pre_positions))
    xyz = np.zeros((count, dot_count, 3), dtype=float)
    xy = base[["x", "y"]].to_numpy()
    for dot in range(dot_count):
        xyz[:, dot, :2] = xy[pre_positions[dot] if pre_positions else config["bout_positions"][0]]
    if pre_frames:
        visible = np.arange(pre_frames) % (flicker_on + flicker_off) < flicker_on
        xyz[:pre_frames, :, 2] = visible[:, None] * config["dot_size_on"]
    xyz[pre_frames:, 0, :2] = xy[config["bout_positions"][0]]
    for step, position in enumerate(config["bout_positions"]):
        start = bout_start + step * (bout_on + bout_off)
        xyz[start:start + bout_on + bout_off, 0, :2] = xy[position]
        xyz[start:start + bout_on, 0, 2] = config["dot_size_on"]
    columns = [f"dot{dot}_{axis}" for dot in range(dot_count) for axis in ("x", "y", "radius")]
    trajectory = pd.DataFrame(xyz.reshape(count, -1), columns=columns)
    metadata = {
        **spec, "hemifield": config["hemifield"], "type": "preconditioning_bout",
        "trajectory_file": spec["stimulus_key"] + "_trajectory.csv",
        "framerate": fps, "n_dots": dot_count, "n_frames": count,
        "bout_positions": config["bout_positions"], "preconditioning_positions": pre_positions,
        "preconditioning_requested_sec": config["preconditioning_duration_sec"] if pre_positions else 0,
        "preconditioning_sec": pre_frames / fps, "preconditioning_pulses": pulses,
        "preconditioning_offset_frame": pre_frames, "delay_frames": delay_frames,
        "delay_sec": delay_frames / fps, "bout_onset_frame": bout_start,
        "bout_onset_sec": bout_start / fps, "bout_offset_frame": count,
        "bout_sec": bout_frames / fps, "csv_duration_sec": count / fps,
        "flicker_on_frames": flicker_on, "flicker_off_frames": flicker_off,
        "flicker_on_sec": flicker_on / fps, "flicker_off_sec": flicker_off / fps,
        "bout_on_frames": bout_on, "bout_off_frames": bout_off,
        "bout_on_sec": bout_on / fps, "bout_off_sec": bout_off / fps,
        "pause_before_sec": config["pause_before_sec"], "pause_after_sec": config["pause_after_sec"],
        "duration_with_pauses_sec": count / fps + config["pause_before_sec"] + config["pause_after_sec"],
    }
    return trajectory, metadata


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(str(v) for v in row) + " |" for row in rows]])


def write_timing_workbook(output, records, generated_at, preview_dir=None):
    """Export Excel timing via Artifact Tool, verifying calculated totals before saving."""
    node = shutil.which("node")
    if node is None:
        bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
        if not bundled.is_file():
            raise RuntimeError("Excel timing export requires Node.js and @oai/artifact-tool.")
        node = str(bundled)
    builder = Path(__file__).with_name("exp_9_timing_workbook.mjs")
    command = [node, str(builder), str(Path(output) / "experiment_9_timing.xlsx")]
    if preview_dir is not None:
        command.append(str(preview_dir))
    subprocess.run(command, input=json.dumps({"records": records, "generated_at": generated_at}),
                   text=True, encoding="utf-8", check=True, timeout=120)


OBSOLETE_REPORTS = (
    "stimulus_descriptions.md", "stimuli_standard.png", "stimuli_summary.csv",
    "parameters/experiment_parameters.csv", "parameters/total_time_sec.csv",
)


def remove_obsolete_reports(output, experiments):
    """Remove only named superseded outputs; preserve inputs and unrelated files."""
    output = Path(output).resolve()
    inputs = {e["path"].resolve() for e in experiments}
    relative_paths = [*OBSOLETE_REPORTS,
                      *[f"parameters/configs/{e['path'].name}" for e in experiments]]
    removed = []
    for relative in relative_paths:
        target = (output / relative).resolve()
        if not target.is_relative_to(output):
            raise ValueError(f"Refusing cleanup outside the stimulus folder: {target}")
        if target.is_file() and target not in inputs:
            target.unlink()
            removed.append(relative)
    configs_dir = output / "parameters/configs"
    if configs_dir.is_dir() and not configs_dir.is_symlink() and not any(configs_dir.iterdir()):
        try:
            configs_dir.rmdir()
        except PermissionError:
            pass  # A sync client may lock the empty directory after its files are removed.
    return removed


def write_markdown(output, experiments, records, generated_at):
    """Write one short operational stimulus specification, without rationale."""
    report = [
        "# Experiment 9 stimulus specification",
        f"Generated: {generated_at}",
        "Positions are zero-based. No initial static display. During preconditioning, fixed dots "
        "flicker simultaneously and synchronously in the same hemifield as the bout. "
        "Each bout uses one dot advancing through the listed positions, ON then OFF at each position, "
        "including the final OFF interval.",
        "Preconditioning ends after a complete ON pulse, with duration nearest the requested target. "
        "The blank delay runs from that pulse's offset to the first bout ON frame.",
    ]
    for experiment in experiments:
        config = experiment["config"]
        first = next(r for r in records if r["hemifield"] == config["hemifield"])
        report.append(
            f"**{config['hemifield'].capitalize()}:** bout positions {config['bout_positions']}; "
            f"bout ON/OFF {first['bout_on_sec']:g}/{first['bout_off_sec']:g} s; "
            f"flicker ON/OFF {first['flicker_on_sec']:g}/{first['flicker_off_sec']:g} s; "
            f"{config['framerate']:g} Hz. External pauses: {config['pause_before_sec']:g} s before "
            f"and {config['pause_after_sec']:g} s after each CSV."
        )
    report += [
        "Each stimulus key below identifies `<key>_trajectory.csv`. Durations are actual seconds. "
        "An empty preconditioning position list means bout alone.",
        _table(["Stimulus", "Side", "Pre positions", "Pre (s)", "Blank (s)", "Bout (s)"],
               [[r["stimulus_key"], r["hemifield"], r["preconditioning_positions"],
                 f"{r['preconditioning_sec']:g}", f"{r['delay_sec']:g}", f"{r['bout_sec']:g}"]
                for r in records]),
        "[Excel timing workbook](experiment_9_timing.xlsx): full timings, totals, and frame boundaries. "
        "[Resolved settings](parameters/resolved_settings.json): all generation parameters. "
        "[Timing CSV](parameters/stimulus_time_summary.csv): machine-readable phase metadata. "
        "Position coordinates are in `parameters/base_position_table_left.csv` and "
        "`parameters/base_position_table_right.csv`.",
    ]
    (Path(output) / "experiment_9_report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")


def save_experiments(experiments, output, *, excel_preview_dir=None):
    """Validate the whole output group before writing CSVs and their records."""
    output = Path(output).resolve()
    experiments = [{**e, "config": {**e["config"], "output_path": str(output)}} for e in experiments]
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    generated = []
    sides = [e["config"]["hemifield"] for e in experiments]
    if len(set(sides)) != len(sides):
        raise ValueError("Supply at most one config per hemifield in each output folder.")
    if len({e["path"].name for e in experiments}) != len(experiments):
        raise ValueError("Config filenames must be unique within an output folder.")
    for experiment in experiments:
        for spec in condition_specs(experiment["config"]):
            generated.append(build_stimulus(experiment["config"], experiment["base"], spec))
    expected = {r["trajectory_file"] for _, r in generated}
    unexpected = {p.name for p in output.glob("*_trajectory.csv")} - expected
    if unexpected:
        raise ValueError(f"Output contains other trajectories: {sorted(unexpected)}. Supply all configs or choose a new --output folder.")
    parameters = output / "parameters"
    parameters.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        side = experiment["config"]["hemifield"]
        name = f"parameters/base_position_table_{side}.csv"
        experiment["base"].to_csv(output / name, index=False)
    resolved = {e["config"]["hemifield"]: e["config"] for e in experiments}
    (parameters / "resolved_settings.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for trajectory, record in generated:
        trajectory.to_csv(output / record["trajectory_file"], index=False)
    records = [r for _, r in generated]
    serializable = [{key: json.dumps(value) if isinstance(value, list) else value for key, value in r.items()} for r in records]
    pd.DataFrame(serializable).to_csv(parameters / "stimulus_time_summary.csv", index=False)
    write_timing_workbook(output, records, generated_at, preview_dir=excel_preview_dir)
    write_markdown(output, experiments, records, generated_at)
    remove_obsolete_reports(output, experiments)
    print(f"Generated {len(records)} stimuli in {output}")
    print(f"CSV: {sum(r['csv_duration_sec'] for r in records):.3f} s; "
          f"with external pauses: {sum(r['duration_with_pauses_sec'] for r in records):.3f} s.")
    print("Report: experiment_9_report.md; timings: experiment_9_timing.xlsx")
    return records


def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", type=Path,
                        default=[root / "configs" / "stimuli" / name for name in DEFAULT_CONFIGS])
    parser.add_argument("--output", type=Path, help="Override the configured output directory.")
    args = parser.parse_args(argv)
    groups = defaultdict(list)
    for path in args.config:
        experiment = load_experiment(path)
        output = args.output.resolve() if args.output else Path(experiment["config"]["output_path"])
        experiment["config"]["output_path"] = str(output)
        groups[output].append(experiment)
    for output, experiments in groups.items():
        save_experiments(experiments, output)


if __name__ == "__main__":
    main()
