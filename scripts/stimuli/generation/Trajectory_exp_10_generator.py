"""Generate the radial RF-preconditioning experiment and its reproducible reports.

Run from the repository with ``python -m scripts.stimuli.generation.Trajectory_exp_10_generator``.
CSV coordinates and displayed dot radii are centimetres; configured diameters are millimetres.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.stimuli.generation.Trajectory_exp_9_generator import _frames, preconditioning_frames
from scripts.stimuli.generation.Trajectory_unified_generator import DEFAULT_EXPERIMENT_CONFIG, build_base_position_table
from src.data_loading import resolve_lab_root
from src.stimuli_timeline import validate_phase_metadata


DEFAULT_CONFIGS = ("trajectory_exp_10_left.json", "trajectory_exp_10_right.json")


def load_experiment(config_path):
    """Validate settings and resolve explicit angles or legacy speed-based geometry."""
    path = Path(config_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"_experiment"}:
        raise ValueError("Expected one _experiment configuration object.")
    config = {**DEFAULT_EXPERIMENT_CONFIG, **raw["_experiment"]}
    if config["hemifield"] not in ("left", "right") or config["static_period_sec"] != 0:
        raise ValueError("Supply left/right hemifield and disable initial static display.")
    positive = ("radius_cm", "speed_cm_sec", "framerate", "update_interval_ms",
                "bout_on_ms", "bout_off_ms", "flicker_on_ms", "flicker_off_ms",
                "preconditioning_duration_sec", "blank_pause_sec")
    for name in positive:
        value = float(config[name])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
        config[name] = value
    for name in ("pause_before_sec", "pause_after_sec"):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.isfinite(config["rotation_angle_deg"]) or config["framerate"] != 60:
        raise ValueError("Use a finite rotation and the 60-Hz phase contract.")
    count = config["n_positions"]
    if type(count) is not int or count < 2:
        raise ValueError("n_positions must be an integer of at least two.")
    if not np.isclose(config["bout_on_ms"] + config["bout_off_ms"], config["update_interval_ms"]):
        raise ValueError("Bout ON + OFF must equal the geometry update interval.")
    if not 0 < config["start_angle_deg"] < config["max_angle_deg"] <= 180:
        raise ValueError("Start/max angles must be positive and inside one hemifield.")
    corrections = config["dot_diameters_mm"]
    radii = config["preconditioning_radii_mm"]
    if not radii or len(radii) != len(set(radii)) or any(not np.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("Preconditioning radii must be positive, finite and unique.")
    for radius in [*radii, config["radius_cm"] * 10, config["pre_only_radius_mm"]]:
        key = f"{radius:g}"
        if key not in corrections or not np.isfinite(corrections[key]) or corrections[key] <= 0:
            raise ValueError(f"Missing positive dot diameter for radius {radius:g} mm.")
    sign = -1 if config["hemifield"] == "left" else 1
    config["angle_ranges"] = [[sign * config["start_angle_deg"], sign * config["max_angle_deg"]]]
    if "position_angles_deg" in config:
        angles = np.asarray(config["position_angles_deg"], dtype=float)
        if (angles.shape != (count,) or not np.isfinite(angles).all()
                or np.any(angles < config["start_angle_deg"])
                or np.any(angles > config["max_angle_deg"])
                or np.any(np.diff(angles) <= 0)):
            raise ValueError("position_angles_deg must contain n_positions finite, increasing "
                             "positive angles within the configured limits.")
        # Reuse the angular positions at every radius; speed must not resample them.
        radians = np.deg2rad(sign * angles)
        rotation = np.deg2rad(config["rotation_angle_deg"])
        x, y = config["radius_cm"] * np.sin(radians), config["radius_cm"] * np.cos(radians)
        base = pd.DataFrame({"position_id": np.arange(1, count + 1), "angle_deg": sign * angles,
                             "x": np.round(x * np.cos(rotation) - y * np.sin(rotation), 3),
                             "y": np.round(x * np.sin(rotation) + y * np.cos(rotation), 3)})
    else:
        # Legacy configs retain speed-based sampling, which may extend past the end angle.
        base, _ = build_base_position_table(config)
        if len(base) < count or np.abs(base.iloc[:count].angle_deg).max() > config["max_angle_deg"]:
            raise ValueError("Requested positions do not fit the configured angular limits.")
        base = base.iloc[:count].copy()
    base.insert(0, "position_index", np.arange(count))
    config["bout_angles_deg"] = base.angle_deg.tolist()
    config["dot_size_on"] = round(config["dot_diameters_mm"][f'{config["radius_cm"] * 10:g}'] / 20, 8)
    return {"path": path, "config": config, "base": base}


def position_tables(config, base):
    """Scale the bout geometry to all radii without recalculating angular spacing.

    Returns one row per position/radius; x/y and dot_radius_cm are centimetres.
    """
    parts = []
    for radius in sorted(set(config["preconditioning_radii_mm"] + [config["radius_cm"] * 10,
                                                                     config["pre_only_radius_mm"]])):
        table = base.copy()
        angles = np.deg2rad(table.angle_deg.to_numpy())
        rotation = np.deg2rad(config["rotation_angle_deg"])
        x, y = radius / 10 * np.sin(angles), radius / 10 * np.cos(angles)
        table["x"] = np.round(x * np.cos(rotation) - y * np.sin(rotation), 6)
        table["y"] = np.round(x * np.sin(rotation) + y * np.cos(rotation), 6)
        table["position_radius_mm"] = radius
        table["dot_diameter_mm"] = config["dot_diameters_mm"][f"{radius:g}"]
        table["dot_radius_cm"] = (table.dot_diameter_mm / 20).round(8)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def condition_specs(config):
    """Return stable report order, independently of future presentation order."""
    prefix = "Le" if config["hemifield"] == "left" else "Ri"
    specs = [("C1_Bout", None, True)]
    specs += [(f"S{i}_Pre{r:g}", r, True) for i, r in enumerate(config["preconditioning_radii_mm"], 1)]
    specs += [(f"C2_Pre{config['pre_only_radius_mm']:g}Only", config["pre_only_radius_mm"], False)]
    return [dict(stimulus_key=prefix + key, pre_radius_mm=radius, bout_present=bout)
            for key, radius, bout in specs]


def build_stimulus(config, tables, spec):
    """Return frame-by-column CSV data and explicit end-exclusive phase metadata."""
    fps = config["framerate"]
    on = _frames(config["bout_on_ms"] / 1000, fps, positive=True)
    off = _frames(config["bout_off_ms"] / 1000, fps, positive=True)
    flicker_on = _frames(config["flicker_on_ms"] / 1000, fps, positive=True)
    flicker_off = _frames(config["flicker_off_ms"] / 1000, fps, positive=True)
    has_pre = spec["pre_radius_mm"] is not None
    pulses, pre_length = (preconditioning_frames(config["preconditioning_duration_sec"], flicker_on, flicker_off, fps)
                          if has_pre else (0, 0))
    pause = _frames(config["blank_pause_sec"], fps) if has_pre and spec["bout_present"] else 0
    test_length = config["n_positions"] * (on + off) if spec["bout_present"] else 0
    test_start, total = pre_length + pause, pre_length + pause + test_length
    dot_count = config["n_positions"] if has_pre else 1
    values = np.zeros((total, dot_count, 3), dtype=float)
    bout = tables.loc[tables.position_radius_mm == config["radius_cm"] * 10]
    bout_xy = bout[["x", "y"]].to_numpy()
    values[:, :, :2] = bout_xy[0]
    if has_pre:
        pre = tables.loc[tables.position_radius_mm == spec["pre_radius_mm"]]
        values[:, :, :2] = pre[["x", "y"]].to_numpy()[None, :, :]
        visible = np.arange(pre_length) % (flicker_on + flicker_off) < flicker_on
        values[:pre_length, :, 2] = visible[:, None] * float(pre.dot_radius_cm.iloc[0])
    if spec["bout_present"]:
        for i, xy in enumerate(bout_xy):
            start = test_start + i * (on + off)
            values[start:start + on + off, 0, :2] = xy
            values[start:start + on, 0, 2] = float(bout.dot_radius_cm.iloc[0])
    frame = pd.DataFrame(values.reshape(total, -1), columns=[f"dot{i}_{axis}" for i in range(dot_count)
                                                           for axis in ("x", "y", "radius")])
    metadata = {
        **spec, "hemifield": config["hemifield"], "trajectory_file": spec["stimulus_key"] + "_trajectory.csv",
        "framerate": fps, "n_frames": total, "n_dots": dot_count,
        "initial_blank_sec": 0,
        "preconditioning_onset_frame": 0,
        "preconditioning_offset_frame": pre_length,
        "preconditioning_sec": pre_length / fps if has_pre else 0,
        "preconditioning_requested_sec": config["preconditioning_duration_sec"] if has_pre else 0,
        "preconditioning_pulses": pulses if has_pre else 0,
        "delay_sec": pause / fps, "delay_frames": pause,
        "test_window_onset_frame": test_start, "test_window_offset_frame": total,
        "test_window_sec": test_length / fps,
        "bout_onset_frame": test_start if spec["bout_present"] else None,
        "bout_offset_frame": total if spec["bout_present"] else None,
        "bout_sec": test_length / fps if spec["bout_present"] else 0,
        "observation_sec": 0,
        "csv_duration_sec": total / fps,
        "bout_on_frames": on, "bout_off_frames": off,
        "flicker_on_frames": flicker_on, "flicker_off_frames": flicker_off,
        "pause_before_sec": config["pause_before_sec"], "pause_after_sec": config["pause_after_sec"],
        "duration_with_pauses_sec": total / fps + config["pause_before_sec"] + config["pause_after_sec"],
    }
    return frame, metadata


def archive_outputs(output):
    """Preserve an existing package before replacing any generated files."""
    output = Path(output)
    if not output.exists() or not any(output.iterdir()):
        return None
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_%f")
    destination = output.parent / "archive" / (stamp + "_stimuli")
    shutil.copytree(output, destination)
    return destination


def write_descriptions(root, records, experiments):
    """Write operational descriptions and provenance, without scientific outcome claims."""
    definitions = root / "definitions"
    config = experiments[0]["config"]
    bout_record = next(r for r in records if r["pre_radius_mm"] is None)
    pre_record = next(r for r in records if r["pre_radius_mm"] is not None and r["bout_present"])
    pre_only = next(r for r in records if not r["bout_present"])
    pre_sec, pause_sec = pre_record["preconditioning_sec"], pre_record["delay_sec"]
    report = ["# Exp. 10 RF-model distance stimuli", "",
              "Status: planned experiment; generated stimuli, not recorded results.", "",
              "Each CSV contains x/y coordinates and displayed dot radii in cm. Configuration dot sizes are diameters in mm.",
              f"Bout: one dot, {config['n_positions']} positions, {config['bout_on_ms']:g} ms ON / {config['bout_off_ms']:g} ms OFF, radius {config['radius_cm'] * 10:g} mm. The {config['rotation_angle_deg']:g}-degree coordinate rotation is applied once.",
              "Angles before rotation: right " + ", ".join(f"{abs(a):.3f}" for a in config["bout_angles_deg"]) + " degrees; left uses the negative angles.",
              "Explicit position_angles_deg take precedence over speed/radius sampling when configured; every preconditioning radius uses the same angles.",
              f"Preconditioning: all {config['n_positions']} fixed dots flicker together for {pre_sec:g} s ({pre_record['preconditioning_pulses']} complete ON pulses; requested target {config['preconditioning_duration_sec']:g} s). Combined conditions then have exactly {pause_sec:g} s blank before the bout.",
              f"Standalone controls have no blank padding: bout alone is {bout_record['csv_duration_sec']:g} s; preconditioning only at {config['pre_only_radius_mm']:g} mm is {pre_only['csv_duration_sec']:g} s. Combined conditions last {pre_record['csv_duration_sec']:g} s.",
              f"CSV frame rate: {config['framerate']:g} Hz. Bout ON/OFF cadence is unchanged, including the final bout OFF interval. Preconditioning-only controls have no bout or blank observation window; their test-window boundaries coincide at the CSV end.",
              f"External pauses: {config['pause_before_sec']:g} s before and {config['pause_after_sec']:g} s after each CSV; these are not encoded in the CSVs.", "",
              "## Conditions", "", "| Stimulus | Preconditioning radius (mm) | Test bout |", "|---|---:|---|"]
    for r in records:
        pre = "None" if r["pre_radius_mm"] is None else f"{r['pre_radius_mm']:g}"
        report.append(f"| {r['stimulus_key']} | {pre} | {str(config['radius_cm'] * 10) + ' mm' if r['bout_present'] else 'None'} |")
    report += ["", "## Accepted provisional size correction", "",
               "| Position radius (mm) | Dot diameter (mm) |", "|---:|---:|"]
    report += [f"| {radius:g} | {config['dot_diameters_mm'][f'{radius:g}']:.2f} |" for radius in config["preconditioning_radii_mm"]]
    report += ["", "Reference: 4.0 mm diameter at 18 mm. Accepted sizes approximate equal angular area for circular dots in a homogeneous-water, centred-eye model with an assumed 13 mm eye-to-image-plane height.",
               "These fixed accepted values are not recalculated at generation time. Dish-bottom thickness, eye-height uncertainty, refraction through the dish, and physical projector calibration remain unverified. Apparent shape is not fully matched.",
               "Partial spatial overlap at 14 mm is intentional. The 18 mm preconditioning-only control measures carry-over at that radius only.",
               "", "## Reproduction and provenance", "",
               "From the social_filters repository: `python -m scripts.stimuli.generation.Trajectory_exp_10_generator`.",
               "Code: `scripts/stimuli/generation/Trajectory_exp_10_generator.py`; inputs: `configs/stimuli/trajectory_exp_10_left.json` and `trajectory_exp_10_right.json` in that external repository.",
               "The generator reuses unified geometry and experiment 9 pulse completion. Effective settings, source hashes, and repository commit are recorded in [parameters/resolved_settings.json](parameters/resolved_settings.json).",
               "[Timing summary](parameters/stimulus_time_summary.csv) contains explicit phases and actual frame boundaries. Missing bout boundaries mean no bout was presented.",
               "[Summary](stimuli_summary.csv) and the standard, polar, geometry, and timing PNGs are derived from the saved CSVs. The geometry figure displays physical radii and dot footprints; the polar figure displays angular coverage only.",
               f"One complete set: {sum(r['csv_duration_sec'] for r in records):g} s CSV content; {sum(r['duration_with_pauses_sec'] for r in records):g} s with configured external pauses, excluding scheduling/hardware overhead. Condition order is report order, not a prescribed recording sequence.", ""]
    (definitions / "STIMULUS_DESCRIPTIONS.md").write_text("\n".join(report), encoding="utf-8")
    (root / "README.md").write_text(
        "# Stimulus files\n\n`definitions/` contains shared trajectories, STIMULUS_DESCRIPTIONS.md, stimuli_summary.csv, four graphical previews, and parameters/.\n\n"
        "`recordings/<fish_id>/` is reserved for genuine block logs, experiment logs, and trial sequences. No recordings have been generated.\n\n"
        "See [the stimulus specification](definitions/STIMULUS_DESCRIPTIONS.md) for parameters, provenance, and regeneration.\n", encoding="utf-8")
    (root / "recordings").mkdir(exist_ok=True)


def save_experiments(experiments, output, *, previews=True):
    """Validate the full two-side package, preserve originals, and write new assets."""
    if sorted(e["config"]["hemifield"] for e in experiments) != ["left", "right"]:
        raise ValueError("Generate both hemifields together for a complete package.")
    left, right = (e["config"] for e in sorted(experiments, key=lambda e: e["config"]["hemifield"]))
    side_specific = {"hemifield", "angle_ranges", "bout_angles_deg"}
    if any(left.get(key) != right.get(key) for key in (set(left) | set(right)) - side_specific):
        raise ValueError("Both sides must share every setting except hemifield and signed angles.")
    generated, tables = [], {}
    for experiment in experiments:
        config = experiment["config"]
        tables[config["hemifield"]] = position_tables(config, experiment["base"])
        generated.extend(build_stimulus(config, tables[config["hemifield"]], spec)
                         for spec in condition_specs(config))
    records = [record for _, record in generated]
    output = Path(output).resolve()
    expected = {r["trajectory_file"] for r in records}
    unexpected = {p.name for p in (output / "definitions").glob("*_trajectory.csv")} - expected
    if unexpected:
        raise ValueError(f"Output contains other trajectories: {sorted(unexpected)}. Use a new package directory.")
    archive_outputs(output)
    definitions = output / "definitions"
    parameters = definitions / "parameters"
    parameters.mkdir(parents=True, exist_ok=True)
    for frame, record in generated:
        frame.to_csv(definitions / record["trajectory_file"], index=False)
    timings = pd.DataFrame(records)
    timings.to_csv(parameters / "stimulus_time_summary.csv", index=False)
    validate_phase_metadata(timings, definitions)
    for side, table in tables.items():
        table.to_csv(parameters / f"base_position_table_{side}.csv", index=False)
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    sources = [Path(__file__), Path(__file__).with_name("Trajectory_unified_generator.py"),
               Path(__file__).with_name("Trajectory_exp_9_generator.py"), repo / "src/stimuli_timeline.py",
               repo / "src/stimulus_visualization.py", *[e["path"] for e in experiments]]
    provenance = dict(generated_at=datetime.now().astimezone().isoformat(), repository_commit=commit,
                      source_sha256={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      note="Source hashes identify the working files; generated code may be uncommitted.")
    resolved = {e["config"]["hemifield"]: {**e["config"], "output_path": str(definitions)} for e in experiments}
    resolved["provenance"] = provenance
    (parameters / "resolved_settings.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    write_descriptions(output, records, experiments)
    if previews:
        from src.stimulus_visualization import export_stimulus_previews
        export_stimulus_previews(definitions, timings, framerate=experiments[0]["config"]["framerate"],
                                 rotation_angle_deg=experiments[0]["config"]["rotation_angle_deg"])
    return timings


def main(argv=None):
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", default=[repo / "configs/stimuli" / name for name in DEFAULT_CONFIGS])
    parser.add_argument("--output", type=Path, help="Override the default stimuli package directory.")
    args = parser.parse_args(argv)
    experiments = [load_experiment(p) for p in args.config]
    ids = {e["config"]["experiment_id"] for e in experiments}
    if len(ids) != 1:
        raise ValueError("Both sides must use the same experiment id.")
    output = args.output or resolve_lab_root() / "Results/Paper/2p_derived" / ids.pop() / "stimuli"
    timings = save_experiments(experiments, output)
    print(f"Generated {len(timings)} stimuli in {output}")
    print(f"CSV content: {timings.csv_duration_sec.sum():g} s; with external pauses: {timings.duration_with_pauses_sec.sum():g} s.")


if __name__ == "__main__":
    main()
