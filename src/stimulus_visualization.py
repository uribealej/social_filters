"""Trajectory inspection and report exports, separate from calcium analysis.

CSV radius/size describes the displayed dot, not its distance from the fish.
Spatial coverage uses visible samples; time traces retain every frame.
"""

import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .stimuli_timeline import get_angles_from_positions


def load_stimulus_trajectories(
    input_dir, *, pattern="*_trajectory.csv", framerate=60, rotation_angle_deg=45
):
    """Read each CSV once into samples with angle, dot size, and visibility.

    Accept plain x/y or dotN_x/dotN_y, with optional radius (preferred) or
    size. Missing size means visible; nonfinite or nonpositive size means invisible.
    Nonfinite coordinates and the origin have undefined angles (NaN).
    Empty files and incomplete coordinate pairs raise an informative error.
    """
    if not np.isfinite(framerate) or framerate <= 0:
        raise ValueError("framerate must be finite and greater than zero")
    if not np.isfinite(rotation_angle_deg):
        raise ValueError("rotation_angle_deg must be finite")
    files = sorted(Path(input_dir).glob(pattern), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No {pattern!r} files in {input_dir}")

    parts = []
    for path in files:
        try:
            wide = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{path.name}: empty trajectory CSV") from exc
        if wide.empty:
            raise ValueError(f"{path.name}: empty trajectory CSV")
        columns = {str(c).lower(): c for c in wide.columns}
        dots = sorted(
            {m[1] for c in columns if (m := re.fullmatch(r"(dot\d+)_(x|y|radius|size)", c))},
            key=lambda dot: int(dot[3:]),
        )
        prefixes = [(dot, dot + "_") for dot in dots] if dots else [("dot0", "")]
        for dot, prefix in prefixes:
            if any(prefix + axis not in columns for axis in ("x", "y")):
                raise ValueError(f"{path.name}: {dot} requires both x and y columns")
            x, y = (
                pd.to_numeric(wide[columns[prefix + axis]], errors="coerce").to_numpy(dtype=float)
                for axis in ("x", "y")
            )
            size_col = next((columns[prefix + key] for key in ("radius", "size")
                             if prefix + key in columns), None)
            size = (pd.to_numeric(wide[size_col], errors="coerce").to_numpy(dtype=float)
                    if size_col is not None else np.full(len(wide), np.nan))
            valid = np.isfinite(x) & np.isfinite(y) & ((x != 0) | (y != 0))
            angles = np.full(len(wide), np.nan)
            angles[valid] = (get_angles_from_positions(x[valid], y[valid], rotation_angle_deg) + 180) % 360 - 180
            visible = valid & ((np.isfinite(size) & (size > 0)) if size_col is not None else True)
            parts.append(pd.DataFrame({
                "stimulus_name": path.stem.removesuffix("_trajectory"),
                "dot_id": dot,
                "time_s": np.arange(len(wide)) / framerate,
                "angle_deg": angles,
                "dot_size": size,
                "visible": visible,
            }))
    return pd.concat(parts, ignore_index=True)


def summarize_stimuli(samples):
    """One row per stimulus; unique visible directions at 0.1 degree precision.

    n_dots counts recorded identities, including dots that never become visible.
    Polar plots consume this exact table so their positions match the export.
    """
    rows = []
    for name, group in samples.groupby("stimulus_name", sort=True):
        angles = group.loc[group["visible"], "angle_deg"].to_numpy()
        angles = np.round((angles[np.isfinite(angles)] + 180) % 360 - 180, 1)
        angles[angles == 180] = -180
        angles[angles == 0] = 0  # avoid displaying negative zero
        unique = np.unique(angles).tolist()
        rows.append({"stimulus_name": name, "unique_angles_deg": unique,
                     "n_unique_angles": len(unique), "n_dots": group["dot_id"].nunique()})
    return pd.DataFrame(rows, columns=["stimulus_name", "unique_angles_deg", "n_unique_angles", "n_dots"])


def _panel_grid(count, cols, panel_size, *, polar=False):
    if count == 0:
        raise ValueError("No stimuli to plot")
    if not isinstance(cols, int) or cols < 1:
        raise ValueError("cols must be a positive integer")
    cols = min(cols, count)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(panel_size[0] * cols, panel_size[1] * rows),
        squeeze=False, layout="constrained",
        subplot_kw={"projection": "polar"} if polar else {},
    )
    axes = axes.ravel()
    for ax in axes[count:]:
        ax.set_visible(False)
    return fig, axes[:count]


def _unwrap_finite_segments(angles):
    """Unwrap each valid run without bridging gaps or propagating NaNs."""
    result = np.array(angles, dtype=float, copy=True)
    indices = np.flatnonzero(np.isfinite(result))
    for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if segment.size:
            result[segment] = np.rad2deg(np.unwrap(np.deg2rad(result[segment])))
    return result


def plot_stimulus_traces(samples, *, cols=3, panel_size=(6, 3.8)):
    """Angle (solid) and dot size (dashed) versus time, one panel per stimulus.

    Return a Figure without displaying it. Angles are unwrapped only here.
    Dot colors are shared across panels; no units are inferred for dot size.
    """
    groups = list(samples.groupby("stimulus_name", sort=True))
    fig, axes = _panel_grid(len(groups), cols, panel_size)
    dots = sorted(samples["dot_id"].unique(), key=lambda dot: int(dot[3:]))
    cmap = plt.get_cmap("tab10" if len(dots) <= 10 else "turbo")
    colors = {dot: cmap(i if len(dots) <= 10 else i / max(len(dots) - 1, 1))
              for i, dot in enumerate(dots)}
    finite_sizes = samples.loc[np.isfinite(samples["dot_size"]), "dot_size"]
    size_max = max(float(finite_sizes.max()), 0) if not finite_sizes.empty else 0
    for ax, (name, group) in zip(axes, groups):
        size_ax = ax.twinx()
        for dot, trace in group.groupby("dot_id", sort=False):
            trace = trace.sort_values("time_s")
            ax.plot(trace["time_s"], _unwrap_finite_segments(trace["angle_deg"]),
                    color=colors[dot], lw=1.4, label=dot)
            size_ax.plot(trace["time_s"], trace["dot_size"], color=colors[dot],
                         lw=1.1, ls="--", alpha=0.7)
        ax.set(title=name, xlabel="Time (s)", ylabel="Angle (deg; solid)")
        size_ax.set_ylabel("Dot size (CSV units; dashed)")
        size_ax.set_ylim(0, size_max * 1.08 if size_max > 0 else 1)
        ax.grid(alpha=0.25)
        handles = [Line2D([], [], color=colors[dot], label=dot)
                   for dot in dots if dot in set(group["dot_id"])]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=min(5, len(handles)), fontsize=8, frameon=False)
        if not np.isfinite(group["dot_size"]).any():
            size_ax.set_yticks([])
            size_ax.set_ylabel("Dot size unavailable")
    return fig


def plot_stimulus_positions(summary, *, cols=4, panel_size=(3.8, 3.8)):
    """Plot the summary's unique directions on a ring, with fish at the center."""
    fig, axes = _panel_grid(len(summary), cols, panel_size, polar=True)
    ticks = np.arange(0, 360, 45)
    labels = [f"{(int(t) + 180) % 360 - 180}\N{DEGREE SIGN}" for t in ticks]
    for ax, row in zip(axes, summary.itertuples(index=False)):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(ticks, labels)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([1], labels=[""])
        ax.spines["polar"].set_visible(False)
        ax.grid(alpha=0.3)
        ax.scatter(np.deg2rad(row.unique_angles_deg), np.ones(len(row.unique_angles_deg)),
                   s=28, color="tab:blue", zorder=3)
        ax.scatter([0], [0], s=35, marker="^", color="0.25", zorder=3)
        ax.annotate("fish", xy=(0, 0), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=8, color="0.35")
        ax.set_title(row.stimulus_name, pad=32)
        if not row.unique_angles_deg:
            ax.text(0.5, 0.25, "No visible directions", transform=ax.transAxes,
                    ha="center", fontsize=9)
    return fig


def save_stimulus_report(summary, standard_figure, polar_figure, output_dir):
    """Save both 300-dpi PNGs and a CSV with complete JSON angle lists.

    Existing report files are replaced. IO errors propagate to the caller.
    Returns the output directory for a concise notebook completion message.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    standard_figure.savefig(output_dir / "stimuli_standard.png", dpi=300)
    polar_figure.savefig(output_dir / "stimuli_polar.png", dpi=300)
    exported = summary.copy()
    exported["unique_angles_deg"] = exported["unique_angles_deg"].map(json.dumps)
    exported.to_csv(output_dir / "stimuli_summary.csv", index=False)
    return output_dir
