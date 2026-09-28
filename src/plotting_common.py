"""Shared stimulus styles, labels, colors and reference-line artists."""

import matplotlib.pyplot as plt
from pathlib import Path
import re


def _natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


STATIC_FLICKER_CATEGORY_ORDER = ["non-responsive", "static-only", "shared", "newly recruited"]


STATIC_FLICKER_CATEGORY_COLORS = {
    "non-responsive": "#bdbdbd",
    "static-only": "#4c78a8",
    "shared": "#59a14f",
    "newly recruited": "#e15759",
}


STATIC_FLICKER_STIMULUS_COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00"]


def _vertical_reference_line(axis, x, **style):
    """Draw a vertical marker without Matplotlib's blended axvline transform.

    Some Windows Matplotlib builds recurse while comparing that transform on
    image axes.  A regular data-coordinate line is visually equivalent here.
    """
    y_limits = axis.get_ylim()
    axis.plot([float(x), float(x)], y_limits, **style)
    axis.set_ylim(y_limits)


def _horizontal_reference_line(axis, y, **style):
    """Draw a horizontal marker without Matplotlib's blended axhline transform."""
    x_limits = axis.get_xlim()
    axis.plot(x_limits, [float(y), float(y)], **style)
    axis.set_xlim(x_limits)


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
