"""Deterministic synthetic data for characterization and regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def continuous_traces() -> np.ndarray:
    """Return continuous traces with shape `(time=12, neurons=3)`."""

    time = np.arange(12, dtype=float)
    return np.column_stack(
        [
            time / 10.0,
            np.sin(time / 3.0),
            np.where((time >= 4) & (time <= 7), 1.0, 0.0),
        ]
    )


def trial_aligned_traces() -> np.ndarray:
    """Return aligned traces with shape `(neurons=3, time=6, trials=2)`."""

    values = np.arange(36, dtype=float).reshape(3, 6, 2)
    return values / 10.0


def stimulus_timing() -> dict[str, dict[str, float | int]]:
    """Return two small stimulus timing dictionaries with frame and time units."""

    return {
        "stim_a": {
            "static_before_sec": 1.0,
            "motion_sec": 2.0,
            "total_sec": 3.0,
            "motion_start_frame": 60,
            "motion_end_frame": 180,
            "total_frames": 180,
        },
        "stim_b": {
            "static_before_sec": 0.5,
            "motion_sec": 1.5,
            "total_sec": 2.0,
            "motion_start_frame": 30,
            "motion_end_frame": 120,
            "total_frames": 120,
        },
    }


def stimulus_log_table() -> pd.DataFrame:
    """Return a stimulus log with stable column order and frame units."""

    return pd.DataFrame(
        {
            "stimulus": ["stim_a", "stim_b"],
            "onset_frame": [10, 30],
            "block": [0, 0],
        }
    )


def response_raster() -> np.ndarray:
    """Return a binary response raster with shape `(time=12, neurons=3)`."""

    raster = np.zeros((12, 3), dtype=np.uint8)
    raster[4:7, 0] = 1
    raster[6:10, 1] = 1
    raster[2, 2] = 1
    return raster


def response_table() -> pd.DataFrame:
    """Return a tidy response table with stable row and column order."""

    return pd.DataFrame(
        {
            "fish_id": ["fish_01", "fish_01", "fish_02", "fish_02"],
            "neuron_id": [0, 0, 0, 0],
            "stimulus": ["stim_a", "stim_b", "stim_a", "stim_b"],
            "response": [1.0, 0.5, 0.75, 0.25],
        }
    )
