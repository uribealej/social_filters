"""Reusable assertions for scientific arrays, tables, and figures."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def assert_array_equal_contract(
    testcase: unittest.TestCase,
    actual,
    expected,
    *,
    shape=None,
    dtype=None,
) -> None:
    """Assert exact array values plus optional shape and dtype contracts."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if shape is not None:
        testcase.assertEqual(tuple(shape), actual_array.shape)
    if dtype is not None:
        testcase.assertEqual(np.dtype(dtype), actual_array.dtype)
    np.testing.assert_array_equal(actual_array, expected_array)


def assert_array_allclose_contract(
    testcase: unittest.TestCase,
    actual,
    expected,
    *,
    shape=None,
    rtol=1e-7,
    atol=0.0,
    equal_nan=True,
) -> None:
    """Assert numerical agreement plus an optional shape contract."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if shape is not None:
        testcase.assertEqual(tuple(shape), actual_array.shape)
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


def assert_frame_contract(
    testcase: unittest.TestCase,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    columns=None,
    check_dtype=True,
) -> None:
    """Assert DataFrame values, index, column order, and optional schema."""

    testcase.assertIsInstance(actual, pd.DataFrame)
    if columns is not None:
        testcase.assertEqual(list(columns), list(actual.columns))
    pd.testing.assert_frame_equal(actual, expected, check_dtype=check_dtype)


def assert_figure_contract(
    testcase: unittest.TestCase,
    figure: Figure,
    *,
    expected_axes=None,
    render=True,
) -> None:
    """Check a figure's structure and optionally render its canvas."""

    testcase.assertIsInstance(figure, Figure)
    if expected_axes is not None:
        testcase.assertEqual(expected_axes, len(figure.axes))
    if render:
        figure.canvas.draw()
    testcase.assertGreater(figure.get_figwidth(), 0)
    testcase.assertGreater(figure.get_figheight(), 0)
