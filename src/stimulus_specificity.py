"""Neuron response summaries, stimulus specificity and selectivity ordering."""

import numpy as np
import pandas as pd

from src.response_selectivity import (
    classify_stimulus_specificity_neuron,
    compute_stimulus_selectivity_metrics,
)


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
