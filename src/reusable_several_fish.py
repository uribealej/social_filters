"""Reusable orchestration helpers for several-fish calcium analysis notebooks."""

import json
import platform
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import src.analysis_tools as at
import src.data_loading as exio
import src.multifish_analysis as mfa
import src.plotting as plott


def _json_safe(value):
    """Convert common notebook objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _slugify_label(label):
    """Make a short label safe for folder and file names."""
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(label).strip())
    clean = "-".join(part for part in clean.split("-") if part)
    return clean or "run"


def load_and_preflight_fish_raster_inputs(settings):
    """Quietly load a configured fish cohort and validate its raster stimulus order.

    Returns the loaded data, resolved paths, and reference-fish metadata used by
    several-fish raster notebooks. Loader output is suppressed on success; any
    unavailable fish or stimulus raises one concise exception.
    """
    settings = dict(settings)
    timing = dict(settings["timing"])
    fish_ids = list(settings["fish_ids"])
    stim_order = list(settings["stim_order"])
    if not fish_ids:
        raise ValueError("settings['fish_ids'] must contain at least one fish.")
    if not stim_order:
        raise ValueError("settings['stim_order'] must contain at least one stimulus ID.")

    data_base = Path(settings["data_base"])
    onedrive_candidates = [
        data_base / "OneDrive - Universite de Lausanne",
        *sorted(data_base.glob("OneDrive - Universit* de Lausanne")),
    ]
    onedrive_root = next((path for path in onedrive_candidates if path.exists()), None)
    if onedrive_root is None:
        raise FileNotFoundError(f"Could not find the OneDrive Lab folder below: {data_base}")

    main_path = onedrive_root / "Lab" / "Data" / "2p"
    analysis_path = onedrive_root / "Lab" / "Analysis"
    all_fish_data = {}
    load_errors = {}
    for fish_id in fish_ids:
        try:
            with redirect_stdout(StringIO()):
                all_fish_data[fish_id] = exio.load_and_align_2p_experiment(
                    fish_id=fish_id,
                    experiment_name=settings["experiment_name"],
                    main_path=main_path,
                    stimuli_main_path=analysis_path,
                    fps_2p=timing["fps_2p"],
                    selected_blocks=settings["selected_blocks"],
                    t_pre_s=timing["t_pre_s"],
                    t_post_s=timing["t_post_s"],
                    verbose=settings.get("verbose_loading", False),
                )
        except Exception as error:
            load_errors[fish_id] = f"{type(error).__name__}: {error}"

    if load_errors:
        details = "; ".join(f"{fish_id}: {message}" for fish_id, message in load_errors.items())
        raise RuntimeError(f"Could not load all configured fish. {details}")

    for fish_id, fish_data in all_fish_data.items():
        try:
            at.resolve_selected_stimuli(
                stim_order,
                stimuli_id_map=fish_data["stimuli_id_map"],
                available_stimuli=fish_data["trial_aligned_traces_z_core"].keys(),
            )
        except Exception as error:
            raise ValueError(
                f"Stimulus order {stim_order} is unavailable for {fish_id}. "
                f"Detected map: {fish_data['stimuli_id_map']}"
            ) from error

    reference_fish_id = fish_ids[0]
    reference_fish = all_fish_data[reference_fish_id]
    return {
        "settings": settings,
        "timing": timing,
        "fish_ids": fish_ids,
        "stim_order": stim_order,
        "main_path": main_path,
        "analysis_path": analysis_path,
        "all_fish_data": all_fish_data,
        "reference_fish_id": reference_fish_id,
        "reference_fish": reference_fish,
        "trial_aligned_traces": reference_fish["trial_aligned_traces_z_core"],
        "stimuli_id_map": reference_fish["stimuli_id_map"],
        "stimuli_durations": reference_fish["stimuli_durations"],
        "stimuli_names": reference_fish["stimuli_names"],
    }


def build_all_fish_raster_figure(
    all_fish_data,
    fish_ids,
    stim_order,
    reference_fish,
    timing,
    raster_settings,
    stimuli_colors,
    stimuli_linestyles,
    figsize=(8, 8),
):
    """Build and order a configured all-fish raster figure.

    ``raster_sort_mode='left_right_index'`` orders kept neurons using the AUC
    response-pair index between the configured left and right control IDs.
    """
    flat_matrix_all_fish = mfa.build_matrix_all_fish(
        all_fish_data=all_fish_data,
        stim_order=stim_order,
        fish_ids=fish_ids,
        combine_mode=raster_settings["combine_mode"],
        trace_type=raster_settings["trace_type"],
    )

    raster_neuron_order = None
    raster_sort_label = None
    raster_left_right_index = None
    if raster_settings["raster_sort_mode"] == "left_right_index":
        left_selection = at.resolve_selected_stimuli(
            [raster_settings["left_control"]],
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
        )
        right_selection = at.resolve_selected_stimuli(
            [raster_settings["right_control"]],
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
        )
        sort_response = mfa.build_zscore_response_matrices_all_fish(
            all_fish_data=all_fish_data,
            fish_ids=fish_ids,
            selected_stimuli=left_selection["stimulus_labels"] + right_selection["stimulus_labels"],
            fps_2p=timing["fps_2p"],
            t_pre_s=timing["t_pre_s"],
        )["pooled_response_matrix"]
        raster_left_right_index = at.compute_response_pair_index(
            pd.DataFrame(
                {
                    "left": sort_response[left_selection["stimulus_labels"][0]],
                    "right": sort_response[right_selection["stimulus_labels"][0]],
                }
            ),
            left_stimulus="left",
            right_stimulus="right",
        )
        sort_values = np.where(np.isfinite(raster_left_right_index), raster_left_right_index, np.inf)
        raster_neuron_order = np.argsort(sort_values, kind="stable")
        raster_sort_label = "left-right index"

    fig, ax, image, neuron_order = plott.plot_allfish_flat_raster(
        data=flat_matrix_all_fish,
        trial_aligned_traces=reference_fish["trial_aligned_traces_z_core"],
        stim_order=stim_order,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        stimuli_durations=reference_fish["stimuli_durations"],
        stimuli_colors=stimuli_colors,
        stimuli_linestyles=stimuli_linestyles,
        fps_2p=timing["fps_2p"],
        t_pre_s=timing["t_pre_s"],
        combine_mode=raster_settings["combine_mode"],
        sort_mode=("corravg" if raster_neuron_order is not None else raster_settings["raster_sort_mode"]),
        neuron_order=raster_neuron_order,
        sort_label=raster_sort_label,
        is_binary=False,
        show_mean_trace=True,
        figsize=figsize,
        fish_id=f"all_fish_{_slugify_label(raster_settings.get('analysis_label', raster_settings.get('experiment_name', 'analysis')))}",
    )
    return {
        "flat_matrix_all_fish": flat_matrix_all_fish,
        "raster_left_right_index": raster_left_right_index,
        "neuron_order": neuron_order,
        "figure": fig,
        "axes": ax,
        "image": image,
    }


def export_notebook_report(
    notebook_path,
    output_dir,
    report_name,
    report_format="pdf",
    fallback_to_html=True,
):
    """Export a saved notebook into the report folder using nbconvert."""
    notebook_path = Path(notebook_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not notebook_path.exists():
        return {
            "ok": False,
            "message": f"Notebook was not found: {notebook_path}",
            "paths": [],
        }

    formats = ["html", "webpdf"] if report_format == "both" else [report_format]
    exported_paths = []
    messages = []
    ok = True

    for fmt in formats:
        nbconvert_format = "webpdf" if fmt == "pdf" else fmt
        suffix = ".pdf" if nbconvert_format in {"pdf", "webpdf"} else f".{nbconvert_format}"
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            nbconvert_format,
            "--output",
            report_name,
            "--output-dir",
            str(output_dir),
            str(notebook_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        expected_path = output_dir / f"{report_name}{suffix}"
        if result.returncode == 0 and expected_path.exists():
            exported_paths.append(expected_path)
            messages.append(f"Exported {nbconvert_format}: {expected_path}")
        else:
            ok = False
            detail = result.stderr.strip() or result.stdout.strip()
            messages.append(f"Could not export {nbconvert_format}: {detail}")

    if not ok and fallback_to_html and report_format == "pdf":
        html_path = output_dir / f"{report_name}.html"
        if not html_path.exists():
            command = [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                "--output",
                report_name,
                "--output-dir",
                str(output_dir),
                str(notebook_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and html_path.exists():
                exported_paths.append(html_path)
                messages.append(f"Exported fallback html: {html_path}")
            else:
                detail = result.stderr.strip() or result.stdout.strip()
                messages.append(f"Could not export fallback html: {detail}")

    return {"ok": ok, "message": "\n".join(messages), "paths": exported_paths}


def save_analysis_report_run(
    settings,
    report_settings,
    analysis_path,
    tables=None,
    notebook_path=None,
    extra_metadata=None,
    report_name=None,
):
    """Save one timestamped analysis run record and an optional named notebook report."""
    if not report_settings.get("save_report", False):
        print("Report saving is off. Set REPORT_SETTINGS['save_report'] = True to save this run.")
        return None

    experiment_name = settings["experiment_name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_label = _slugify_label(report_settings.get("run_label", "run"))
    run_id = f"{timestamp}_{run_label}"

    report_root = Path(analysis_path) / experiment_name / "reports"
    run_dir = report_root / run_id
    tables_dir = run_dir / "tables"
    run_dir.mkdir(parents=True, exist_ok=False)
    tables_dir.mkdir(parents=True, exist_ok=True)

    comments = str(report_settings.get("comments", "")).strip()
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "fish_ids": list(settings.get("fish_ids", [])),
        "run_label": report_settings.get("run_label", "run"),
        "report_format": report_settings.get("report_format", "pdf"),
        "python": sys.version,
        "platform": platform.platform(),
        "notebook_path": str(notebook_path) if notebook_path is not None else None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with (run_dir / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(settings), handle, indent=2, ensure_ascii=False)
    with (run_dir / "report_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report_settings), handle, indent=2, ensure_ascii=False)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metadata), handle, indent=2, ensure_ascii=False)
    with (run_dir / "comments.md").open("w", encoding="utf-8") as handle:
        handle.write("# Comments\n\n")
        handle.write(comments + "\n" if comments else "_No comments were written for this run._\n")

    saved_tables = {}
    for name, table in (tables or {}).items():
        if table is None:
            continue
        table_path = tables_dir / f"{_slugify_label(name)}.csv"
        pd.DataFrame(table).to_csv(table_path, index=False)
        saved_tables[name] = table_path

    export_result = None
    if notebook_path is not None and report_settings.get("export_notebook", True):
        report_name = _slugify_label(report_name or "several_fish_report")
        export_result = export_notebook_report(
            notebook_path=notebook_path,
            output_dir=run_dir,
            report_name=report_name,
            report_format=report_settings.get("report_format", "pdf"),
            fallback_to_html=report_settings.get("fallback_to_html", True),
        )

    print(f"Saved analysis run folder: {run_dir}")
    if saved_tables:
        print("Saved tables:", ", ".join(saved_tables))
    if export_result is not None:
        print(export_result["message"])
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "tables": saved_tables,
        "export": export_result,
    }


def resolve_stimulus_set(selection, reference_fish, fallback_labels=None):
    """Resolve a user-selected stimulus list, using fallback labels when selection is None."""
    if selection is None:
        if fallback_labels is None:
            fallback_labels = reference_fish["stimuli_names"]
        selection = list(fallback_labels)
    return at.resolve_selected_stimuli(
        selection,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
    )


def build_response_window_validation(
    all_fish_data,
    fish_ids,
    stimulus_ids,
    stimulus_labels,
    fps_2p,
    t_pre_s,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """Build a response-window validation table for selected fish and stimuli."""
    rows = []
    for fid in fish_ids:
        fish = all_fish_data[fid]
        trial_aligned_z = fish["trial_aligned_traces_z_core"]
        for stim_id, stim_label in zip(stimulus_ids, stimulus_labels):
            stim_key = stim_id if stim_id in trial_aligned_z else str(stim_id)
            arr = trial_aligned_z[stim_key]
            window = at.compute_response_window_frames(
                n_time=arr.shape[1],
                fps_2p=fps_2p,
                t_pre_s=t_pre_s,
                motion_onset_s=motion_onset_s,
                stimulus=stim_id,
                stimuli_durations=fish["stimuli_durations"],
                stimuli_id_map=fish["stimuli_id_map"],
                tau_s=tau_s,
                motion_duration_key=motion_duration_key,
            )
            rows.append(
                {
                    "fish_id": fid,
                    "stimulus": stim_label,
                    "stimulus_id": stim_id,
                    "n_time": arr.shape[1],
                    "start_frame": window["start_frame"],
                    "stop_frame": window["stop_frame"],
                    "n_response_frames": window["n_frames"],
                    "start_s": window["start_s"],
                    "end_s": window["end_s"],
                }
            )
    return pd.DataFrame(rows)


def _resolve_response_control_column(control, stimulus_ids, stimulus_labels, response_columns):
    """Resolve a configured control ID/name to the response-matrix column label."""
    response_columns = list(response_columns)
    if control in response_columns:
        return control

    control_text = str(control)
    for column in response_columns:
        if str(column) == control_text:
            return column

    for stim_id, stim_label in zip(stimulus_ids, stimulus_labels):
        if control == stim_id or control_text == str(stim_id):
            if stim_label in response_columns:
                return stim_label
            if str(stim_label) in response_columns:
                return str(stim_label)

    available = ", ".join(str(column) for column in response_columns)
    raise KeyError(
        f"left_right_controls value {control!r} could not be resolved to a "
        f"response column. Available response columns: {available}"
    )


def resolve_response_control_columns(
    preference_controls,
    stimulus_ids,
    stimulus_labels,
    response_columns,
):
    """Resolve two left/right control IDs or names to response-matrix columns."""
    if preference_controls is None:
        return None
    if len(preference_controls) != 2:
        raise ValueError("preference_controls must contain exactly two values.")
    return tuple(
        _resolve_response_control_column(
            control=control,
            stimulus_ids=stimulus_ids,
            stimulus_labels=stimulus_labels,
            response_columns=response_columns,
        )
        for control in preference_controls
    )


def build_selected_neuron_summary(
    response_matrices,
    pooled_response_matrix,
    response_row_metadata,
    active_matrices,
    response_stimulus_ids,
    response_stimulus_labels,
    plot_stimulus_ids,
    plot_stimulus_labels,
    analysis_label="",
    preference_controls=None,
    filter_mode="none",
    filter_threshold=0.3,
    filter_range=(-0.3, 0.3),
):
    """Build summary tables, left/right preference values, and the plot keep mask."""
    summary_all = mfa.build_neuron_stimulus_summary_table(
        response_matrices=response_matrices,
        active_matrices=active_matrices,
        row_metadata=response_row_metadata,
        selected_stimulus_ids=response_stimulus_ids,
        selected_stimulus_labels=response_stimulus_labels,
        analysis_label=analysis_label,
    )
    summary_plot = mfa.build_neuron_stimulus_summary_table(
        response_matrices=response_matrices,
        active_matrices=active_matrices,
        row_metadata=response_row_metadata,
        selected_stimulus_ids=plot_stimulus_ids,
        selected_stimulus_labels=plot_stimulus_labels,
        analysis_label=analysis_label,
    )
    summary_plot = mfa.add_selectivity_metrics_to_summary_table(
        summary_plot,
        selected_stimulus_labels=plot_stimulus_labels,
    )

    resolved_preference_controls = resolve_response_control_columns(
        preference_controls=preference_controls,
        stimulus_ids=response_stimulus_ids,
        stimulus_labels=response_stimulus_labels,
        response_columns=pd.DataFrame(pooled_response_matrix).columns,
    )

    if resolved_preference_controls is None:
        preference_index = np.full(pooled_response_matrix.shape[0], np.nan, dtype=float)
    else:
        preference_index = at.compute_response_pair_index(
            pooled_response_matrix,
            left_stimulus=resolved_preference_controls[0],
            right_stimulus=resolved_preference_controls[1],
        )
    keep_mask = at.build_response_index_keep_mask(
        preference_index,
        mode=filter_mode,
        threshold=filter_threshold,
        value_range=filter_range,
    )

    for table in (summary_plot, summary_all):
        table["left_right_index"] = preference_index
        table["plot_neuron_keep"] = keep_mask

    filter_table = response_row_metadata.copy().reset_index(drop=True)
    filter_table["left_right_index"] = preference_index
    filter_table["plot_neuron_keep"] = keep_mask
    if resolved_preference_controls is not None:
        filter_table["left_right_left_control"] = resolved_preference_controls[0]
        filter_table["left_right_right_control"] = resolved_preference_controls[1]

    return {
        "neuron_summary_table": summary_plot,
        "neuron_summary_all_stimuli_table": summary_all,
        "left_right_index": preference_index,
        "plot_neuron_keep_mask": keep_mask,
        "plot_neuron_keep_indices": np.flatnonzero(keep_mask),
        "resolved_preference_controls": resolved_preference_controls,
        "selected_auc_response_matrix": pd.DataFrame(pooled_response_matrix).loc[
            keep_mask,
            list(plot_stimulus_labels),
        ].copy(),
        "plot_filter_table": filter_table,
    }


def build_high_sparseness_raster_data(
    all_fish_data,
    fish_ids,
    summary_table,
    stim_order,
    preferred_stimulus_order,
    lifetime_sparseness_threshold=0.65,
    keep_mask=None,
    combine_mode="mean",
    trace_type="zscore",
):
    """Prepare the matrix and row order for a high lifetime-sparseness raster."""
    metric_table = pd.DataFrame(summary_table).copy()
    if keep_mask is not None and not metric_table.empty:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        global_ids = metric_table["global_neuron_id"].to_numpy(dtype=int)
        if keep_mask.shape[0] <= global_ids.max(initial=-1):
            raise ValueError("keep_mask is shorter than summary_table global_neuron_id values.")
        metric_table = metric_table.loc[keep_mask[global_ids]].copy()

    response_matrix = mfa.build_matrix_all_fish(
        all_fish_data,
        stim_order,
        fish_ids=fish_ids,
        combine_mode=combine_mode,
        trace_type=trace_type,
    )
    preference_rank = {
        stimulus: rank
        for rank, stimulus in enumerate(preferred_stimulus_order)
    }
    selected_table = metric_table.loc[
        np.isfinite(metric_table["lifetime_sparseness"])
        & (metric_table["lifetime_sparseness"] > float(lifetime_sparseness_threshold))
    ].copy()
    selected_table["_preferred_stimulus_rank"] = (
        selected_table["preferred_stimulus"].map(preference_rank).fillna(len(preference_rank))
    )
    selected_table["_max_auc_sort"] = selected_table["max_response"].fillna(-np.inf)
    selected_table = selected_table.sort_values(
        by=["_preferred_stimulus_rank", "_max_auc_sort", "global_neuron_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    global_order = selected_table["global_neuron_id"].to_numpy(dtype=int)
    raster_matrix = response_matrix[global_order, :] if global_order.size else response_matrix[:0, :]
    return {
        "matrix": raster_matrix,
        "summary_table": selected_table,
        "global_neuron_order": global_order,
        "plot_neuron_order": np.arange(raster_matrix.shape[0], dtype=int),
    }


def plot_lifetime_sparseness_analysis(
    all_fish_data,
    fish_ids,
    reference_fish,
    timing,
    selected_stimuli,
    lifetime_sparseness_threshold,
    stimuli_colors,
    stimuli_linestyles,
    combine_mode="concat",
    sparseness_figsize=(5, 4),
    raster_figsize=(12, 8),
    analysis_label="",
):
    """Plot lifetime sparseness and a high-sparseness z-score raster only."""
    if combine_mode not in {"mean", "concat"}:
        raise ValueError("combine_mode must be 'mean' or 'concat'.")
    selection = at.resolve_selected_stimuli(
        selected_stimuli,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
    )
    response_results = mfa.build_zscore_response_matrices_all_fish(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        selected_stimuli=selection["stimulus_labels"],
        fps_2p=timing["fps_2p"],
        t_pre_s=timing["t_pre_s"],
    )
    response_matrix = response_results["pooled_response_matrix"]
    metric_rows = [
        mfa.compute_stimulus_selectivity_metrics(
            row.to_numpy(dtype=float),
            stimulus_labels=selection["stimulus_labels"],
        )
        for _, row in response_matrix.iterrows()
    ]
    summary_table = pd.concat(
        [
            response_results["row_metadata"].reset_index(drop=True),
            pd.DataFrame(metric_rows).reset_index(drop=True),
        ],
        axis=1,
    )

    fig_sparseness, ax_sparseness = plt.subplots(figsize=sparseness_figsize)
    plott.plot_stimulus_specificity_sparseness(
        summary_table,
        selected_stimulus_labels=selection["stimulus_labels"],
        analysis_label=analysis_label,
        ax=ax_sparseness,
    )
    plt.tight_layout()
    plt.show()

    high_sparseness = build_high_sparseness_raster_data(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        summary_table=summary_table,
        stim_order=selection["stimulus_ids"],
        preferred_stimulus_order=selection["stimulus_labels"],
        lifetime_sparseness_threshold=lifetime_sparseness_threshold,
        combine_mode=combine_mode,
        trace_type="zscore",
    )
    if high_sparseness["matrix"].shape[0] == 0:
        print("No neurons exceeded the selected lifetime-sparseness threshold.")
        high_sparseness_figure = None
    else:
        high_sparseness_figure, _, _, _ = plott.plot_allfish_flat_raster(
            data=high_sparseness["matrix"],
            trial_aligned_traces=reference_fish["trial_aligned_traces_z_core"],
            stim_order=selection["stimulus_ids"],
            stimuli_id_map=reference_fish["stimuli_id_map"],
            stimuli_durations=reference_fish["stimuli_durations"],
            stimuli_colors=stimuli_colors,
            stimuli_linestyles=stimuli_linestyles,
            fps_2p=timing["fps_2p"],
            t_pre_s=timing["t_pre_s"],
            combine_mode=combine_mode,
            sort_mode="unsorted",
            neuron_order=high_sparseness["plot_neuron_order"],
            sort_label=f"lifetime sparseness > {lifetime_sparseness_threshold}",
            is_binary=False,
            show_mean_trace=True,
            figsize=raster_figsize,
            fish_id=f"all_fish_{_slugify_label(analysis_label or 'analysis')}_high_lifetime_sparseness_zscore",
            vmax=4,
        )
        plt.show()

    return {
        "summary_table": summary_table,
        "selection": selection,
        "sparseness_figure": fig_sparseness,
        "high_sparseness": high_sparseness,
        "high_sparseness_figure": high_sparseness_figure,
    }


def build_pooled_mean_trace_by_stimulus(
    all_fish_data,
    fish_ids,
    stim_order,
    trace_type="zscore",
    use_kept_neurons=False,
    response_row_metadata=None,
    keep_mask=None,
):
    """Build a dict of stimulus ID to per-fish mean traces shaped (n_fish, n_time)."""
    trial_key_by_type = {
        "dfof": "trial_aligned_traces",
        "raster": "trial_aligned_traces_raster",
        "zscore": "trial_aligned_traces_z_core",
        "norm": "trial_aligned_traces_norm",
    }
    if trace_type not in trial_key_by_type:
        raise ValueError("trace_type must be 'dfof', 'raster', 'zscore', or 'norm'.")

    if keep_mask is not None:
        if response_row_metadata is None:
            raise ValueError("response_row_metadata is required when keep_mask is provided.")
        use_kept_neurons = True
        fish_keep_masks = build_fish_keep_masks(
            response_row_metadata=response_row_metadata,
            keep_mask=keep_mask,
            fish_ids=fish_ids,
        )
    else:
        fish_keep_masks = {fid: None for fid in fish_ids}

    trial_key = trial_key_by_type[trace_type]
    mean_traces = {stim: [] for stim in stim_order}
    for fid in fish_ids:
        fish = all_fish_data[fid]
        kept = np.asarray(fish.get("kept_neuron_indices", []), dtype=int)
        fish_keep_mask = fish_keep_masks[fid]
        for stim in stim_order:
            traces = fish[trial_key]
            key = stim if stim in traces else str(stim)
            arr = np.asarray(traces[key], dtype=float)
            if use_kept_neurons and kept.size:
                arr = arr[kept, :, :]
            if fish_keep_mask is not None:
                if arr.shape[0] != fish_keep_mask.shape[0]:
                    raise ValueError(
                        f"Trace rows for {fid}, stimulus {stim!r} do not match "
                        f"the filter rows: {arr.shape[0]} vs {fish_keep_mask.shape[0]}."
                    )
                arr = arr[fish_keep_mask, :, :]
            mean_by_neuron = np.nanmean(arr, axis=2)
            mean_traces[stim].append(np.nanmean(mean_by_neuron, axis=0))

    return {stim: np.vstack(rows) for stim, rows in mean_traces.items()}


def build_plot_all_fish_mean_zscore_traces(
    all_fish_data,
    fish_ids,
    stim_order,
    reference_fish,
    timing,
    stimuli_colors,
    stimuli_linestyles,
    figsize=(7, 4.5),
):
    """Plot the all-fish mean z-score traces for the selected stimulus order."""
    selection = at.resolve_selected_stimuli(
        stim_order,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_z_core"].keys(),
    )
    mean_traces = build_pooled_mean_trace_by_stimulus(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        stim_order=selection["stimulus_ids"],
        trace_type="zscore",
        use_kept_neurons=True,
    )
    fig, ax, colors_used, _ = plott.plot_stimulus_means(
        mean_traces=mean_traces,
        stimuli_ids=selection["stimulus_ids"],
        stimuli_names=selection["stimulus_labels"],
        title_prefix="",
        fps_2p=timing["fps_2p"],
        t_post_s=timing["t_post_s"],
        t_pre_s=timing["t_pre_s"],
        stimuli_durations=reference_fish["stimuli_durations"],
        plots_path=None,
        prefix=None,
        save=False,
        stimuli_colors=stimuli_colors,
        stimuli_linestyles=stimuli_linestyles,
        close_after=False,
        kept_cells=None,
        comment="all_stimuli",
        figsize=figsize,
    )
    return {
        "mean_traces": mean_traces,
        "stimulus_ids": selection["stimulus_ids"],
        "stimulus_labels": selection["stimulus_labels"],
        "figure": fig,
        "axes": ax,
        "colors_used": colors_used,
    }


def plot_left_right_active_overlap_diagnostics(
    all_fish_data,
    fish_ids,
    reference_fish,
    timing,
    side_stimuli,
    show_overlap=True,
    show_significant_raster=True,
    active_fraction_threshold=0.10,
    min_epoch_s=1.0,
    min_active_reps=2,
    expected_reps=4,
    require_expected_reps=True,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
    overlap_figsize=(5, 4),
    raster_figsize=(12, 7),
):
    """Plot separate overlap heatmaps and significant-raster diagnostics per side."""
    expected_sides = {"left", "right"}
    if set(side_stimuli) != expected_sides:
        raise ValueError("side_stimuli must contain exactly 'left' and 'right'.")

    selections = {
        side: at.resolve_selected_stimuli(
            stimuli,
            stimuli_id_map=reference_fish["stimuli_id_map"],
            available_stimuli=reference_fish["trial_aligned_traces_raster"].keys(),
        )
        for side, stimuli in side_stimuli.items()
    }
    side_stimulus_ids = {side: selection["stimulus_ids"] for side, selection in selections.items()}
    side_stimulus_labels = {side: selection["stimulus_labels"] for side, selection in selections.items()}
    active_stim_order = list(dict.fromkeys(side_stimulus_ids["left"] + side_stimulus_ids["right"]))
    active_matrices = mfa.build_active_neuron_matrices_all_fish(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        stim_order=active_stim_order,
        fps_2p=timing["fps_2p"],
        t_pre_s=timing["t_pre_s"],
        motion_onset_s=motion_onset_s,
        active_fraction_threshold=active_fraction_threshold,
        min_epoch_s=min_epoch_s,
        min_active_reps=min_active_reps,
        expected_reps=expected_reps,
        require_expected_reps=require_expected_reps,
        tau_s=tau_s,
        motion_duration_key=motion_duration_key,
    )

    results = {}
    for side in ("left", "right"):
        diagnostic_data = build_overlap_diagnostic_data(
            all_fish_data=all_fish_data,
            active_matrices=active_matrices,
            fish_ids=fish_ids,
            side_stimulus_ids=side_stimulus_ids,
            side_stimulus_labels=side_stimulus_labels,
            response_row_metadata=None,
            keep_mask=None,
            side_to_plot=side,
            aggregation="mean_per_fish",
            plot_mode="significant_raster",
            combine_mode="mean",
            sort_mode="decision_then_mean",
        )
        results[side] = diagnostic_data

        if show_overlap:
            fig, ax = plt.subplots(figsize=overlap_figsize)
            sns.heatmap(
                diagnostic_data["matrix_to_plot"],
                vmin=0,
                vmax=1,
                square=True,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                cbar_kws={"label": "Jaccard overlap"},
                ax=ax,
            )
            ax.set(title=f"{side.capitalize()} active-neuron overlap", xlabel="", ylabel="")
            plt.tight_layout()
            plt.show()

        if show_significant_raster:
            diagnostic = diagnostic_data["active_trace_diagnostic"]
            if diagnostic["trace_matrix"].shape[0] == 0:
                print(f"No active neurons passed the {side} settings.")
            else:
                plott.plot_active_trace_decision_diagnostic(
                    diagnostic,
                    fps_2p=timing["fps_2p"],
                    stimuli_durations=reference_fish["stimuli_durations"],
                    t_pre_s=timing["t_pre_s"],
                    sort_mode=None,
                    neuron_order=diagnostic_data["diagnostic_neuron_order"],
                    trace_cmap="Greys",
                    show_active_count_trace=True,
                    active_count_threshold=0.5,
                    active_count_ylabel="# Active neurons",
                    active_count_color="black",
                    active_count_height_ratio=1.8,
                    figsize=raster_figsize,
                )
                plt.show()

    return results


def plot_motion_active_neuron_counts(
    all_fish_data,
    fish_ids,
    reference_fish,
    timing,
    selected_stimuli,
    stimuli_colors,
    figsize=(10, 4.5),
    active_fraction_threshold=0.10,
    min_epoch_s=1.0,
    min_active_reps=2,
    expected_reps=4,
    require_expected_reps=True,
    motion_onset_s=8.0,
    tau_s=6.0,
    motion_duration_key="motion_sec",
):
    """Plot mean motion-active neuron counts with one point per fish."""
    selection = at.resolve_selected_stimuli(
        selected_stimuli,
        stimuli_id_map=reference_fish["stimuli_id_map"],
        available_stimuli=reference_fish["trial_aligned_traces_raster"].keys(),
    )
    active_matrices = mfa.build_active_neuron_matrices_all_fish(
        all_fish_data=all_fish_data,
        fish_ids=fish_ids,
        stim_order=selection["stimulus_ids"],
        fps_2p=timing["fps_2p"],
        t_pre_s=timing["t_pre_s"],
        motion_onset_s=motion_onset_s,
        active_fraction_threshold=active_fraction_threshold,
        min_epoch_s=min_epoch_s,
        min_active_reps=min_active_reps,
        expected_reps=expected_reps,
        require_expected_reps=require_expected_reps,
        tau_s=tau_s,
        motion_duration_key=motion_duration_key,
    )
    counts = pd.DataFrame(
        {
            label: [active_matrices[fish_id][stim_id].sum() for fish_id in fish_ids]
            for stim_id, label in zip(selection["stimulus_ids"], selection["stimulus_labels"])
        },
        index=fish_ids,
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(counts.shape[1])
    colors = [stimuli_colors.get(label, f"C{idx}") for idx, label in enumerate(counts.columns)]
    ax.bar(x, counts.mean(axis=0), color=colors, alpha=0.7)
    for offset, fish_id in zip(np.linspace(-0.12, 0.12, len(fish_ids)), fish_ids):
        ax.scatter(x + offset, counts.loc[fish_id], color="black", s=32, zorder=3)
    ax.set(
        xticks=x,
        xticklabels=counts.columns,
        ylabel="Active neurons per fish",
        title="Motion-period active neurons per stimulus",
    )
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()
    return {"selection": selection, "active_matrices": active_matrices, "counts": counts, "figure": fig, "axes": ax}


def build_fish_keep_masks(response_row_metadata, keep_mask, fish_ids):
    """Split a pooled neuron keep mask into per-fish masks in response-row order."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    metadata = pd.DataFrame(response_row_metadata).reset_index(drop=True)
    if keep_mask.shape[0] != metadata.shape[0]:
        raise ValueError("keep_mask length does not match response_row_metadata rows.")
    if "fish_id" not in metadata.columns:
        raise ValueError("response_row_metadata is missing fish_id.")

    fish_column = metadata["fish_id"].to_numpy()
    return {fid: keep_mask[fish_column == fid] for fid in fish_ids}


def build_filtered_trial_aligned_traces_for_fish(
    fish_data,
    stimulus_ids,
    trace_key="trial_aligned_traces_z_core",
    fish_keep_mask=None,
):
    """Return selected trial-aligned traces after preprocessing and optional plot filtering."""
    kept = np.asarray(fish_data.get("kept_neuron_indices", []), dtype=int)
    traces = fish_data[trace_key]
    selected = {}
    for stim_id in stimulus_ids:
        key = stim_id if stim_id in traces else str(stim_id)
        arr = np.asarray(traces[key])
        if kept.size:
            arr = arr[kept, :, :]
        if fish_keep_mask is not None:
            fish_keep_mask = np.asarray(fish_keep_mask, dtype=bool)
            if arr.shape[0] != fish_keep_mask.shape[0]:
                raise ValueError(
                    f"Trace rows for stimulus {stim_id!r} do not match the filter rows: "
                    f"{arr.shape[0]} vs {fish_keep_mask.shape[0]}."
                )
            arr = arr[fish_keep_mask, :, :]
        selected[stim_id] = arr
    return selected


def subset_active_matrices(active_matrices, fish_ids, stim_order):
    """Subset per-fish active matrices to selected stimulus columns."""
    subset = {}
    for fid in fish_ids:
        active_matrix = active_matrices[fid]
        column_keys = []
        for stim in stim_order:
            if stim in active_matrix.columns:
                column_keys.append(stim)
            elif str(stim) in active_matrix.columns:
                column_keys.append(str(stim))
            else:
                raise KeyError(f"Stimulus {stim!r} was not found in active_matrices for {fid}.")
        subset[fid] = active_matrix.loc[:, column_keys].copy()
    return subset


def filter_active_matrices_by_keep_mask(active_matrices, response_row_metadata, keep_mask, fish_ids):
    """Apply a pooled neuron keep mask back to each fish active matrix."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape[0] != response_row_metadata.shape[0]:
        raise ValueError("keep_mask length does not match response_row_metadata rows.")

    filtered = {}
    fish_column = response_row_metadata["fish_id"].to_numpy()
    for fid in fish_ids:
        fish_rows = fish_column == fid
        fish_keep_mask = keep_mask[fish_rows]
        active_matrix = active_matrices[fid]
        if active_matrix.shape[0] != fish_keep_mask.shape[0]:
            raise ValueError(
                f"Active matrix rows for {fid} do not match response metadata rows: "
                f"{active_matrix.shape[0]} vs {fish_keep_mask.shape[0]}."
            )
        filtered[fid] = active_matrix.loc[fish_keep_mask].reset_index(drop=True)
    return filtered


def build_diagnostic_trace_source_data(all_fish_data, fish_ids, trial_key, stim_order, active_matrices_source):
    """Align z-score diagnostic traces to active-matrix rows when needed."""
    if trial_key != "trial_aligned_traces_z_core":
        return all_fish_data

    trace_source_data = {}
    for fid in fish_ids:
        fish = all_fish_data[fid]
        active_row_count = active_matrices_source[fid].shape[0]
        kept_indices = np.asarray(fish.get("kept_neuron_indices", []), dtype=int)
        z_traces = {}
        for stim in stim_order:
            traces = fish[trial_key]
            key = stim if stim in traces else str(stim)
            arr = np.asarray(traces[key])
            if arr.shape[0] == active_row_count:
                z_traces[key] = arr
            elif kept_indices.shape[0] == active_row_count and kept_indices.max(initial=-1) < arr.shape[0]:
                z_traces[key] = arr[kept_indices, :, :]
            else:
                raise ValueError(
                    f"Cannot align z-score rows for {fid}, stimulus {stim!r}: "
                    f"trace rows={arr.shape[0]}, active rows={active_row_count}, "
                    f"kept indices={kept_indices.shape[0]}."
                )
        fish_copy = fish.copy()
        fish_copy[trial_key] = z_traces
        trace_source_data[fid] = fish_copy
    return trace_source_data


def apply_diagnostic_keep_mask(diagnostic, keep_mask):
    """Subset a pooled diagnostic result by pooled neuron keep mask."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    diagnostic = diagnostic.copy()
    diagnostic["trace_matrix"] = diagnostic["trace_matrix"][keep_mask, :]
    diagnostic["decision_matrix"] = diagnostic["decision_matrix"][keep_mask, :]
    diagnostic["row_metadata"] = diagnostic["row_metadata"].loc[keep_mask].reset_index(drop=True)
    return diagnostic


def build_decision_then_mean_sort_order(diagnostic):
    """Sort diagnostic rows by active-decision signature, then mean trace strength."""
    trace_matrix = np.asarray(diagnostic["trace_matrix"], dtype=float)
    decision_matrix = np.asarray(diagnostic["decision_matrix"], dtype=int)
    if trace_matrix.shape[0] == 0:
        return np.array([], dtype=int)
    weights = 2 ** np.arange(decision_matrix.shape[1] - 1, -1, -1)
    signatures = decision_matrix @ weights
    mean_strength = np.nanmean(trace_matrix, axis=1)
    mean_strength = np.nan_to_num(mean_strength, nan=-np.inf)
    return np.lexsort((-mean_strength, -signatures))


def build_metric_sort_order(row_metadata, metric_table, metric_column, side="left"):
    """Sort diagnostic rows by a metric stored in a neuron summary table."""
    if metric_column not in metric_table.columns:
        raise KeyError(f"metric_table is missing {metric_column!r}.")
    metric_source = metric_table[["fish_id", "neuron_id", metric_column]].copy()
    order_source = row_metadata.reset_index(names="_diagnostic_row").merge(
        metric_source,
        on=["fish_id", "neuron_id"],
        how="left",
        validate="one_to_one",
    )
    values = order_source[metric_column].to_numpy(dtype=float)
    original_rows = order_source["_diagnostic_row"].to_numpy(dtype=int)
    finite = np.isfinite(values)
    missing_rank = np.where(finite, 0, 1)
    if metric_column == "left_right_index" and side == "right":
        metric_rank = np.where(finite, values, np.inf)
    else:
        metric_rank = np.where(finite, -values, np.inf)
    return original_rows[np.lexsort((original_rows, metric_rank, missing_rank))]


def build_overlap_diagnostic_data(
    all_fish_data,
    active_matrices,
    fish_ids,
    side_stimulus_ids,
    side_stimulus_labels,
    response_row_metadata,
    keep_mask=None,
    side_to_plot="left",
    aggregation="mean_per_fish",
    plot_mode="significant_raster",
    combine_mode="mean",
    sort_mode="decision_then_mean",
    metric_table=None,
):
    """Build overlap matrices plus the pooled diagnostic data used by the notebook."""
    valid_plots = {"significant_raster", "z_score"}
    valid_aggregations = {"pooled", "mean_per_fish"}
    valid_sort_modes = {"decision_then_mean", "left_right_index", "lifetime_sparseness"}
    if plot_mode not in valid_plots:
        raise ValueError(f"plot_mode must be one of {sorted(valid_plots)}.")
    if aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {sorted(valid_aggregations)}.")
    if sort_mode not in valid_sort_modes:
        raise ValueError(f"sort_mode must be one of {sorted(valid_sort_modes)}.")
    if side_to_plot not in side_stimulus_ids:
        raise KeyError(f"side_to_plot {side_to_plot!r} was not found in side_stimulus_ids.")

    combined_stim_order = []
    for stim_order in side_stimulus_ids.values():
        combined_stim_order.extend(stim_order)
    active_overlap = subset_active_matrices(active_matrices, fish_ids, combined_stim_order)
    if keep_mask is not None:
        active_overlap = filter_active_matrices_by_keep_mask(
            active_overlap,
            response_row_metadata=response_row_metadata,
            keep_mask=keep_mask,
            fish_ids=fish_ids,
        )

    overlap_results = {"pooled": {}, "mean_per_fish": {}}
    for side, stim_order in side_stimulus_ids.items():
        side_overlap = mfa.build_active_neuron_overlap_matrices_all_fish(
            active_matrices=active_overlap,
            side_stimuli={side: stim_order},
            condition_labels=side_stimulus_labels[side],
        )
        overlap_results["pooled"][side] = side_overlap["pooled"][side]
        overlap_results["mean_per_fish"][side] = side_overlap["mean_per_fish"][side]

    diagnostic_stim_order = side_stimulus_ids[side_to_plot]
    diagnostic_trial_key = {
        "significant_raster": "trial_aligned_traces_raster",
        "z_score": "trial_aligned_traces_z_core",
    }[plot_mode]
    active_diagnostic = subset_active_matrices(active_matrices, fish_ids, diagnostic_stim_order)
    diagnostic_source = build_diagnostic_trace_source_data(
        all_fish_data,
        fish_ids=fish_ids,
        trial_key=diagnostic_trial_key,
        stim_order=diagnostic_stim_order,
        active_matrices_source=active_diagnostic,
    )
    sort_source = mfa.build_pooled_active_trace_diagnostic(
        all_fish_data=all_fish_data,
        active_matrices=active_diagnostic,
        fish_ids=fish_ids,
        stim_order=diagnostic_stim_order,
        combine_mode=combine_mode,
        trial_key="trial_aligned_traces_raster",
    )
    diagnostic = mfa.build_pooled_active_trace_diagnostic(
        all_fish_data=diagnostic_source,
        active_matrices=active_diagnostic,
        fish_ids=fish_ids,
        stim_order=diagnostic_stim_order,
        combine_mode=combine_mode,
        trial_key=diagnostic_trial_key,
    )

    if keep_mask is not None:
        diagnostic = apply_diagnostic_keep_mask(diagnostic, keep_mask)
        sort_source = apply_diagnostic_keep_mask(sort_source, keep_mask)

    if sort_mode == "decision_then_mean":
        neuron_order = build_decision_then_mean_sort_order(sort_source)
    else:
        if metric_table is None:
            raise ValueError("metric_table is required for metric-based diagnostic sorting.")
        metric_column = "left_right_index" if sort_mode == "left_right_index" else "lifetime_sparseness"
        neuron_order = build_metric_sort_order(
            diagnostic["row_metadata"],
            metric_table=metric_table,
            metric_column=metric_column,
            side=side_to_plot,
        )

    return {
        "overlap_results": overlap_results,
        "matrix_to_plot": overlap_results[aggregation][side_to_plot],
        "active_trace_diagnostic": diagnostic,
        "sort_source_diagnostic": sort_source,
        "diagnostic_neuron_order": neuron_order,
        "diagnostic_trial_key": diagnostic_trial_key,
        "diagnostic_stim_order": diagnostic_stim_order,
    }
