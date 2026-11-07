"""Report serialization and notebook export for several-fish analysis."""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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


# Shared filename-label conversion for several-fish figures. Keep the historical name.
slugify_label = _slugify_label


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
