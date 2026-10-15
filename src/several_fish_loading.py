"""Cohort loading and stimulus preflight for several-fish raster workflows."""

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path

import src.data_loading as exio
from src.trial_alignment import resolve_selected_stimuli


def _automatic_onedrive_candidates():
    """Yield OneDrive roots in environment, home, then legacy-layout order."""
    for variable in ("OneDriveCommercial", "OneDrive"):
        value = os.environ.get(variable)
        if value:
            yield Path(value)

    for data_base in (Path.home(), Path(r"D:\Alejandro\Data")):
        yield data_base / "OneDrive - Universite de Lausanne"
        yield data_base / "OneDrive - Université de Lausanne"
        yield from sorted(data_base.glob("OneDrive - Universit* de Lausanne"))


def _resolve_cohort_paths(data_base):
    """Find explicit or automatically discovered Lab data and analysis roots.

    Args:
        data_base (str, Path, or None): Parent of the OneDrive Lab folder. None
            searches OneDriveCommercial, OneDrive, the current user's university
            OneDrive folders, then those below D:\\Alejandro\\Data. Automatic
            discovery requires both Lab/Data/2p and Lab/Analysis directories;
            explicit paths retain their existing lookup and failure behavior.

    Returns:
        tuple[Path, Path]: Imaging data root and analysis root, in that order.
    """
    if data_base is None:
        checked_roots = set()
        attempted_roots = []
        for onedrive_root in _automatic_onedrive_candidates():
            key = onedrive_root.resolve()
            if key in checked_roots:
                continue
            checked_roots.add(key)
            attempted_roots.append(str(onedrive_root))
            main_path = onedrive_root / "Lab" / "Data" / "2p"
            analysis_path = onedrive_root / "Lab" / "Analysis"
            if main_path.is_dir() and analysis_path.is_dir():
                return main_path, analysis_path
        raise FileNotFoundError(
            "Could not find a OneDrive Lab folder containing both Lab/Data/2p "
            "and Lab/Analysis. Tried: " + "; ".join(attempted_roots)
            + ". Set settings['data_base'] to the parent of your university OneDrive folder."
        )

    data_base = Path(data_base)
    onedrive_candidates = [
        data_base / "OneDrive - Universite de Lausanne",
        *sorted(data_base.glob("OneDrive - Universit* de Lausanne")),
    ]
    onedrive_root = next((path for path in onedrive_candidates if path.exists()), None)
    if onedrive_root is None:
        raise FileNotFoundError(f"Could not find the OneDrive Lab folder below: {data_base}")

    main_path = onedrive_root / "Lab" / "Data" / "2p"
    analysis_path = onedrive_root / "Lab" / "Analysis"
    return main_path, analysis_path


def _load_configured_fish(settings, timing, fish_ids, main_path, analysis_path):
    """Load every configured fish quietly or report all load failures.

    Args:
        settings (dict): Experiment and selected-block configuration.
        timing (dict): Imaging frame rate and peri-stimulus window in seconds.
        fish_ids (list): Fish IDs in the requested cohort order.
        main_path (Path): Imaging data root.
        analysis_path (Path): Stimulus and analysis data root.

    Returns:
        dict: Fish ID to the aligned experiment bundle, in cohort order.
    """
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
    return all_fish_data


def _validate_stimulus_order(all_fish_data, stim_order):
    """Require every fish to contain the requested stimulus IDs in its trace bundle.

    Args:
        all_fish_data (dict): Aligned experiment bundles keyed by fish ID.
        stim_order (list): Requested stimulus IDs in plot order.
    """
    for fish_id, fish_data in all_fish_data.items():
        try:
            resolve_selected_stimuli(
                stim_order,
                stimuli_id_map=fish_data["stimuli_id_map"],
                available_stimuli=fish_data["trial_aligned_traces_z_core"].keys(),
            )
        except Exception as error:
            raise ValueError(
                f"Stimulus order {stim_order} is unavailable for {fish_id}. "
                f"Detected map: {fish_data['stimuli_id_map']}"
            ) from error


def load_and_preflight_fish_raster_inputs(settings):
    """Load a configured fish cohort and validate its raster stimulus order.

    Args:
        settings (dict): Experiment, data root, fish IDs, stimulus order, selected
            blocks, and timing configuration for the several-fish notebook.
            data_base is the parent of the university OneDrive folder; set it
            to None to discover the Lab roots automatically on this computer.

    Returns:
        dict: Copied settings and timing, ordered fish/stimulus IDs, resolved paths,
        all aligned fish bundles, and first-fish trace/stimulus metadata. Loader
        output is suppressed on success; unavailable fish or stimuli raise.
    """
    settings = dict(settings)
    timing = dict(settings["timing"])
    fish_ids = list(settings["fish_ids"])
    stim_order = list(settings["stim_order"])
    if not fish_ids:
        raise ValueError("settings['fish_ids'] must contain at least one fish.")
    if not stim_order:
        raise ValueError("settings['stim_order'] must contain at least one stimulus ID.")

    main_path, analysis_path = _resolve_cohort_paths(settings["data_base"])
    all_fish_data = _load_configured_fish(
        settings, timing, fish_ids, main_path, analysis_path)
    _validate_stimulus_order(all_fish_data, stim_order)

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
