"""Helpers for Exp5 mixed-effects feature-decomposition analyses."""

import warnings

import numpy as np
import pandas as pd


REQUIRED_RESPONSE_COLUMNS = [
    "fish_id",
    "neuron_id",
    "fish_neuron_id",
    "stimulus_name",
    "response",
    "hemifield",
    "stimulus_class",
    "position_id",
    "motion_level",
    "is_bout_like",
    "is_full_bout",
    "is_rocking",
    "is_flicker",
]

REQUIRED_METADATA_COLUMNS = [
    "stimulus_name",
    "hemifield",
    "stimulus_class",
    "position_id",
    "motion_level",
    "is_bout_like",
    "is_full_bout",
    "is_rocking",
    "is_flicker",
]


def _metadata_to_frame(stimulus_metadata):
    if isinstance(stimulus_metadata, pd.DataFrame):
        metadata = stimulus_metadata.copy()
    elif isinstance(stimulus_metadata, dict):
        metadata = pd.DataFrame.from_dict(stimulus_metadata, orient="index")
        metadata.index.name = "stimulus_name"
        metadata = metadata.reset_index()
    else:
        metadata = pd.DataFrame(stimulus_metadata)

    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"stimulus_metadata is missing column(s): {missing}")

    metadata = metadata[REQUIRED_METADATA_COLUMNS].copy()
    if metadata["stimulus_name"].duplicated().any():
        duplicated = metadata.loc[
            metadata["stimulus_name"].duplicated(), "stimulus_name"
        ].tolist()
        raise ValueError(f"stimulus_metadata has duplicate stimuli: {duplicated}")

    for col in ["is_bout_like", "is_full_bout", "is_rocking", "is_flicker"]:
        metadata[col] = metadata[col].astype(int)

    return metadata


def build_lme_response_table(response_matrices_by_fish, stimulus_metadata):
    """
    Build one long row per fish, filtered neuron, and stimulus response.

    Parameters
    ----------
    response_matrices_by_fish : dict
        Mapping fish_id -> DataFrame shaped neurons x stimuli. Values should be
        the z-score AUC responses produced by
        ``build_zscore_response_matrices_all_fish``.
    stimulus_metadata : pandas.DataFrame, dict, or records
        Editable stimulus feature metadata containing one row per stimulus.

    Returns
    -------
    pandas.DataFrame
        Long-format response table with the required LME columns.
    """
    metadata = _metadata_to_frame(stimulus_metadata)
    metadata_by_name = metadata.set_index("stimulus_name", drop=False)

    rows = []
    for fish_id, response_matrix in response_matrices_by_fish.items():
        matrix = pd.DataFrame(response_matrix).copy()
        missing_metadata = [
            stimulus for stimulus in matrix.columns if stimulus not in metadata_by_name.index
        ]
        if missing_metadata:
            raise KeyError(
                f"Missing stimulus metadata for fish {fish_id!r}: {missing_metadata}"
            )

        for neuron_id, responses in matrix.iterrows():
            fish_neuron_id = f"{fish_id}__{neuron_id}"
            for stimulus_name, response in responses.items():
                meta = metadata_by_name.loc[stimulus_name]
                rows.append(
                    {
                        "fish_id": str(fish_id),
                        "neuron_id": int(neuron_id),
                        "fish_neuron_id": fish_neuron_id,
                        "stimulus_name": str(stimulus_name),
                        "response": float(response),
                        "hemifield": meta["hemifield"],
                        "stimulus_class": meta["stimulus_class"],
                        "position_id": meta["position_id"],
                        "motion_level": meta["motion_level"],
                        "is_bout_like": int(meta["is_bout_like"]),
                        "is_full_bout": int(meta["is_full_bout"]),
                        "is_rocking": int(meta["is_rocking"]),
                        "is_flicker": int(meta["is_flicker"]),
                    }
                )

    df = pd.DataFrame(rows, columns=REQUIRED_RESPONSE_COLUMNS)
    return df


def validate_lme_response_table(
    df,
    stimulus_metadata,
    expected_neurons_by_fish=None,
    expected_stimuli=None,
    response_range=(-100.0, 100.0),
    response_range_error=False,
    verbose=True,
):
    """
    Validate the LME response table and return compact summary tables.
    """
    df = pd.DataFrame(df).copy()
    metadata = _metadata_to_frame(stimulus_metadata)
    missing = [col for col in REQUIRED_RESPONSE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"response table is missing column(s): {missing}")

    if expected_stimuli is None:
        expected_stimuli = metadata["stimulus_name"].tolist()
    expected_stimuli = list(expected_stimuli)
    expected_stimulus_set = set(expected_stimuli)

    observed_stimulus_set = set(df["stimulus_name"].dropna().unique())
    missing_stimuli = sorted(expected_stimulus_set - observed_stimulus_set)
    extra_stimuli = sorted(observed_stimulus_set - expected_stimulus_set)
    if missing_stimuli or extra_stimuli:
        raise ValueError(
            "Stimulus mismatch in response table: "
            f"missing={missing_stimuli}, extra={extra_stimuli}"
        )

    duplicate_mask = df.duplicated(["fish_id", "neuron_id", "stimulus_name"])
    if duplicate_mask.any():
        raise ValueError(
            "Duplicated fish/neuron/stimulus rows found: "
            f"{int(duplicate_mask.sum())}"
        )

    response_values = pd.to_numeric(df["response"], errors="coerce")
    if response_values.isna().any():
        raise ValueError(f"Missing or non-numeric response values: {int(response_values.isna().sum())}")
    if not np.isfinite(response_values.to_numpy(dtype=float)).all():
        raise ValueError("Response values contain non-finite values.")

    out_of_range_summary = pd.DataFrame()
    if response_range is not None:
        lower, upper = response_range
        out_of_range = (response_values < lower) | (response_values > upper)
        if out_of_range.any():
            out_of_range_df = df.loc[out_of_range].copy()
            out_of_range_summary = (
                out_of_range_df.groupby("stimulus_name")["response"]
                .agg(["count", "min", "max"])
                .reset_index()
            )
            message = (
                f"Response values outside expected range {response_range}: "
                f"{int(out_of_range.sum())}. Inspect summaries before fitting."
            )
            if response_range_error:
                raise ValueError(message)
            warnings.warn(message)

    metadata_labels = metadata.set_index("stimulus_name")
    category_columns = [
        "hemifield",
        "stimulus_class",
        "position_id",
        "motion_level",
        "is_bout_like",
        "is_full_bout",
        "is_rocking",
        "is_flicker",
    ]
    merged = df.merge(
        metadata[["stimulus_name", *category_columns]],
        on="stimulus_name",
        how="left",
        suffixes=("", "_expected"),
    )
    for col in category_columns:
        mismatch = merged[col].astype(str) != merged[f"{col}_expected"].astype(str)
        if mismatch.any():
            bad_stimuli = sorted(merged.loc[mismatch, "stimulus_name"].unique().tolist())
            raise ValueError(f"Incorrect {col} labels for stimulus/stimuli: {bad_stimuli}")

    rows_per_fish = df.groupby("fish_id").size().rename("n_rows").reset_index()
    neurons_per_fish = (
        df.groupby("fish_id")["neuron_id"].nunique().rename("n_neurons").reset_index()
    )
    rows_per_stimulus = (
        df.groupby("stimulus_name").size().reindex(expected_stimuli).rename("n_rows").reset_index()
    )
    rows_per_neuron = (
        df.groupby(["fish_id", "neuron_id"]).size().rename("n_stimulus_rows").reset_index()
    )

    expected_n_stimuli = len(expected_stimuli)
    bad_neuron_rows = rows_per_neuron[
        rows_per_neuron["n_stimulus_rows"] != expected_n_stimuli
    ]
    if not bad_neuron_rows.empty:
        raise ValueError(
            "Each neuron must have exactly "
            f"{expected_n_stimuli} stimulus rows; failures={len(bad_neuron_rows)}"
        )

    if expected_neurons_by_fish is not None:
        expected_neurons_by_fish = {
            str(fish_id): int(n) for fish_id, n in expected_neurons_by_fish.items()
        }
        observed = dict(
            zip(neurons_per_fish["fish_id"].astype(str), neurons_per_fish["n_neurons"])
        )
        mismatches = {
            fish_id: {"expected": expected, "observed": observed.get(fish_id)}
            for fish_id, expected in expected_neurons_by_fish.items()
            if observed.get(fish_id) != expected
        }
        if mismatches:
            raise ValueError(f"Neuron count mismatch by fish: {mismatches}")

    category_counts = {
        col: df[col].value_counts(dropna=False).rename("n_rows").reset_index()
        for col in ["hemifield", "stimulus_class", "position_id", "motion_level"]
    }
    response_summary = response_values.describe().to_frame(name="response")

    summaries = {
        "rows_per_fish": rows_per_fish,
        "neurons_per_fish": neurons_per_fish,
        "rows_per_stimulus": rows_per_stimulus,
        "category_counts": category_counts,
        "response_summary": response_summary,
        "out_of_range_summary": out_of_range_summary,
    }

    if verbose:
        print("Rows per fish")
        print(rows_per_fish.to_string(index=False))
        print("\nNeurons per fish")
        print(neurons_per_fish.to_string(index=False))
        print("\nRows per stimulus")
        print(rows_per_stimulus.to_string(index=False))
        print("\nCategory counts")
        for col, table in category_counts.items():
            print(f"\n{col}")
            print(table.to_string(index=False))
        print("\nResponse descriptive statistics")
        print(response_summary.to_string())
        if not out_of_range_summary.empty:
            print(f"\nResponses outside expected range {response_range}")
            print(out_of_range_summary.to_string(index=False))

    return summaries


def _fit_one_mixedlm(df, spec, reml=False, method="lbfgs", maxiter=500):
    try:
        import statsmodels.formula.api as smf
        from patsy import dmatrices
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for mixed-effects model fitting. "
            "Install/update the social_filters environment before running this cell."
        ) from exc

    formula = spec["formula"]
    groups = spec["groups"]
    _, fixed_effect_matrix = dmatrices(formula, data=df, return_type="dataframe")
    design_rank = int(np.linalg.matrix_rank(fixed_effect_matrix.to_numpy(dtype=float)))
    n_columns = int(fixed_effect_matrix.shape[1])
    if design_rank < n_columns:
        columns = fixed_effect_matrix.columns.tolist()
        raise ValueError(
            "Fixed-effect design matrix is rank deficient "
            f"(rank {design_rank} for {n_columns} columns). "
            "Remove redundant predictors from this model formula. "
            "For the default Exp5 metadata, avoid combining position_id with "
            "is_full_bout/is_bout_like, and avoid combining motion_level with "
            "feature flags that are fully determined by motion categories. "
            f"Design columns: {columns}"
        )

    vc_formula = spec.get("vc_formula")
    model = smf.mixedlm(
        formula=formula,
        data=df,
        groups=df[groups],
        vc_formula=vc_formula,
    )
    return model.fit(reml=reml, method=method, maxiter=maxiter)


def fit_lme_models(df, model_specs, reml=False, method="lbfgs", maxiter=500):
    """
    Fit every editable model specification and continue after failures.
    """
    df = pd.DataFrame(df).copy()
    if isinstance(model_specs, dict):
        iterator = model_specs.items()
    else:
        iterator = [(spec["name"], spec) for spec in model_specs]

    results = {}
    for model_name, spec in iterator:
        model_name = str(model_name)
        spec = dict(spec)
        spec.setdefault("name", model_name)
        try:
            missing_keys = [key for key in ["formula", "groups"] if key not in spec]
            if missing_keys:
                raise ValueError(f"Model spec is missing required key(s): {missing_keys}")
            result = _fit_one_mixedlm(
                df,
                spec=spec,
                reml=reml,
                method=method,
                maxiter=maxiter,
            )
            results[model_name] = {
                "status": "success",
                "spec": spec,
                "result": result,
                "error": None,
            }
            if not getattr(result, "converged", True):
                warnings.warn(f"Model {model_name!r} finished but did not converge.")
        except Exception as exc:
            results[model_name] = {
                "status": "failed",
                "spec": spec,
                "result": None,
                "error": str(exc),
            }
            print(f"Model {model_name!r} failed: {exc}")

    return results


def _safe_result_attr(result, attr):
    try:
        return getattr(result, attr)
    except Exception:
        return np.nan


def _random_variance_table(model_name, result):
    rows = []
    cov_re = _safe_result_attr(result, "cov_re")
    if cov_re is not None and not (isinstance(cov_re, float) and np.isnan(cov_re)):
        cov_re_df = pd.DataFrame(cov_re)
        for row_label in cov_re_df.index:
            for col_label in cov_re_df.columns:
                rows.append(
                    {
                        "model_name": model_name,
                        "component": f"group:{row_label}:{col_label}",
                        "variance": cov_re_df.loc[row_label, col_label],
                    }
                )

    vcomp = _safe_result_attr(result, "vcomp")
    if vcomp is not None and not (isinstance(vcomp, float) and np.isnan(vcomp)):
        for idx, value in enumerate(np.asarray(vcomp).ravel()):
            rows.append(
                {
                    "model_name": model_name,
                    "component": f"variance_component_{idx}",
                    "variance": value,
                }
            )
    return rows


def summarize_lme_model_results(fit_results, df):
    """
    Convert mixed-model fit objects into tidy coefficient and comparison tables.
    """
    df = pd.DataFrame(df)
    coefficient_rows = []
    comparison_rows = []
    random_variance_rows = []

    n_fish = int(df["fish_id"].nunique()) if "fish_id" in df.columns else np.nan
    n_neurons = (
        int(df["fish_neuron_id"].nunique())
        if "fish_neuron_id" in df.columns
        else np.nan
    )

    for model_name, payload in fit_results.items():
        spec = payload.get("spec", {})
        result = payload.get("result")
        status = payload.get("status")
        error = payload.get("error")

        comparison_rows.append(
            {
                "model_name": model_name,
                "formula": spec.get("formula"),
                "groups": spec.get("groups"),
                "notes": spec.get("notes"),
                "status": status,
                "error": error,
                "aic": _safe_result_attr(result, "aic") if result is not None else np.nan,
                "bic": _safe_result_attr(result, "bic") if result is not None else np.nan,
                "log_likelihood": _safe_result_attr(result, "llf") if result is not None else np.nan,
                "n_observations": int(_safe_result_attr(result, "nobs")) if result is not None else len(df),
                "n_fish": n_fish,
                "n_neurons": n_neurons,
                "converged": bool(getattr(result, "converged", False)) if result is not None else False,
            }
        )

        if result is None:
            continue

        params = pd.Series(_safe_result_attr(result, "fe_params"))
        bse = pd.Series(_safe_result_attr(result, "bse_fe"))
        tvalues = pd.Series(_safe_result_attr(result, "tvalues")).reindex(params.index)
        pvalues = pd.Series(_safe_result_attr(result, "pvalues")).reindex(params.index)
        try:
            conf_int = result.conf_int().reindex(params.index)
        except Exception:
            conf_int = pd.DataFrame(index=params.index, columns=[0, 1], dtype=float)

        for term in params.index:
            coefficient_rows.append(
                {
                    "model_name": model_name,
                    "formula": spec.get("formula"),
                    "term": term,
                    "coefficient": params.get(term, np.nan),
                    "std_error": bse.get(term, np.nan),
                    "t_or_z": tvalues.get(term, np.nan),
                    "p_value": pvalues.get(term, np.nan),
                    "ci_lower": conf_int.loc[term, 0] if term in conf_int.index else np.nan,
                    "ci_upper": conf_int.loc[term, 1] if term in conf_int.index else np.nan,
                }
            )

        random_variance_rows.extend(_random_variance_table(model_name, result))

    return {
        "fit_results": fit_results,
        "fixed_effects": pd.DataFrame(coefficient_rows),
        "model_comparison": pd.DataFrame(comparison_rows),
        "random_effects": pd.DataFrame(random_variance_rows),
    }
