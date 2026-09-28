"""Presentation of fitted LME coefficients, comparisons and responses."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _plot_lme_fixed_effects(fixed_effects, include_intercept=False, figsize=None):
    coef_df = pd.DataFrame(fixed_effects).copy()
    if coef_df.empty:
        return None, None
    if not include_intercept:
        coef_df = coef_df[coef_df["term"] != "Intercept"].copy()
    if coef_df.empty:
        return None, None

    model_names = coef_df["model_name"].drop_duplicates().tolist()
    if figsize is None:
        figsize = (8, max(2.8, 2.3 * len(model_names)))
    fig, axes = plt.subplots(len(model_names), 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, model_name in zip(axes, model_names):
        model_df = coef_df[coef_df["model_name"] == model_name].copy()
        model_df = model_df.sort_values("coefficient")
        y = np.arange(model_df.shape[0])
        ax.axvline(0, color="0.35", linewidth=1, linestyle="--")
        xerr = None
        if {"ci_lower", "ci_upper"}.issubset(model_df.columns):
            lower = model_df["coefficient"] - model_df["ci_lower"]
            upper = model_df["ci_upper"] - model_df["coefficient"]
            if np.isfinite(lower).all() and np.isfinite(upper).all():
                xerr = np.vstack([lower.to_numpy(), upper.to_numpy()])
        ax.errorbar(
            model_df["coefficient"],
            y,
            xerr=xerr,
            fmt="o",
            color="0.15",
            ecolor="0.45",
            capsize=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(model_df["term"])
        ax.set_title(model_name)
        ax.set_xlabel("Fixed-effect coefficient")
        ax.grid(axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, axes


def _plot_lme_model_comparison(model_comparison, figsize=(7, 4)):
    comparison = pd.DataFrame(model_comparison).copy()
    if comparison.empty:
        return None, None
    comparison = comparison[comparison["status"] == "success"].copy()
    comparison = comparison.dropna(subset=["aic", "bic"], how="all")
    if comparison.empty:
        return None, None

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(comparison.shape[0])
    width = 0.38
    if "aic" in comparison.columns:
        ax.bar(x - width / 2, comparison["aic"], width=width, label="AIC")
    if "bic" in comparison.columns:
        ax.bar(x + width / 2, comparison["bic"], width=width, label="BIC")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["model_name"], rotation=35, ha="right")
    ax.set_ylabel("Information criterion")
    ax.set_title("Model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def _plot_lme_observed_vs_fitted(fit_results, response_table, figsize=None):
    if response_table is None:
        return None, None
    df = pd.DataFrame(response_table)
    successful = [
        (name, payload["result"])
        for name, payload in fit_results.items()
        if payload.get("status") == "success" and payload.get("result") is not None
    ]
    if not successful:
        return None, None

    if figsize is None:
        figsize = (5 * len(successful), 4)
    fig, axes = plt.subplots(1, len(successful), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    observed = df["response"].to_numpy(dtype=float)
    for ax, (model_name, result) in zip(axes, successful):
        fitted = np.asarray(result.fittedvalues, dtype=float)
        finite = np.isfinite(observed) & np.isfinite(fitted)
        ax.scatter(fitted[finite], observed[finite], s=10, alpha=0.25, linewidths=0)
        if np.any(finite):
            low = float(np.nanmin([fitted[finite].min(), observed[finite].min()]))
            high = float(np.nanmax([fitted[finite].max(), observed[finite].max()]))
            ax.plot([low, high], [low, high], color="0.35", linewidth=1, linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("Fitted response")
        ax.set_ylabel("Observed response")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, axes


def _plot_lme_response_distribution(response_table, figsize=(10, 4)):
    if response_table is None:
        return None, None
    df = pd.DataFrame(response_table).copy()
    required = {"stimulus_class", "position_id", "response"}
    if df.empty or not required.issubset(df.columns):
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, col, title in zip(
        axes,
        ["stimulus_class", "position_id"],
        ["Response by stimulus class", "Response by position"],
    ):
        labels = df[col].dropna().drop_duplicates().tolist()
        values = [df.loc[df[col] == label, "response"].dropna().to_numpy() for label in labels]
        positions = np.arange(1, len(labels) + 1)
        if values:
            ax.boxplot(values, positions=positions, showfliers=False)
        ax.axhline(0, color="0.35", linewidth=1, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Response")
    fig.tight_layout()
    return fig, axes


def plot_lme_model_outputs(results, response_table=None, include_intercept=False):
    """
    Plot fixed effects, model comparison, fitted values, and response summaries.
    """
    figures = {}
    fixed_fig, fixed_axes = _plot_lme_fixed_effects(
        results.get("fixed_effects"),
        include_intercept=include_intercept,
    )
    if fixed_fig is not None:
        figures["fixed_effects"] = (fixed_fig, fixed_axes)

    comparison_fig, comparison_ax = _plot_lme_model_comparison(
        results.get("model_comparison")
    )
    if comparison_fig is not None:
        figures["model_comparison"] = (comparison_fig, comparison_ax)

    fitted_fig, fitted_axes = _plot_lme_observed_vs_fitted(
        results.get("fit_results", {}),
        response_table,
    )
    if fitted_fig is not None:
        figures["observed_vs_fitted"] = (fitted_fig, fitted_axes)

    distribution_fig, distribution_axes = _plot_lme_response_distribution(response_table)
    if distribution_fig is not None:
        figures["response_distribution"] = (distribution_fig, distribution_axes)

    return figures
