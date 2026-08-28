import ast
import os
import tempfile
import warnings
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skbio
import typer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ritme._decorators import helper_function, main_function
from ritme.evaluate_models import (
    _trial_simplicity_key,
    build_tuned_model_from_result,
    load_experiment_config,
)
from ritme.explain_features import explain_features
from ritme.feature_space.feature_provenance import (
    build_provenance_map,
    match_provenance,
    matched_otu_coverage,
    same_feature_space,
)
from ritme.find_best_model_config import (
    _load_phylogeny,
    _load_taxonomy,
    _process_phylogeny,
    _process_taxonomy,
)
from ritme.split_train_test import adaptive_k_folds
from ritme.tune_models import (
    DEFAULT_NN_CORN_MAX_LEVELS,
    TASK_METRICS,
    retrain_fixed_configs,
)

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("seaborn-v0_8-pastel")


@helper_function
def _parse_param_value(raw: str):
    """Recover the typed value from an MLflow param string. Values that are
    not Python literals (e.g. 'tax_class') are the string itself, so the
    fallback is data, not an error path."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


@main_function
def load_trial_records(path_to_logs: str) -> pd.DataFrame:
    """Load mlflow_logs.csv with params.* preserved as raw strings and
    metrics.* converted to numeric."""
    records = pd.read_csv(path_to_logs, dtype=str)
    metric_cols = [c for c in records.columns if c.startswith("metrics.")]
    records[metric_cols] = records[metric_cols].apply(pd.to_numeric, errors="coerce")
    return records


@main_function
def reconstruct_trial_config(row: pd.Series) -> dict:
    """Rebuild a trial's typed config dict from its params.* cells.

    MLflow does not log a ``None``-valued param, so an empty CSV cell is
    indistinguishable from a key that was never set. ``data_*`` keys are
    therefore always emitted (``None`` when the cell is empty), since the
    trainables require them; other ``params.*`` cells are kept only when
    non-null.
    """
    config = {
        col[len("params.") :]: (_parse_param_value(value) if pd.notna(value) else None)
        for col, value in row.items()
        if col.startswith("params.data_")
    }
    config.update(
        {
            col[len("params.") :]: _parse_param_value(value)
            for col, value in row.items()
            if col.startswith("params.")
            and not col.startswith("params.data_")
            and pd.notna(value)
        }
    )
    config["mlflow_run_id"] = row["run_id"]
    return config


@main_function
def filter_reproducible_trials(records: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Keep only completed, full-K-fold trials with finite mean and SE.

    A single-split run (``k_folds=1``) never logs per-fold columns at all —
    raise the intended guard clause here rather than a bare ``KeyError``.
    """
    required = {f"metrics.{metric}_mean", f"metrics.{metric}_se", "metrics.n_folds"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(
            f"Trial records are missing required column(s) {sorted(missing)}. "
            "This is typically a single-split run (k_folds=1), which logs no "
            "per-fold SE and cannot be used for stability analysis."
        )
    full_k = records["metrics.n_folds"].max()
    mask = (
        (records["status"] == "FINISHED")
        & np.isfinite(records[f"metrics.{metric}_mean"])
        & np.isfinite(records[f"metrics.{metric}_se"])
        & (records["metrics.n_folds"] == full_k)
    )
    return records[mask].copy()


@main_function
def select_band_trials(
    records: pd.DataFrame,
    model_type: str,
    metric: str,
    mode: str,
    band_se_factor: float = 1.0,
    max_trials: int = 15,
    anchor_run_id: str = None,
    symmetric: bool = False,
) -> pd.DataFrame:
    """Select trials whose mean CV score lies within ``band_se_factor`` SE of
    an anchor trial's mean, within the requested model-type scope ('all' =
    every type).

    By default (``anchor_run_id=None``, ``symmetric=False``) the anchor is
    the single best-performing trial and the test is the classic one-sided
    1-SE rule (``mean - anchor_mean <= factor * anchor_se``) — it only
    bounds how much *worse* a trial may be than the anchor, never how much
    better. This is the candidate pool used to choose the deployed
    reference model, matching ``evaluate_models._select_best_with_one_se``.

    Pass a specific ``anchor_run_id`` (typically the reference itself) with
    ``symmetric=True`` to instead select trials whose performance is
    indistinguishable from that trial in *either* direction
    (``|mean - anchor_mean| <= factor * anchor_se``) — the set that is
    retrained and plotted.
    """
    if model_type != "all":
        available = sorted(records["experiment_name"].unique())
        if model_type not in available:
            raise ValueError(
                f"Model type {model_type!r} not found in trial records. "
                f"Available: {available} (or 'all')."
            )
        records = records[records["experiment_name"] == model_type]

    reproducible = filter_reproducible_trials(records, metric)
    if reproducible.empty:
        raise ValueError(
            f"No completed K-fold trial with finite {metric}_mean/_se found. "
            "Stability analysis requires a K-fold run (k_folds > 1)."
        )

    sign = 1 if mode == "min" else -1
    means = sign * reproducible[f"metrics.{metric}_mean"]
    if anchor_run_id is None:
        anchor_idx = means.idxmin()
    else:
        anchor_matches = reproducible.index[reproducible["run_id"] == anchor_run_id]
        if len(anchor_matches) == 0:
            raise ValueError(
                f"Anchor trial {anchor_run_id!r} not found among the "
                "reproducible trials in this scope."
            )
        anchor_idx = anchor_matches[0]
    anchor_mean = means.loc[anchor_idx]
    anchor_se = reproducible.loc[anchor_idx, f"metrics.{metric}_se"]

    deviation = means - anchor_mean
    if symmetric:
        deviation = deviation.abs()
    band = reproducible[deviation <= band_se_factor * anchor_se]
    band = band.sort_values(f"metrics.{metric}_mean", ascending=(mode == "min"))
    if len(band) > max_trials:
        warnings.warn(
            f"Band holds {len(band)} trials; keeping the {max_trials} best "
            f"(raise max_trials to include more)."
        )
        band = band.head(max_trials)
    return band.reset_index(drop=True)


@helper_function
def _one_se_winner_run_id(band_of_type: pd.DataFrame, metric: str, mode: str) -> str:
    sign = 1 if mode == "min" else -1
    means = sign * band_of_type[f"metrics.{metric}_mean"]
    best_mean = means.min()
    best_se = band_of_type.loc[means.idxmin(), f"metrics.{metric}_se"]
    in_band = band_of_type[means - best_mean <= best_se]
    keys = {
        row["run_id"]: _trial_simplicity_key(
            row["experiment_name"],
            {"nb_features": row["metrics.nb_features"]},
            reconstruct_trial_config(row),
        )
        for _, row in in_band.iterrows()
    }
    return min(keys, key=keys.get)


@main_function
def select_reference_run_id(band: pd.DataFrame, metric: str, mode: str) -> str:
    """Applies ritme's deployment rule (per model type the 1-SE-rule winner,
    then the winner with the best mean across types) to the candidate band.

    Can differ from the trial ``find_best_model_config`` deployed when
    ``max_trials`` truncated the candidate band: truncation keeps the best
    performers, while the 1-SE rule picks the simplest trial.
    """
    sign = 1 if mode == "min" else -1
    winners = {
        _one_se_winner_run_id(group, metric, mode): group
        for _, group in band.groupby("experiment_name")
    }
    winner_means = {
        run_id: sign
        * float(group.loc[group["run_id"] == run_id, f"metrics.{metric}_mean"].iloc[0])
        for run_id, group in winners.items()
    }
    return min(winner_means, key=winner_means.get)


@main_function
def compute_normalized_importance(
    tmodel,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    max_background_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Per-feature importance and rank for one trial, using the trial's
    native method (coefficients or mean absolute SHAP).

    Returns columns ``feature``, ``importance`` and ``rank`` (rank 1 = most
    important, ties share the lowest rank). ``importance`` holds raw
    magnitudes on the model's own scale and is **not** comparable across
    trials — only ``rank`` is used for cross-trial comparison.
    """
    result = explain_features(
        tmodel, train_val, test, max_background_samples=max_background_samples
    )
    if isinstance(result, pd.DataFrame):
        importance = result.groupby("feature", sort=False)["abs_coefficient"].mean()
    else:
        values = np.abs(result.values)
        if values.ndim == 3:
            values = values.mean(axis=2)
        importance = pd.Series(values.mean(axis=0), index=result.feature_names)

    table = importance.rename("importance").rename_axis("feature").reset_index()
    table["rank"] = table["importance"].rank(ascending=False, method="min").astype(int)
    return table


@main_function
def align_feature_ranks(
    reference_run_id: str,
    importances: Dict[str, pd.DataFrame],
    provenances: Dict[str, Dict],
    configs: Dict[str, dict],
    top_n: int = 15,
) -> pd.DataFrame:
    """Locate the reference trial's top-N features in every band trial and
    report the matched importance rank plus the match kind.

    ``trac`` band trials are always reported ``not_attributable``: they
    share no feature namespace with the other trials.

    ``split`` matches report the minimum (best) rank among the matched
    parts, making ``rank`` an optimistic upper bound. ``n_matched_parts``
    and ``otu_coverage`` record how many parts contributed and what
    fraction of the feature's source OTUs they account for; coverage below
    1.0 means the remainder sits in the trial's lumped bucket and is not
    represented in the rank.
    """
    reference = importances[reference_run_id].nsmallest(top_n, "rank")
    reference_prov = provenances[reference_run_id]
    reference_config = configs[reference_run_id]

    rows = []
    for run_id, trial_imp in importances.items():
        is_trac_trial = configs[run_id].get("model") == "trac"
        trial_prov = provenances[run_id]
        identical_space = same_feature_space(reference_config, configs[run_id])
        trial_ranks = trial_imp.set_index("feature")["rank"]
        for _, ref_row in reference.iterrows():
            if is_trac_trial:
                match_kind = "not_attributable"
                matched_columns, matched_rank = (), np.nan
                coverage = np.nan
            else:
                target = reference_prov[ref_row["feature"]]
                match = match_provenance(target, trial_prov, identical_space)
                matched_columns = match.columns
                matched_rank = (
                    float(trial_ranks.loc[list(matched_columns)].min())
                    if matched_columns
                    else np.nan
                )
                match_kind = match.kind
                coverage = matched_otu_coverage(target, trial_prov, matched_columns)
            rows.append(
                {
                    "feature": ref_row["feature"],
                    "reference_rank": int(ref_row["rank"]),
                    "run_id": run_id,
                    "match_kind": match_kind,
                    "matched_column": ";".join(matched_columns),
                    "n_matched_parts": len(matched_columns),
                    "otu_coverage": coverage,
                    "rank": matched_rank,
                }
            )
    return pd.DataFrame.from_records(rows)


@main_function
def compute_rank_agreement(ranks: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Per band trial: the fraction of the reference's top-N features that
    are also within that trial's own top-N.

    ``not_attributable`` reference features (e.g. against a ``trac`` or ILR
    trial) are excluded from the containment denominator — comparison is
    undefined for them, not failed, so they must not read as disagreement.

    Returns one row per ``run_id`` with column ``top_n_containment``, which
    is NaN when no reference feature is comparable at all.
    """
    rows = []
    for run_id, group in ranks.groupby("run_id"):
        comparable = group[group["match_kind"] != "not_attributable"]
        containment = (
            float((comparable["rank"] <= top_n).sum()) / len(comparable)
            if len(comparable)
            else np.nan
        )
        rows.append({"run_id": run_id, "top_n_containment": containment})
    return pd.DataFrame.from_records(rows)


_MATCH_ANNOTATIONS = {"contained": "◆", "lumped": "L", "split": "s"}
_METRIC_LABELS = {"rmse_val": "RMSE val", "roc_auc_macro_ovr_val": "ROC AUC val"}

# One size for every text element in the figure -- axis labels, tick labels,
# heatmap cells, colorbar and legends -- so no element reads as more
# important than another. Only the title steps up.
_BASE_FONTSIZE = 13
_TITLE_FONTSIZE = 16
# Mean glyph advance of DejaVu Sans, in units of the font size.
_AVG_CHAR_WIDTH_EM = 0.55
_CELL_LEGEND_ENTRIES = (
    "N = rank in that trial",
    "◆ = contained in a coarser taxon",
    "L = lumped (low-abundance/variance)",
    "s = split into finer parts",
    "– = feature absent",
    "grey n/a = not comparable (e.g. ILR/trac)",
)


@helper_function
def _legend_max_chars(width_inches: float) -> int:
    """Characters that fit on one legend line spanning ``width_inches``."""
    per_char = _AVG_CHAR_WIDTH_EM * _BASE_FONTSIZE / 72
    return max(20, int(width_inches / per_char))


@helper_function
def _pack_legend_lines(entries: tuple, max_chars: int) -> list:
    """Greedily pack legend entries into lines of at most ``max_chars``,
    except an entry longer than that, which gets its own over-wide line
    rather than being broken mid-symbol."""
    lines, current = [], ""
    for entry in entries:
        candidate = f"{current}   {entry}" if current else entry
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = entry
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


@helper_function
def _rank_cell_text(row: pd.Series) -> str:
    if row["match_kind"] == "not_attributable":
        return "n/a"
    if row["match_kind"] == "absent":
        return "–"
    suffix = _MATCH_ANNOTATIONS.get(row["match_kind"], "")
    return f"{int(row['rank'])}{suffix}"


@main_function
def plot_stability(
    manifest: pd.DataFrame,
    ranks: pd.DataFrame,
    metric: str,
    top_n: int = 15,
    show: bool = True,
) -> Optional[Figure]:
    """Two-panel stability figure: per-trial performance (mean ± SE) and the
    rank heatmap of the reference trial's top features across band trials.

    The figure title names the band's model type, so trial columns carry
    only their feature-engineering configuration; a band spanning several
    model types keeps the model name on the column label.

    The reference (deployed) trial is displayed first, then the rest in
    best -> worst order. The performance panel marks the reference with a
    square and the best performer with a star (omitted when they
    coincide), and shades the reference's own 1-SE interval.

    ``top_n`` only sets the colour-scale clamp at ``2 * top_n``; which
    feature rows are drawn comes entirely from ``ranks``.
    """
    reference_run_id = manifest.loc[manifest["is_reference"], "run_id"].iloc[0]
    best_run_id = manifest.loc[manifest["is_best"], "run_id"].iloc[0]
    rest = manifest.loc[manifest["run_id"] != reference_run_id, "run_id"].tolist()
    trial_order = [reference_run_id] + rest
    manifest = manifest.set_index("run_id").loc[trial_order].reset_index()
    model_types = manifest["experiment_name"].unique().tolist()

    features = (
        ranks[ranks["run_id"] == reference_run_id]
        .sort_values("reference_rank")["feature"]
        .tolist()
    )
    n_rows, n_cols = len(features), len(trial_order)

    fig, (ax_perf, ax_ranks) = plt.subplots(
        2,
        1,
        figsize=(max(6, 1.6 * n_cols + 3), 5 + 0.5 * n_rows),
        gridspec_kw={"height_ratios": [1, max(2, 0.25 * n_rows)]},
        sharex=True,
    )

    positions = np.arange(n_cols)
    ax_perf.errorbar(
        positions,
        manifest["metric_mean_logged"],
        yerr=manifest["metric_se_logged"],
        fmt="o",
        capsize=4,
        label="Trial mean ± SE",
    )
    reference_pos = 0
    best_pos = trial_order.index(best_run_id)
    ax_perf.scatter(
        [reference_pos],
        [manifest["metric_mean_logged"].iloc[reference_pos]],
        marker="s",
        s=90,
        zorder=3,
    )
    if best_pos != reference_pos:
        ax_perf.scatter(
            [best_pos],
            [manifest["metric_mean_logged"].iloc[best_pos]],
            marker="*",
            s=160,
            zorder=3,
        )
    reference_row = manifest.iloc[reference_pos]
    reference_mean = reference_row["metric_mean_logged"]
    reference_se = reference_row["metric_se_logged"]
    ax_perf.axhspan(
        reference_mean - reference_se,
        reference_mean + reference_se,
        alpha=0.15,
    )
    ax_perf.set_ylabel(_METRIC_LABELS.get(metric, metric), fontsize=_BASE_FONTSIZE)
    ax_perf.tick_params(axis="y", labelsize=_BASE_FONTSIZE)
    ax_perf.set_xticks(positions)
    ax_perf.set_xticklabels([])
    ax_perf.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=_BASE_FONTSIZE,
        frameon=False,
    )

    # Ranks worse than the cap are clamped for colouring only; the cell
    # text always prints the true rank.
    rank_cap = 2 * top_n
    matrix = np.full((n_rows, n_cols), np.nan)
    pivot = ranks.set_index(["feature", "run_id"])
    capped_any = False
    for i, feature in enumerate(features):
        for j, run_id in enumerate(trial_order):
            row = pivot.loc[(feature, run_id)]
            if pd.notna(row["rank"]):
                capped_any = capped_any or row["rank"] > rank_cap
                matrix[i, j] = min(row["rank"], rank_cap)
    im = ax_ranks.imshow(matrix, cmap="viridis_r", aspect="auto")
    cbar = fig.colorbar(im, ax=ax_ranks, fraction=0.03, pad=0.02)
    cbar.set_label("rank", fontsize=_BASE_FONTSIZE)
    cbar.ax.tick_params(labelsize=_BASE_FONTSIZE)
    if capped_any:
        # the top of the scale is a censored value, not an observed one
        ticks = [t for t in cbar.get_ticks() if im.norm.vmin <= t < rank_cap]
        cbar.set_ticks(
            ticks + [rank_cap],
            labels=[f"{t:g}" for t in ticks] + [f"≥{rank_cap}"],
        )
    for i, feature in enumerate(features):
        for j, run_id in enumerate(trial_order):
            row = pivot.loc[(feature, run_id)]
            if row["match_kind"] == "not_attributable":
                ax_ranks.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="grey", edgecolor="none"
                    )
                )
            ax_ranks.text(
                j,
                i,
                _rank_cell_text(row),
                ha="center",
                va="center",
                fontsize=_BASE_FONTSIZE,
            )
    ax_ranks.set_yticks(range(n_rows))
    ax_ranks.set_yticklabels(features, fontsize=_BASE_FONTSIZE)
    ax_ranks.set_xticks(range(n_cols))
    labels = []
    for j in range(n_cols):
        # config_summary is aggregation|selection|transform|enrich -- two
        # feature-engineering steps per line keeps the column narrow.
        steps = str(manifest["config_summary"].iloc[j]).split("|")
        lines = ["|".join(steps[:2]), "|".join(steps[2:])]
        if len(model_types) > 1:
            lines.insert(0, str(manifest["experiment_name"].iloc[j]))
        labels.append("\n".join(lines))
    ax_ranks.set_xticklabels(labels, fontsize=_BASE_FONTSIZE, rotation=30, ha="right")
    ax_ranks.set_xlabel("Trial", fontsize=_BASE_FONTSIZE)

    fig.suptitle(
        f"Feature stability among top-performing {' / '.join(model_types)} " "trials",
        fontsize=_TITLE_FONTSIZE,
    )

    # The heatmap's width is known only after tight_layout: lay out once to
    # measure it, wrap the legend, then lay out again with the room it needs.
    plt.tight_layout(rect=(0, 0.05, 1, 0.94))
    ranks_pos = ax_ranks.get_position()
    legend_lines = _pack_legend_lines(
        _CELL_LEGEND_ENTRIES,
        _legend_max_chars(ranks_pos.width * fig.get_figwidth()),
    )
    legend_inches = (len(legend_lines) + 1) * 1.4 * _BASE_FONTSIZE / 72
    plt.tight_layout(rect=(0, min(0.3, legend_inches / fig.get_figheight()), 1, 0.94))
    # The colorbar shrinks ax_ranks's own width to make room for itself;
    # match ax_perf's width/left edge to ax_ranks's (post-colorbar) so both
    # panels' data lines up with the heatmap's column centers below.
    ranks_pos = ax_ranks.get_position()
    perf_pos = ax_perf.get_position()
    ax_perf.set_position([ranks_pos.x0, perf_pos.y0, ranks_pos.width, perf_pos.height])
    fig.text(
        ranks_pos.x0,
        0.01,
        "\n".join(legend_lines),
        fontsize=_BASE_FONTSIZE,
        ha="left",
        va="bottom",
    )
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig


@helper_function
def _config_summary(config: dict) -> str:
    parts = [
        config.get("data_aggregation"),
        config.get("data_selection"),
        config.get("data_transform"),
        config.get("data_enrich"),
    ]
    return "|".join("none" if p is None else str(p) for p in parts)


@helper_function
def _band_manifest(
    band: pd.DataFrame,
    reference_run_id: str,
    metric: str,
    retrained_means: Dict[str, float],
) -> pd.DataFrame:
    # band arrives sorted best -> worst, so the first surviving row is the
    # best performer -- not necessarily the reference.
    best_run_id = band["run_id"].iloc[0]
    manifest = pd.DataFrame(
        {
            "run_id": band["run_id"].values,
            "experiment_name": band["experiment_name"].values,
            "is_reference": band["run_id"].values == reference_run_id,
            "is_best": band["run_id"].values == best_run_id,
            "metric_mean_logged": band[f"metrics.{metric}_mean"].values,
            "metric_se_logged": band[f"metrics.{metric}_se"].values,
            "nb_features": band["metrics.nb_features"].values,
        }
    )
    manifest["metric_mean_retrained"] = manifest["run_id"].map(retrained_means)
    manifest["metric_dev"] = (
        manifest["metric_mean_retrained"] - manifest["metric_mean_logged"]
    )
    manifest["config_summary"] = [
        _config_summary(reconstruct_trial_config(row)) for _, row in band.iterrows()
    ]
    return manifest


@main_function
def explain_stability(
    exp_config: dict,
    trial_records: pd.DataFrame,
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    model_type: str = "all",
    tax: pd.DataFrame = None,
    tree_phylo: skbio.TreeNode = None,
    band_se_factor: float = 1.0,
    max_trials: int = 15,
    top_n: int = 15,
    max_background_samples: Optional[int] = None,
    max_concurrent_trials: int = 2,
) -> tuple:
    """Select the Rashomon band of comparably-performing trials, retrain it
    deterministically, and report how stable the reference trial's top
    features are across the band.

    Returns ``(manifest, importances_long, ranks, agreement, figure)``.
    """
    metric, mode = TASK_METRICS[exp_config.get("task_type", "regression")]
    # Candidate pool for picking the reference, mirroring
    # evaluate_models._select_best_with_one_se.
    candidate_band = select_band_trials(
        trial_records,
        model_type,
        metric,
        mode,
        band_se_factor=band_se_factor,
        max_trials=max_trials,
    )
    reference_run_id = select_reference_run_id(candidate_band, metric, mode)

    reference_model = candidate_band.loc[
        candidate_band["run_id"] == reference_run_id, "experiment_name"
    ].iloc[0]
    if reference_model == "trac":
        raise ValueError(
            "Reference trial is 'trac'; trac's log-contrast coefficients are "
            "labeled by clade, not by design-matrix column, so it cannot "
            "serve as the reference for cross-trial feature-rank alignment. "
            "Scope model_type to a non-trac type."
        )

    # The band that is retrained and plotted is anchored on the reference,
    # so every displayed trial falls inside the figure's shaded band.
    band = select_band_trials(
        trial_records,
        model_type,
        metric,
        mode,
        band_se_factor=band_se_factor,
        max_trials=max_trials,
        anchor_run_id=reference_run_id,
        symmetric=True,
    )

    if "trac" in band["experiment_name"].values and (tree_phylo is None or tax is None):
        raise ValueError(
            "Band includes a 'trac' trial; retraining it requires both the "
            "phylogenetic tree (tree_phylo) and the taxonomy table (tax), "
            f"but tree_phylo={'set' if tree_phylo is not None else 'None'} "
            f"and tax={'set' if tax is not None else 'None'}."
        )

    trial_configs = [reconstruct_trial_config(row) for _, row in band.iterrows()]
    if tax is None and any(
        c.get("data_aggregation") is not None for c in trial_configs
    ):
        raise ValueError(
            "Band includes a trial using taxonomic aggregation "
            "(data_aggregation) but no tax was provided; retraining such a "
            "trial requires the taxonomy table."
        )

    # ! Process taxonomy and phylogeny by microbial feature table, exactly as
    # find_best_model_config does before dispatching trials.
    ft_col = [x for x in train_val.columns if x.startswith("F")]
    if tax is not None:
        tax = _process_taxonomy(tax, train_val[ft_col])
    if tree_phylo is not None:
        tree_phylo = _process_phylogeny(tree_phylo, train_val[ft_col])

    k_folds = adaptive_k_folds(
        train_val,
        group_by_column=exp_config.get("group_by_column"),
        stratify_by=exp_config.get("stratify_by"),
        target=exp_config.get("target"),
        task_type=exp_config.get("task_type", "regression"),
        requested=exp_config.get("k_folds"),
    )

    # ! Retrain inside a temporary directory: trial checkpoints and artifacts
    # (needed by build_tuned_model_from_result / build_provenance_map) live
    # under this directory and must be consumed before it is torn down.
    with tempfile.TemporaryDirectory() as path2exp:
        grid = retrain_fixed_configs(
            trial_configs,
            train_val,
            exp_config.get("target"),
            exp_config.get("group_by_column"),
            exp_config.get("stratify_by"),
            exp_config.get("seed_data"),
            exp_config.get("seed_model"),
            tax,
            tree_phylo,
            path2exp,
            max_concurrent_trials,
            task_type=exp_config.get("task_type", "regression"),
            k_folds=k_folds,
            nn_corn_max_levels=exp_config.get(
                "nn_corn_max_levels", DEFAULT_NN_CORN_MAX_LEVELS
            ),
        )

        results_by_run_id = {
            result.config["trial_config"]["mlflow_run_id"]: result for result in grid
        }
        failed_run_ids = [
            run_id
            for run_id, result in results_by_run_id.items()
            if result.error is not None
        ]
        if failed_run_ids:
            warnings.warn(
                f"Dropped {len(failed_run_ids)} band trial(s) that failed to "
                f"retrain (run_id(s): {failed_run_ids}); excluded from the "
                f"manifest, importances, and rank agreement."
            )

        importances: Dict[str, pd.DataFrame] = {}
        provenances: Dict[str, Dict] = {}
        configs_by_run_id: Dict[str, dict] = {}
        retrained_means: Dict[str, float] = {}
        for run_id, result in results_by_run_id.items():
            if result.error is not None:
                continue
            inner = result.config["trial_config"]
            tmodel = build_tuned_model_from_result(
                inner["model"], result, train_val, trial_config=inner
            )
            importances[run_id] = compute_normalized_importance(
                tmodel,
                train_val,
                test,
                max_background_samples=max_background_samples,
            )
            provenances[run_id] = build_provenance_map(tmodel)
            configs_by_run_id[run_id] = inner
            retrained_means[run_id] = result.metrics[f"{metric}_mean"]

        if reference_run_id not in importances:
            raise ValueError(
                f"Reference trial {reference_run_id} failed to retrain; "
                "cannot compute stability without it."
            )

        reference_importance = importances[reference_run_id]["importance"]
        if len(reference_importance) > 1 and np.allclose(
            reference_importance, reference_importance.iloc[0]
        ):
            warnings.warn(
                f"Reference trial {reference_run_id} has no discriminative "
                "feature importances -- every feature ties at the same "
                f"value ({reference_importance.iloc[0]:.4g}), typically "
                "because the model is degenerate (e.g. regularized to a "
                "constant predictor). Rank-based stability comparisons "
                "against it are uninformative."
            )

    surviving_band = band[band["run_id"].isin(importances.keys())]
    manifest = _band_manifest(surviving_band, reference_run_id, metric, retrained_means)
    importances_long = pd.concat(
        [df.assign(run_id=run_id) for run_id, df in importances.items()],
        ignore_index=True,
    )
    ranks = align_feature_ranks(
        reference_run_id, importances, provenances, configs_by_run_id, top_n=top_n
    )
    agreement = compute_rank_agreement(ranks, top_n=top_n)
    fig = plot_stability(manifest, ranks, metric, top_n=top_n, show=False)

    return manifest, importances_long, ranks, agreement, fig


@main_function
def cli_explain_stability(
    path_to_exp: str,
    model_type: str,
    path_to_train_val: str,
    path_to_test: str,
    path_to_tax: str = None,
    path_to_tree_phylo: str = None,
    band_se_factor: float = 1.0,
    max_trials: int = 15,
    top_n: int = 15,
    max_background_samples: Optional[int] = None,
    max_concurrent_trials: int = 2,
) -> None:
    """Compute feature-importance stability across the near-optimal band of
    a completed experiment and write artifacts to disk.

    Thin CLI wrapper around :func:`explain_stability` — loads the trial
    records, train/test pickles, and optional taxonomy/phylogeny, delegates
    the analysis, and persists the result.

    Args:
        path_to_exp: Path to the experiment directory containing
            ``experiment_config.json`` and ``mlflow_logs.csv``.
        model_type: Model type to scope the band to (e.g. "linreg", "xgb"),
            or "all".
        path_to_train_val: Path to the pickled train/validation DataFrame.
        path_to_test: Path to the pickled test DataFrame.
        path_to_tax: Path to a taxonomy TSV. Required when the band includes
            aggregation-using trials or trac.
        path_to_tree_phylo: Path to a phylogeny Newick file. Required when
            the band includes trac.
        band_se_factor: Half-width of the comparable-performance band, in
            units of the *reference* trial's standard error (the same factor
            is applied to the best trial's SE when picking the reference).
        max_trials: Maximum number of band trials to retrain.
        top_n: Number of the reference model's top features to compare. Also
            sets the figure's colour-scale clamp at ``2 * top_n``.
        max_background_samples: If set, subsample the SHAP background to
            this many rows. Ignored for coefficient-bearing models.
        max_concurrent_trials: Maximum number of concurrent retrain trials.

    Side Effects:
        Writes into ``path_to_exp/stability_<model_type>/``:
            ``stability_trials.csv``, ``stability_importances.csv``,
            ``stability_ranks.csv``, ``stability_agreement.csv``,
            ``stability_plot.png``.
    """
    exp_config = load_experiment_config(path_to_exp)
    if exp_config.get("tracking_uri") != "mlruns":
        raise ValueError(
            "explain-stability requires a local MLflow trial table; "
            f"experiment tracking_uri is {exp_config.get('tracking_uri')!r}, "
            "not 'mlruns'."
        )

    path_to_logs = os.path.join(path_to_exp, "mlflow_logs.csv")
    if not os.path.exists(path_to_logs):
        raise ValueError(f"No mlflow_logs.csv found at {path_to_logs}.")

    trial_records = load_trial_records(path_to_logs)
    train_val = pd.read_pickle(path_to_train_val)
    test = pd.read_pickle(path_to_test)
    tax = _load_taxonomy(path_to_tax) if path_to_tax is not None else None
    tree_phylo = (
        _load_phylogeny(path_to_tree_phylo) if path_to_tree_phylo is not None else None
    )

    manifest, importances_long, ranks, agreement, fig = explain_stability(
        exp_config,
        trial_records,
        train_val,
        test,
        model_type=model_type,
        tax=tax,
        tree_phylo=tree_phylo,
        band_se_factor=band_se_factor,
        max_trials=max_trials,
        top_n=top_n,
        max_background_samples=max_background_samples,
        max_concurrent_trials=max_concurrent_trials,
    )

    out_dir = os.path.join(path_to_exp, f"stability_{model_type}")
    os.makedirs(out_dir, exist_ok=True)

    trials_path = os.path.join(out_dir, "stability_trials.csv")
    manifest.to_csv(trials_path, index=False)
    print(f"Stability trial manifest saved in {trials_path}.")

    importances_path = os.path.join(out_dir, "stability_importances.csv")
    importances_long.to_csv(importances_path, index=False)
    print(f"Stability importances saved in {importances_path}.")

    ranks_path = os.path.join(out_dir, "stability_ranks.csv")
    ranks.to_csv(ranks_path, index=False)
    print(f"Stability ranks saved in {ranks_path}.")

    agreement_path = os.path.join(out_dir, "stability_agreement.csv")
    agreement.to_csv(agreement_path, index=False)
    print(f"Stability agreement stats saved in {agreement_path}.")

    plot_path = os.path.join(out_dir, "stability_plot.png")
    fig.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Stability plot saved in {plot_path}.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    typer.run(cli_explain_stability)
