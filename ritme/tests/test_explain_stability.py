import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from ritme.explain_stability import (
    _CELL_LEGEND_ENTRIES,
    _legend_max_chars,
    _pack_legend_lines,
    _parse_param_value,
    _rank_cell_text,
    align_feature_ranks,
    cli_explain_stability,
    compute_normalized_importance,
    compute_rank_agreement,
    explain_stability,
    filter_reproducible_trials,
    load_trial_records,
    plot_stability,
    reconstruct_trial_config,
    select_band_trials,
    select_reference_run_id,
)
from ritme.feature_space.feature_provenance import FeatureProvenance

matplotlib.use("Agg")


def _records_df():
    return pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3", "r4"],
            "experiment_name": ["linreg", "linreg", "linreg", "xgb"],
            "status": ["FINISHED", "FINISHED", "RUNNING", "FINISHED"],
            "params.model": ["linreg", "linreg", "linreg", "xgb"],
            "params.alpha": ["0.6412595812188677", "21.771379898089695", "0.1", None],
            "params.n_estimators": [None, None, None, "2289"],
            "params.data_aggregation": ["tax_class", None, None, "tax_genus"],
            "params.data_enrich_with": ["['body-site']", None, None, None],
            "metrics.rmse_val_mean": [3.1, 3.2, 3.0, 2.9],
            "metrics.rmse_val_se": [0.3, 0.2, 0.1, np.nan],
            "metrics.n_folds": [5.0, 5.0, 5.0, 5.0],
            "metrics.nb_features": [10.0, 20.0, 5.0, 21.0],
        }
    )


class TestParseParamValue(unittest.TestCase):
    def test_types_roundtrip(self):
        self.assertIsNone(_parse_param_value("None"))
        self.assertEqual(_parse_param_value("2289"), 2289)
        self.assertEqual(_parse_param_value("0.6412595812188677"), 0.6412595812188677)
        self.assertEqual(_parse_param_value("['body-site']"), ["body-site"])
        self.assertIs(_parse_param_value("True"), True)
        self.assertEqual(_parse_param_value("tax_class"), "tax_class")
        self.assertEqual(_parse_param_value("abundance_ith"), "abundance_ith")


class TestLoadTrialRecords(unittest.TestCase):
    def test_params_stay_str_metrics_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mlflow_logs.csv")
            _records_df().to_csv(path, index=False)
            records = load_trial_records(path)
        self.assertEqual(records["params.alpha"].iloc[0], "0.6412595812188677")
        self.assertTrue(pd.api.types.is_float_dtype(records["metrics.rmse_val_mean"]))


class TestReconstructTrialConfig(unittest.TestCase):
    def test_typed_config_with_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mlflow_logs.csv")
            _records_df().to_csv(path, index=False)
            records = load_trial_records(path)
        config = reconstruct_trial_config(records.iloc[0])
        self.assertEqual(config["model"], "linreg")
        self.assertEqual(config["alpha"], 0.6412595812188677)
        self.assertEqual(config["data_aggregation"], "tax_class")
        self.assertEqual(config["data_enrich_with"], ["body-site"])
        self.assertEqual(config["mlflow_run_id"], "r1")
        self.assertNotIn("n_estimators", config)

    def test_data_key_defaults_to_none_when_unlogged(self):
        # r2's params.data_aggregation cell is NaN in the fixture, because
        # MLflow never logs a None-valued param — it must still surface as an
        # explicit None, since the trainables require data_* keys to exist.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mlflow_logs.csv")
            _records_df().to_csv(path, index=False)
            records = load_trial_records(path)
        config = reconstruct_trial_config(records.iloc[1])
        self.assertIn("data_aggregation", config)
        self.assertIsNone(config["data_aggregation"])


class TestFilterReproducibleTrials(unittest.TestCase):
    def test_drops_running_and_nan_se(self):
        kept = filter_reproducible_trials(_records_df(), "rmse_val")
        self.assertEqual(sorted(kept["run_id"]), ["r1", "r2"])

    def test_single_split_missing_columns_raises_clear_error(self):
        # a k_folds=1 run never logs metrics.n_folds/_se at all -- this must
        # surface as the intended guard-clause ValueError, not a bare
        # KeyError from indexing a column that doesn't exist.
        records = _records_df().drop(columns=["metrics.n_folds", "metrics.rmse_val_se"])
        with self.assertRaises(ValueError):
            filter_reproducible_trials(records, "rmse_val")


class TestSelectBandTrials(unittest.TestCase):
    def test_band_membership_min_metric(self):
        records = _records_df()
        band = select_band_trials(records, "linreg", "rmse_val", "min")
        # best reproducible linreg mean = 3.1 (r1), se = 0.3 -> band <= 3.4
        self.assertEqual(sorted(band["run_id"]), ["r1", "r2"])

    def test_anchor_run_id_selects_trials_within_se_of_that_trial_not_best(self):
        # r1 is best but has a narrow se=0.1, so its own 1-SE band excludes
        # r3; r2's se=0.5 includes it. The anchored band must use r2's
        # window, not best's.
        records = pd.DataFrame(
            {
                "run_id": ["r1", "r2", "r3"],
                "experiment_name": ["linreg"] * 3,
                "status": ["FINISHED"] * 3,
                "params.model": ["linreg"] * 3,
                "metrics.rmse_val_mean": [3.0, 3.05, 3.4],
                "metrics.rmse_val_se": [0.1, 0.5, 0.2],
                "metrics.n_folds": [5.0] * 3,
                "metrics.nb_features": [20.0, 10.0, 15.0],
            }
        )
        best_anchored = select_band_trials(records, "linreg", "rmse_val", "min")
        self.assertEqual(sorted(best_anchored["run_id"]), ["r1", "r2"])

        r2_anchored = select_band_trials(
            records, "linreg", "rmse_val", "min", anchor_run_id="r2"
        )
        self.assertEqual(sorted(r2_anchored["run_id"]), ["r1", "r2", "r3"])

    def test_symmetric_excludes_trials_notably_better_than_anchor(self):
        # the one-sided 1-SE rule bounds only how much worse a trial may be
        # than the anchor; symmetric=True must exclude notably better ones
        # too, so every trial falls inside the plotted band.
        records = pd.DataFrame(
            {
                "run_id": ["ref", "much_better", "t3"],
                "experiment_name": ["linreg"] * 3,
                "status": ["FINISHED"] * 3,
                "params.model": ["linreg"] * 3,
                "metrics.rmse_val_mean": [3.5, 3.1, 3.6],
                "metrics.rmse_val_se": [0.2, 0.1, 0.2],
                "metrics.n_folds": [5.0] * 3,
                "metrics.nb_features": [10.0, 20.0, 15.0],
            }
        )
        one_sided = select_band_trials(
            records, "linreg", "rmse_val", "min", anchor_run_id="ref"
        )
        self.assertIn("much_better", one_sided["run_id"].tolist())

        symmetric = select_band_trials(
            records,
            "linreg",
            "rmse_val",
            "min",
            anchor_run_id="ref",
            symmetric=True,
        )
        self.assertNotIn("much_better", symmetric["run_id"].tolist())
        self.assertEqual(sorted(symmetric["run_id"]), ["ref", "t3"])

    def test_anchor_run_id_unknown_raises(self):
        with self.assertRaises(ValueError):
            select_band_trials(
                _records_df(), "linreg", "rmse_val", "min", anchor_run_id="nope"
            )

    def test_band_se_factor_zero_keeps_only_best(self):
        band = select_band_trials(
            _records_df(), "linreg", "rmse_val", "min", band_se_factor=0.0
        )
        self.assertEqual(list(band["run_id"]), ["r1"])

    def test_max_trials_cap_warns(self):
        with self.assertWarns(UserWarning):
            band = select_band_trials(
                _records_df(), "linreg", "rmse_val", "min", max_trials=1
            )
        self.assertEqual(len(band), 1)

    def test_unknown_scope_raises(self):
        with self.assertRaises(ValueError):
            select_band_trials(_records_df(), "trac", "rmse_val", "min")

    def test_no_finite_se_raises(self):
        records = _records_df()
        records["metrics.rmse_val_se"] = np.nan
        with self.assertRaises(ValueError):
            select_band_trials(records, "linreg", "rmse_val", "min")


class TestSelectReferenceRunId(unittest.TestCase):
    def test_one_se_winner_is_simplest_in_band(self):
        band = select_band_trials(_records_df(), "linreg", "rmse_val", "min")
        # r2 (mean 3.2, 20 features) is within 1 SE of r1 (mean 3.1, 10
        # features) but not simpler, so the reference stays r1.
        self.assertEqual(select_reference_run_id(band, "rmse_val", "min"), "r1")

    def test_simpler_trial_within_one_se_wins(self):
        records = _records_df()
        records.loc[records["run_id"] == "r2", "metrics.nb_features"] = 2.0
        band = select_band_trials(records, "linreg", "rmse_val", "min")
        self.assertEqual(select_reference_run_id(band, "rmse_val", "min"), "r2")


class TestComputeNormalizedImportance(unittest.TestCase):
    @patch("ritme.explain_stability.explain_features")
    def test_coef_path_multiclass_mean(self, mock_explain):
        mock_explain.return_value = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f1", "f2"],
                "class": ["a", "a", "b", "b"],
                "coefficient": [1.0, 0.5, -3.0, 0.5],
                "abs_coefficient": [1.0, 0.5, 3.0, 0.5],
            }
        )
        imp = compute_normalized_importance(MagicMock(), None, None)
        f1 = imp.set_index("feature").loc["f1"]
        self.assertAlmostEqual(f1["importance"], 2.0)
        self.assertEqual(int(f1["rank"]), 1)

    @patch("ritme.explain_stability.explain_features")
    def test_shap_path_2d_and_3d(self, mock_explain):
        values_3d = np.stack(
            [np.array([[1.0, -2.0], [3.0, 0.0]]) for _ in range(2)], axis=2
        )
        mock_explain.return_value = shap.Explanation(
            values=values_3d,
            base_values=np.zeros(2),
            data=np.zeros((2, 2)),
            feature_names=["f1", "f2"],
        )
        imp = compute_normalized_importance(MagicMock(), None, None)
        self.assertAlmostEqual(imp.set_index("feature").loc["f1", "importance"], 2.0)
        self.assertEqual(int(imp.set_index("feature").loc["f1", "rank"]), 1)


def _prov(column, kind="taxon", otus=None, snapshot="t0"):
    return FeatureProvenance(column, kind, snapshot, frozenset(otus or {column}))


def _imp(features):
    table = pd.DataFrame(
        {"feature": features, "importance": np.linspace(1.0, 0.1, len(features))}
    )
    table["rank"] = range(1, len(features) + 1)
    return table


class TestAlignFeatureRanks(unittest.TestCase):
    def setUp(self):
        self.importances = {
            "ref": _imp(["clr_F1", "clr_F2", "shannon_entropy"]),
            "t2": _imp(["shannon_entropy", "clr_F1"]),
        }
        self.provenances = {
            "ref": {
                "clr_F1": _prov("clr_F1", otus={"F1"}),
                "clr_F2": _prov("clr_F2", otus={"F2"}),
                "shannon_entropy": _prov(
                    "shannon_entropy", kind="diversity", otus=set()
                ),
            },
            "t2": {
                "clr_F1": _prov("clr_F1", otus={"F1"}),
                "shannon_entropy": _prov(
                    "shannon_entropy", kind="diversity", otus=set()
                ),
            },
        }
        self.configs = {
            "ref": {"data_transform": "clr"},
            "t2": {"data_transform": "clr"},
        }

    def test_alignment_ranks_and_kinds(self):
        ranks = align_feature_ranks(
            "ref", self.importances, self.provenances, self.configs, top_n=3
        )
        t2 = ranks[ranks["run_id"] == "t2"].set_index("feature")
        self.assertEqual(t2.loc["clr_F1", "match_kind"], "identical")
        self.assertEqual(t2.loc["clr_F1", "rank"], 2)
        self.assertEqual(t2.loc["clr_F2", "match_kind"], "absent")
        self.assertTrue(np.isnan(t2.loc["clr_F2", "rank"]))
        # reference matched against itself: all identical, rank == reference_rank
        ref = ranks[ranks["run_id"] == "ref"]
        self.assertTrue((ref["match_kind"] == "identical").all())

    def test_agreement_stats(self):
        ranks = align_feature_ranks(
            "ref", self.importances, self.provenances, self.configs, top_n=3
        )
        stats = compute_rank_agreement(ranks, top_n=3)
        t2 = stats.set_index("run_id").loc["t2"]
        # 2 of 3 reference top-3 features rank within top 3 of t2
        self.assertAlmostEqual(t2["top_n_containment"], 2 / 3)
        self.assertNotIn("kendall_tau", stats.columns)

    def test_n_matched_parts_counts_split_columns(self):
        ranks = align_feature_ranks(
            "ref", self.importances, self.provenances, self.configs, top_n=3
        )
        # clr_F1 matched identically (1 part); clr_F2 absent (0 parts)
        t2 = ranks[ranks["run_id"] == "t2"].set_index("feature")
        self.assertEqual(t2.loc["clr_F1", "n_matched_parts"], 1)
        self.assertEqual(t2.loc["clr_F2", "n_matched_parts"], 0)

    def test_n_matched_parts_counts_a_real_split_into_multiple_parts(self):
        # a coarser reference aggregate (agg_A over {F1,F2,F3}) resolves in
        # the trial into three finer per-OTU columns -- match_kind "split"
        # with n_matched_parts == 3, not the 0/1-part cases above.
        importances = {
            "ref": _imp(["agg_A"]),
            "t3": _imp(["F1", "F2", "F3"]),
        }
        provenances = {
            "ref": {"agg_A": _prov("agg_A", otus={"F1", "F2", "F3"})},
            "t3": {
                "F1": _prov("F1", otus={"F1"}),
                "F2": _prov("F2", otus={"F2"}),
                "F3": _prov("F3", otus={"F3"}),
            },
        }
        configs = {"ref": {"data_transform": None}, "t3": {"data_transform": None}}
        ranks = align_feature_ranks("ref", importances, provenances, configs, top_n=1)
        t3 = ranks[ranks["run_id"] == "t3"].set_index("feature")
        self.assertEqual(t3.loc["agg_A", "match_kind"], "split")
        self.assertEqual(t3.loc["agg_A", "n_matched_parts"], 3)
        # all three source OTUs are accounted for
        self.assertAlmostEqual(t3.loc["agg_A", "otu_coverage"], 1.0)

    def test_partial_split_reports_its_otu_coverage_shortfall(self):
        # only one of the reference aggregate's three OTUs survived the
        # trial's selection; the other two sit in its lumped bucket and are
        # not represented in the reported (best-of-parts) rank.
        importances = {"ref": _imp(["agg_A"]), "t3": _imp(["F1", "F_low_abun"])}
        provenances = {
            "ref": {"agg_A": _prov("agg_A", otus={"F1", "F2", "F3"})},
            "t3": {
                "F1": _prov("F1", otus={"F1"}),
                "F_low_abun": _prov("F_low_abun", kind="lumped", otus={"F2", "F3"}),
            },
        }
        configs = {"ref": {"data_transform": None}, "t3": {"data_transform": None}}
        ranks = align_feature_ranks("ref", importances, provenances, configs, top_n=1)
        t3 = ranks[ranks["run_id"] == "t3"].set_index("feature")
        self.assertEqual(t3.loc["agg_A", "match_kind"], "split")
        self.assertEqual(t3.loc["agg_A", "n_matched_parts"], 1)
        self.assertAlmostEqual(t3.loc["agg_A", "otu_coverage"], 1 / 3)

    def test_trac_band_trial_is_not_attributable_without_crashing(self):
        # trac shares no feature namespace with the reference, so it must
        # be reported not_attributable, never matched. Its feature names
        # are absent from its own provenance map, so any lookup there
        # would raise.
        importances = dict(self.importances)
        importances["trac1"] = _imp(["some__clade; lineage"])
        provenances = dict(self.provenances)
        provenances["trac1"] = {}  # would KeyError if ever indexed
        configs = dict(self.configs)
        configs["trac1"] = {"model": "trac"}

        ranks = align_feature_ranks("ref", importances, provenances, configs, top_n=3)
        trac_rows = ranks[ranks["run_id"] == "trac1"]
        self.assertTrue((trac_rows["match_kind"] == "not_attributable").all())
        self.assertTrue(trac_rows["rank"].isna().all())
        self.assertTrue((trac_rows["n_matched_parts"] == 0).all())

    def test_not_attributable_excluded_from_containment_denominator(self):
        importances = dict(self.importances)
        importances["trac1"] = _imp(["some__clade; lineage"])
        provenances = dict(self.provenances)
        provenances["trac1"] = {}
        configs = dict(self.configs)
        configs["trac1"] = {"model": "trac"}

        ranks = align_feature_ranks("ref", importances, provenances, configs, top_n=3)
        stats = compute_rank_agreement(ranks, top_n=3)
        trac_stats = stats.set_index("run_id").loc["trac1"]
        # every reference feature is not_attributable against trac1 ->
        # containment is undefined (NaN), not 0.0 ("total disagreement")
        self.assertTrue(np.isnan(trac_stats["top_n_containment"]))


class TestPackLegendLines(unittest.TestCase):
    def test_packs_entries_up_to_the_width_without_splitting_them(self):
        entries = ("aaa", "bbb", "ccc")
        # "aaa   bbb" is 9 chars, adding "   ccc" would make 15 > 12
        self.assertEqual(_pack_legend_lines(entries, 12), ["aaa   bbb", "ccc"])

    def test_entry_longer_than_the_line_is_kept_whole(self):
        # breaking a legend entry mid-symbol would make it unreadable; an
        # over-long entry gets its own (over-wide) line instead.
        self.assertEqual(_pack_legend_lines(("a" * 30, "b"), 10), ["a" * 30, "b"])

    def test_max_chars_scales_with_the_available_width(self):
        self.assertGreater(_legend_max_chars(8.0), _legend_max_chars(4.0))
        # never collapses to an unusable width for a tiny figure
        self.assertGreaterEqual(_legend_max_chars(0.1), 20)


def _manifest():
    return pd.DataFrame(
        {
            "run_id": ["ref", "t2"],
            "experiment_name": ["linreg", "linreg"],
            "metric_mean_logged": [3.0, 3.1],
            "metric_se_logged": [0.2, 0.3],
            "is_reference": [True, False],
            "is_best": [True, False],
            "config_summary": ["none|none|clr|none", "none|none|clr|none"],
        }
    )


def _figure_text_sizes(fig):
    """Font size of the figure title and the set of sizes used by every other
    non-empty text artist in the figure."""
    title = [t for t in fig.texts if t.get_text().startswith("Feature stability")][0]
    body = [t for t in fig.texts if t is not title]
    for ax in fig.axes:
        body.extend(
            [
                ax.title,
                ax.xaxis.label,
                ax.yaxis.label,
                *ax.texts,
                *ax.get_xticklabels(),
                *ax.get_yticklabels(),
            ]
        )
        legend = ax.get_legend()
        if legend is not None:
            body.extend(legend.get_texts())
    sizes = {t.get_fontsize() for t in body if t.get_text().strip()}
    return title.get_fontsize(), sizes


class TestPlotStability(unittest.TestCase):
    def setUp(self):
        self.importances = {
            "ref": _imp(["clr_F1", "clr_F2"]),
            "t2": _imp(["clr_F2", "clr_F1"]),
        }
        self.ranks = pd.DataFrame(
            {
                "feature": ["clr_F1", "clr_F2"] * 2,
                "reference_rank": [1, 2, 1, 2],
                "run_id": ["ref", "ref", "t2", "t2"],
                "match_kind": ["identical"] * 4,
                "matched_column": ["clr_F1", "clr_F2"] * 2,
                "rank": [1.0, 2.0, 2.0, 1.0],
            }
        )

    def test_returns_figure_with_two_content_panels(self):
        # 2 content panels (performance, ranks) + 1 colorbar axes = 3
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 3)
        plt.close(fig)

    def test_show_true_returns_none(self):
        result = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=True,
        )
        self.assertIsNone(result)

    def test_rank_cell_text_all_match_kinds(self):
        cases = [
            ("identical", 3.0, "3"),
            ("contained", 4.0, "4◆"),
            ("lumped", 5.0, "5L"),
            ("split", 6.0, "6s"),
            ("absent", np.nan, "–"),
            ("not_attributable", np.nan, "n/a"),
        ]
        for match_kind, rank, expected in cases:
            with self.subTest(match_kind=match_kind):
                row = pd.Series({"match_kind": match_kind, "rank": rank})
                self.assertEqual(_rank_cell_text(row), expected)

    def test_not_attributable_cell_gets_grey_rectangle(self):
        # clr_F2 x t2 is not_attributable (rank i=1, trial j=1); every other
        # (feature, run_id) combination stays identical so the matrix/pivot
        # lookups plot_stability performs still resolve.
        ranks = pd.DataFrame(
            {
                "feature": ["clr_F1", "clr_F2", "clr_F1", "clr_F2"],
                "reference_rank": [1, 2, 1, 2],
                "run_id": ["ref", "ref", "t2", "t2"],
                "match_kind": [
                    "identical",
                    "identical",
                    "identical",
                    "not_attributable",
                ],
                "matched_column": ["clr_F1", "clr_F2", "clr_F1", ""],
                "rank": [1.0, 2.0, 1.0, np.nan],
            }
        )
        fig = plot_stability(
            _manifest(),
            ranks,
            "rmse_val",
            show=False,
        )
        ax_ranks = fig.axes[1]
        grey_patches = [p for p in ax_ranks.patches if isinstance(p, Rectangle)]
        self.assertEqual(len(grey_patches), 1)
        patch = grey_patches[0]
        self.assertEqual(patch.get_facecolor(), to_rgba("grey"))
        # anchored at (j - 0.5, i - 0.5) for column j=1 (t2), row i=1 (clr_F2)
        self.assertAlmostEqual(patch.get_x(), 0.5)
        self.assertAlmostEqual(patch.get_y(), 0.5)
        plt.close(fig)

    def test_legend_shows_only_trial_mean_se(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_perf = fig.axes[0]
        legend = ax_perf.get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(labels, ["Trial mean ± SE"])
        plt.close(fig)

    def test_ylabel_uses_friendly_metric_name(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_perf = fig.axes[0]
        self.assertEqual(ax_perf.get_ylabel(), "RMSE val")
        plt.close(fig)

    def test_ylabel_falls_back_to_raw_metric_for_unknown_names(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "some_new_metric",
            show=False,
        )
        ax_perf = fig.axes[0]
        self.assertEqual(ax_perf.get_ylabel(), "some_new_metric")
        plt.close(fig)

    def test_xlabel_and_colorbar_label_are_simplified(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_ranks = fig.axes[1]
        self.assertEqual(ax_ranks.get_xlabel(), "Trial")
        cbar_ax = fig.axes[2]
        self.assertEqual(cbar_ax.get_ylabel(), "rank")
        plt.close(fig)

    def _ranks_with_worst(self, worst_rank):
        ranks = self.ranks.copy()
        ranks.loc[ranks.index[-1], "rank"] = worst_rank
        return ranks

    def test_colorbar_top_tick_marks_ranks_clamped_at_the_cap(self):
        # ranks beyond 2*top_n share the darkest colour, so the top of the
        # scale is a censored value, not an exact rank.
        fig = plot_stability(
            _manifest(),
            self._ranks_with_worst(400.0),
            "rmse_val",
            top_n=15,
            show=False,
        )
        labels = [t.get_text() for t in fig.axes[2].get_yticklabels()]
        self.assertEqual(labels[-1], "≥30")
        plt.close(fig)

    def test_colorbar_top_tick_stays_exact_when_nothing_is_clamped(self):
        # a worst rank of exactly 30 is observed, not censored -- adding
        # "≥" there would claim a clamp that never happened.
        fig = plot_stability(
            _manifest(),
            self._ranks_with_worst(30.0),
            "rmse_val",
            top_n=15,
            show=False,
        )
        labels = [t.get_text() for t in fig.axes[2].get_yticklabels()]
        self.assertNotIn("≥", "".join(labels))
        plt.close(fig)

    def test_clamped_cell_still_prints_its_true_rank(self):
        # only the colour saturates; the number must stay truthful.
        fig = plot_stability(
            _manifest(),
            self._ranks_with_worst(400.0),
            "rmse_val",
            top_n=15,
            show=False,
        )
        texts = [t.get_text() for t in fig.axes[1].texts]
        self.assertIn("400", texts)
        plt.close(fig)

    def test_ytick_labels_are_plain_feature_names(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_ranks = fig.axes[1]
        labels = [t.get_text() for t in ax_ranks.get_yticklabels()]
        self.assertEqual(labels, ["clr_F1", "clr_F2"])
        plt.close(fig)

    def test_xtick_labels_split_config_over_two_lines_without_model_name(self):
        # the model type moves to the figure title, so a column carries only
        # its four feature-engineering steps, two per line.
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_ranks = fig.axes[1]
        labels = [t.get_text() for t in ax_ranks.get_xticklabels()]
        self.assertEqual(labels, ["none|none\nclr|none", "none|none\nclr|none"])
        plt.close(fig)

    def test_xtick_labels_carry_no_reference_or_best_marker(self):
        # the performance panel already marks the reference (square) and the
        # best trial (star); repeating them on the column labels is noise.
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_ranks = fig.axes[1]
        for label in ax_ranks.get_xticklabels():
            self.assertNotIn("▶", label.get_text())
            self.assertNotIn("★", label.get_text())
        plt.close(fig)

    def test_xtick_labels_keep_model_name_when_band_spans_model_types(self):
        # with model_type="all" the model name is the only thing telling
        # two columns apart, so the label must carry it.
        manifest = _manifest()
        manifest["experiment_name"] = ["linreg", "xgb"]
        fig = plot_stability(manifest, self.ranks, "rmse_val", show=False)
        ax_ranks = fig.axes[1]
        labels = [t.get_text() for t in ax_ranks.get_xticklabels()]
        self.assertEqual(
            labels,
            ["linreg\nnone|none\nclr|none", "xgb\nnone|none\nclr|none"],
        )
        plt.close(fig)

    def test_every_text_element_but_the_title_shares_one_font_size(self):
        # tick labels, axis labels, heatmap cells, colorbar and both legends
        # must render at one size, so no element reads as more important.
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        _, body_sizes = _figure_text_sizes(fig)
        self.assertEqual(len(body_sizes), 1, f"mixed body font sizes: {body_sizes}")
        plt.close(fig)

    def test_title_is_larger_than_the_rest_of_the_text(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        title_size, body_sizes = _figure_text_sizes(fig)
        self.assertGreater(title_size, max(body_sizes))
        plt.close(fig)

    def test_capped_colorbar_ticks_keep_the_shared_font_size(self):
        # set_ticks() rebuilds the tick artists after tick_params() ran, so
        # the censored ">=N" scale must not fall back to the rcParam size.
        ranks = self.ranks.copy()
        ranks["rank"] = ranks["rank"] * 100
        fig = plot_stability(_manifest(), ranks, "rmse_val", top_n=1, show=False)
        cbar_ax = fig.axes[-1]
        labels = [t for t in cbar_ax.get_yticklabels() if t.get_text().strip()]
        self.assertTrue(any(label.get_text().startswith("\u2265") for label in labels))
        _, body_sizes = _figure_text_sizes(fig)
        self.assertEqual(len(body_sizes), 1, f"mixed body font sizes: {body_sizes}")
        plt.close(fig)

    def test_suptitle_names_the_model_type_of_the_band(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        titles = [t.get_text() for t in fig.texts]
        self.assertIn("Feature stability among top-performing linreg trials", titles)
        plt.close(fig)

    def test_legend_block_starts_at_the_heatmap_left_edge(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_ranks = fig.axes[1]
        legend = [t for t in fig.texts if "rank in that trial" in t.get_text()][0]
        self.assertAlmostEqual(
            legend.get_position()[0], ax_ranks.get_position().x0, places=4
        )
        plt.close(fig)

    def test_legend_wraps_to_the_heatmap_width(self):
        # a narrow (2-trial) figure cannot fit all six entries on one line;
        # the block must wrap rather than run past the heatmap's right edge.
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_ranks = fig.axes[1]
        legend = [t for t in fig.texts if "rank in that trial" in t.get_text()][0]
        lines = legend.get_text().split("\n")
        self.assertGreater(len(lines), 1)
        width_inches = ax_ranks.get_position().width * fig.get_figwidth()
        max_chars = _legend_max_chars(width_inches)
        for line in lines:
            # a single entry wider than the panel is the documented
            # exception: it keeps its own line rather than break mid-symbol.
            self.assertTrue(
                len(line) <= max_chars or line in _CELL_LEGEND_ENTRIES,
                f"line overruns the panel but is not one entry: {line!r}",
            )
        plt.close(fig)

    def test_perf_panel_width_matches_ranks_panel_after_colorbar(self):
        # the colorbar shrinks ax_ranks's width; ax_perf must be resized to
        # match so its data lines up with the heatmap columns below.
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_perf, ax_ranks = fig.axes[0], fig.axes[1]
        ranks_pos = ax_ranks.get_position()
        pos = ax_perf.get_position()
        self.assertAlmostEqual(pos.x0, ranks_pos.x0, places=4)
        self.assertAlmostEqual(pos.width, ranks_pos.width, places=4)
        plt.close(fig)

    def test_axhspan_shades_best_trial_band(self):
        fig = plot_stability(
            _manifest(),
            self.ranks,
            "rmse_val",
            show=False,
        )
        ax_perf = fig.axes[0]
        self.assertEqual(len(ax_perf.patches), 1)
        band = ax_perf.patches[0]
        manifest = _manifest()
        expected_low = (
            manifest["metric_mean_logged"].iloc[0]
            - manifest["metric_se_logged"].iloc[0]
        )
        expected_high = (
            manifest["metric_mean_logged"].iloc[0]
            + manifest["metric_se_logged"].iloc[0]
        )
        self.assertAlmostEqual(band.get_y(), expected_low)
        self.assertAlmostEqual(band.get_y() + band.get_height(), expected_high)
        plt.close(fig)

    def test_axhspan_anchors_on_reference_not_best_when_they_differ(self):
        # the reference (simplest-within-1SE) is often not the best
        # performer; the shaded band must track the reference's own
        # mean/SE.
        manifest = pd.DataFrame(
            {
                "run_id": ["ref", "best", "t3"],
                "experiment_name": ["linreg"] * 3,
                "metric_mean_logged": [3.4, 3.0, 3.2],
                "metric_se_logged": [0.3, 0.2, 0.25],
                "is_reference": [True, False, False],
                "is_best": [False, True, False],
                "config_summary": ["none|none|clr|none"] * 3,
            }
        )
        ranks = pd.DataFrame(
            {
                "feature": ["clr_F1"] * 3,
                "reference_rank": [1] * 3,
                "run_id": ["ref", "best", "t3"],
                "match_kind": ["identical"] * 3,
                "matched_column": ["clr_F1"] * 3,
                "rank": [1.0] * 3,
            }
        )
        fig = plot_stability(
            manifest,
            ranks,
            "rmse_val",
            show=False,
        )
        ax_perf = fig.axes[0]
        band = ax_perf.patches[0]
        self.assertAlmostEqual(band.get_y(), 3.4 - 0.3)
        self.assertAlmostEqual(band.get_y() + band.get_height(), 3.4 + 0.3)
        plt.close(fig)

    def test_panels_share_x_axis_at_non_default_trial_count(self):
        # Panel A (errorbar) autoscales with margins while panel B (imshow)
        # fixes its own limits, so without sharex they align only at
        # particular trial counts.
        n = 5
        run_ids = [f"r{i}" for i in range(n)]
        manifest = pd.DataFrame(
            {
                "run_id": run_ids,
                "experiment_name": ["linreg"] * n,
                "metric_mean_logged": np.linspace(3.0, 3.4, n),
                "metric_se_logged": [0.2] * n,
                "is_reference": [i == n - 1 for i in range(n)],
                "is_best": [i == 0 for i in range(n)],
                "config_summary": ["none|none|clr|none"] * n,
            }
        )
        ranks = pd.DataFrame(
            {
                "feature": ["clr_F1"] * n,
                "reference_rank": [1] * n,
                "run_id": run_ids,
                "match_kind": ["identical"] * n,
                "matched_column": ["clr_F1"] * n,
                "rank": [1.0] * n,
            }
        )
        fig = plot_stability(
            manifest,
            ranks,
            "rmse_val",
            show=False,
        )
        ax_perf, ax_ranks = fig.axes[0], fig.axes[1]
        self.assertEqual(ax_perf.get_xlim(), ax_ranks.get_xlim())
        plt.close(fig)


class TestExplainStability(unittest.TestCase):
    @patch("ritme.explain_stability.plot_stability")
    @patch("ritme.explain_stability.build_provenance_map")
    @patch("ritme.explain_stability.compute_normalized_importance")
    @patch("ritme.explain_stability.build_tuned_model_from_result")
    @patch("ritme.explain_stability.retrain_fixed_configs")
    def test_orchestration(
        self, mock_retrain, mock_build, mock_imp, mock_prov, mock_plot
    ):
        records = _records_df()
        results = []
        for run_id, mean in [("r1", 3.1), ("r2", 3.2)]:
            result = MagicMock()
            result.error = None
            result.config = {
                "trial_config": {
                    "model": "linreg",
                    "mlflow_run_id": run_id,
                    "data_transform": None,
                }
            }
            result.metrics = {"rmse_val_mean": mean}
            results.append(result)
        mock_retrain.return_value = results
        mock_imp.return_value = _imp(["clr_F1"])
        mock_prov.return_value = {
            "clr_F1": FeatureProvenance("clr_F1", "taxon", "t0", frozenset({"F1"}))
        }
        mock_plot.return_value = MagicMock()

        exp_config = {
            "task_type": "regression",
            "target": "y",
            "group_by_column": None,
            "seed_data": 1,
            "seed_model": 2,
            "max_cuncurrent_trials": 2,
        }
        # r1's logged config used taxonomic aggregation (data_aggregation=
        # "tax_class" in _records_df()), so real usage would require a
        # taxonomy table -- supply a minimal one matching train_val's "F1".
        tax = pd.DataFrame(
            {"Taxon": ["d__A; p__B; c__C; o__D; f__E; g__F"]}, index=["1"]
        )
        manifest, importances, ranks, agreement, fig = explain_stability(
            exp_config,
            records,
            train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
            test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
            model_type="linreg",
            tax=tax,
        )

        self.assertEqual(sorted(manifest["run_id"]), ["r1", "r2"])
        self.assertTrue(
            manifest.loc[manifest["run_id"] == "r1", "is_reference"].iloc[0]
        )
        self.assertAlmostEqual(
            manifest.loc[manifest["run_id"] == "r1", "metric_dev"].iloc[0], 0.0
        )
        self.assertEqual(len(mock_retrain.call_args.args[0]), 2)

    @patch("ritme.explain_stability.plot_stability")
    @patch("ritme.explain_stability.build_provenance_map")
    @patch("ritme.explain_stability.compute_normalized_importance")
    @patch("ritme.explain_stability.build_tuned_model_from_result")
    @patch("ritme.explain_stability.retrain_fixed_configs")
    def test_band_recomputed_around_reference_not_best(
        self, mock_retrain, mock_build, mock_imp, mock_prov, mock_plot
    ):
        # r1 is best (se=0.1) but r2, the simpler reference, has se=0.5, so
        # r3 (mean=3.4) falls within 1 SE of the reference though not of
        # best. The retrained/plotted band must therefore include r3.
        records = pd.DataFrame(
            {
                "run_id": ["r1", "r2", "r3"],
                "experiment_name": ["linreg"] * 3,
                "status": ["FINISHED"] * 3,
                "params.model": ["linreg"] * 3,
                "metrics.rmse_val_mean": [3.0, 3.05, 3.4],
                "metrics.rmse_val_se": [0.1, 0.5, 0.2],
                "metrics.n_folds": [5.0] * 3,
                "metrics.nb_features": [20.0, 10.0, 15.0],
            }
        )
        results = []
        for run_id, mean in [("r1", 3.0), ("r2", 3.05), ("r3", 3.4)]:
            result = MagicMock()
            result.error = None
            result.config = {
                "trial_config": {
                    "model": "linreg",
                    "mlflow_run_id": run_id,
                    "data_transform": None,
                }
            }
            result.metrics = {"rmse_val_mean": mean}
            results.append(result)
        mock_retrain.return_value = results
        mock_imp.return_value = _imp(["clr_F1"])
        mock_prov.return_value = {
            "clr_F1": FeatureProvenance("clr_F1", "taxon", "t0", frozenset({"F1"}))
        }
        mock_plot.return_value = MagicMock()

        exp_config = {
            "task_type": "regression",
            "target": "y",
            "group_by_column": None,
            "seed_data": 1,
            "seed_model": 2,
            "max_cuncurrent_trials": 2,
        }
        manifest, importances, ranks, agreement, fig = explain_stability(
            exp_config,
            records,
            train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
            test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
            model_type="linreg",
        )

        self.assertEqual(sorted(manifest["run_id"]), ["r1", "r2", "r3"])
        self.assertTrue(
            manifest.loc[manifest["run_id"] == "r2", "is_reference"].iloc[0]
        )
        self.assertEqual(len(mock_retrain.call_args.args[0]), 3)

    @patch("ritme.explain_stability.plot_stability")
    @patch("ritme.explain_stability.build_provenance_map")
    @patch("ritme.explain_stability.compute_normalized_importance")
    @patch("ritme.explain_stability.build_tuned_model_from_result")
    @patch("ritme.explain_stability.retrain_fixed_configs")
    def test_degenerate_reference_warns(
        self, mock_retrain, mock_build, mock_imp, mock_prov, mock_plot
    ):
        # a reference whose coefficients are all zero (e.g. strong
        # regularization) has no signal to rank, so it must warn rather
        # than silently plot "all top features tie at rank 1".
        records = _records_df()
        results = []
        for run_id, mean in [("r1", 3.1), ("r2", 3.2)]:
            result = MagicMock()
            result.error = None
            result.config = {
                "trial_config": {
                    "model": "linreg",
                    "mlflow_run_id": run_id,
                    "data_transform": None,
                }
            }
            result.metrics = {"rmse_val_mean": mean}
            results.append(result)
        mock_retrain.return_value = results
        degenerate_imp = pd.DataFrame(
            {"feature": ["clr_F1", "clr_F2"], "importance": [0.0, 0.0], "rank": [1, 1]}
        )
        mock_imp.return_value = degenerate_imp
        mock_prov.return_value = {
            "clr_F1": FeatureProvenance("clr_F1", "taxon", "t0", frozenset({"F1"})),
            "clr_F2": FeatureProvenance("clr_F2", "taxon", "t0", frozenset({"F2"})),
        }
        mock_plot.return_value = MagicMock()

        exp_config = {
            "task_type": "regression",
            "target": "y",
            "group_by_column": None,
            "seed_data": 1,
            "seed_model": 2,
            "max_cuncurrent_trials": 2,
        }
        tax = pd.DataFrame(
            {"Taxon": ["d__A; p__B; c__C; o__D; f__E; g__F"]}, index=["1"]
        )
        with self.assertWarnsRegex(UserWarning, "no discriminative"):
            explain_stability(
                exp_config,
                records,
                train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                model_type="linreg",
                tax=tax,
            )

    def test_trac_without_tree_raises(self):
        records = _records_df()
        records["experiment_name"] = "trac"
        records["params.model"] = "trac"
        with self.assertRaises(ValueError):
            explain_stability(
                {
                    "task_type": "regression",
                    "target": "y",
                    "group_by_column": None,
                    "seed_data": 1,
                    "seed_model": 2,
                    "max_cuncurrent_trials": 2,
                },
                records,
                train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                model_type="trac",
            )

    def test_reference_is_trac_raises(self):
        # tree_phylo is set and data_aggregation cleared so the other two
        # guards cannot fire; only the reference-is-trac guard can.
        records = _records_df()
        records["experiment_name"] = "trac"
        records["params.model"] = "trac"
        records["params.data_aggregation"] = None
        with self.assertRaisesRegex(ValueError, "Reference trial is 'trac'"):
            explain_stability(
                {
                    "task_type": "regression",
                    "target": "y",
                    "group_by_column": None,
                    "seed_data": 1,
                    "seed_model": 2,
                    "max_cuncurrent_trials": 2,
                },
                records,
                train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                model_type="trac",
                tree_phylo="dummy_tree",
            )

    @patch("ritme.explain_stability.retrain_fixed_configs")
    def test_trac_band_trial_without_taxonomy_raises(self, mock_retrain):
        # trac needs the taxonomy as well as the tree, so a trac trial with
        # a tree but no tax must be rejected up front rather than failing
        # inside Ray and being dropped from the band.
        records = _records_df()
        records.loc[records["run_id"] == "r2", "experiment_name"] = "trac"
        records.loc[records["run_id"] == "r2", "params.model"] = "trac"
        records["params.data_aggregation"] = None
        with self.assertRaisesRegex(ValueError, "requires both"):
            explain_stability(
                {
                    "task_type": "regression",
                    "target": "y",
                    "group_by_column": None,
                    "seed_data": 1,
                    "seed_model": 2,
                    "max_cuncurrent_trials": 2,
                },
                records,
                train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                model_type="all",
                tree_phylo="dummy_tree",
            )
        mock_retrain.assert_not_called()

    @patch("ritme.explain_stability.retrain_fixed_configs")
    def test_aggregation_without_taxonomy_raises(self, mock_retrain):
        # a band trial used taxonomic aggregation but no tax table was
        # supplied; this must be rejected up front rather than crashing
        # inside Ray.
        records = _records_df()
        records.loc[records["run_id"] == "r1", "params.data_aggregation"] = "tax_genus"
        with self.assertRaisesRegex(ValueError, "taxonomic aggregation"):
            explain_stability(
                {
                    "task_type": "regression",
                    "target": "y",
                    "group_by_column": None,
                    "seed_data": 1,
                    "seed_model": 2,
                    "max_cuncurrent_trials": 2,
                },
                records,
                train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                model_type="linreg",
                tax=None,
            )
        mock_retrain.assert_not_called()

    @patch("ritme.explain_stability.plot_stability")
    @patch("ritme.explain_stability.build_provenance_map")
    @patch("ritme.explain_stability.compute_normalized_importance")
    @patch("ritme.explain_stability.build_tuned_model_from_result")
    @patch("ritme.explain_stability.retrain_fixed_configs")
    def test_failed_trial_dropped_from_every_output_table(
        self, mock_retrain, mock_build, mock_imp, mock_prov, mock_plot
    ):
        # 3 trials in-band; r1 is the simplest (fewest nb_features) so it is
        # the reference and survives retraining; r3 fails to retrain and
        # must disappear from both the manifest and importances_long.
        records = pd.DataFrame(
            {
                "run_id": ["r1", "r2", "r3"],
                "experiment_name": ["linreg", "linreg", "linreg"],
                "status": ["FINISHED", "FINISHED", "FINISHED"],
                "params.model": ["linreg", "linreg", "linreg"],
                "metrics.rmse_val_mean": [3.0, 3.05, 3.1],
                "metrics.rmse_val_se": [1.0, 1.0, 1.0],
                "metrics.n_folds": [5.0, 5.0, 5.0],
                "metrics.nb_features": [10.0, 20.0, 30.0],
            }
        )
        results = []
        for run_id, mean, has_error in [
            ("r1", 3.0, False),
            ("r2", 3.05, False),
            ("r3", 3.1, True),
        ]:
            result = MagicMock()
            result.error = RuntimeError("boom") if has_error else None
            result.config = {
                "trial_config": {
                    "model": "linreg",
                    "mlflow_run_id": run_id,
                    "data_transform": None,
                }
            }
            result.metrics = {"rmse_val_mean": mean}
            results.append(result)
        mock_retrain.return_value = results
        mock_imp.return_value = _imp(["clr_F1"])
        mock_prov.return_value = {
            "clr_F1": FeatureProvenance("clr_F1", "taxon", "t0", frozenset({"F1"}))
        }
        mock_plot.return_value = MagicMock()

        exp_config = {
            "task_type": "regression",
            "target": "y",
            "group_by_column": None,
            "seed_data": 1,
            "seed_model": 2,
            "max_cuncurrent_trials": 2,
        }
        with self.assertWarns(UserWarning):
            manifest, importances_long, ranks, agreement, fig = explain_stability(
                exp_config,
                records,
                train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                model_type="linreg",
            )

        self.assertEqual(sorted(manifest["run_id"]), ["r1", "r2"])
        self.assertNotIn("r3", manifest["run_id"].tolist())
        self.assertNotIn("r3", importances_long["run_id"].tolist())

    @patch("ritme.explain_stability.plot_stability")
    @patch("ritme.explain_stability.build_provenance_map")
    @patch("ritme.explain_stability.compute_normalized_importance")
    @patch("ritme.explain_stability.build_tuned_model_from_result")
    @patch("ritme.explain_stability.retrain_fixed_configs")
    def test_reference_trial_retrain_failure_raises(
        self, mock_retrain, mock_build, mock_imp, mock_prov, mock_plot
    ):
        # r1 is the reference; make it the one that fails to retrain while
        # r2 succeeds, so the guard fires instead of crashing downstream in
        # align_feature_ranks.
        records = _records_df()
        results = []
        for run_id, mean, has_error in [("r1", 3.1, True), ("r2", 3.2, False)]:
            result = MagicMock()
            result.error = RuntimeError("boom") if has_error else None
            result.config = {
                "trial_config": {
                    "model": "linreg",
                    "mlflow_run_id": run_id,
                    "data_transform": None,
                }
            }
            result.metrics = {"rmse_val_mean": mean}
            results.append(result)
        mock_retrain.return_value = results
        mock_imp.return_value = _imp(["clr_F1"])
        mock_prov.return_value = {
            "clr_F1": FeatureProvenance("clr_F1", "taxon", "t0", frozenset({"F1"}))
        }
        mock_plot.return_value = MagicMock()

        exp_config = {
            "task_type": "regression",
            "target": "y",
            "group_by_column": None,
            "seed_data": 1,
            "seed_model": 2,
            "max_cuncurrent_trials": 2,
        }
        with self.assertRaises(ValueError):
            explain_stability(
                exp_config,
                records,
                train_val=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                test=pd.DataFrame({"F1": [1.0], "y": [2.0]}),
                model_type="linreg",
            )


class TestCliExplainStability(unittest.TestCase):
    @patch("ritme.explain_stability.explain_stability")
    @patch("ritme.explain_stability.load_trial_records")
    @patch("ritme.explain_stability.load_experiment_config")
    @patch("ritme.explain_stability.pd.read_pickle")
    def test_writes_all_artifacts(
        self, mock_read_pickle, mock_load_config, mock_load_records, mock_api
    ):
        mock_load_config.return_value = {"tracking_uri": "mlruns"}
        mock_read_pickle.return_value = pd.DataFrame()
        mock_load_records.return_value = _records_df()
        fig = MagicMock()
        mock_api.return_value = (
            _manifest(),
            _imp(["f1"]),
            pd.DataFrame({"a": [1]}),
            pd.DataFrame({"run_id": ["r1"]}),
            fig,
        )
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "mlflow_logs.csv"), "w").close()
            cli_explain_stability(tmp, "linreg", "tv.pkl", "test.pkl")
            out_dir = os.path.join(tmp, "stability_linreg")
            for name in [
                "stability_trials.csv",
                "stability_importances.csv",
                "stability_ranks.csv",
                "stability_agreement.csv",
            ]:
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)))
            fig.savefig.assert_called_once()

    @patch("ritme.explain_stability.load_experiment_config")
    def test_wandb_experiment_raises(self, mock_load_config):
        mock_load_config.return_value = {"tracking_uri": "wandb"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                cli_explain_stability(tmp, "linreg", "tv.pkl", "test.pkl")

    def test_missing_logs_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "experiment_config.json"), "w") as f:
                f.write('{"tracking_uri": "mlruns"}')
            with self.assertRaises(ValueError):
                cli_explain_stability(tmp, "linreg", "tv.pkl", "test.pkl")
