import unittest

import numpy as np
import pandas as pd

from ritme.evaluate_models import TunedModel
from ritme.feature_space.feature_provenance import (
    FeatureProvenance,
    MatchResult,
    build_provenance_map,
    match_provenance,
    matched_otu_coverage,
    same_feature_space,
    split_snapshot_suffix,
    strip_transform_prefix,
)


def _make_tmodel(data_config, tax, selected, final_cols, model_type=None):
    tmodel = TunedModel(None, data_config, tax, "unused_path", model_type=model_type)
    tmodel.snapshot_selected_map = selected
    tmodel.final_feature_cols = final_cols
    return tmodel


class TestNameParsing(unittest.TestCase):
    def test_split_snapshot_suffix(self):
        self.assertEqual(split_snapshot_suffix("clr_F1"), ("clr_F1", "t0"))
        self.assertEqual(split_snapshot_suffix("clr_F1__t-2"), ("clr_F1", "t-2"))
        self.assertEqual(
            split_snapshot_suffix("shannon_entropy__t-1"),
            ("shannon_entropy", "t-1"),
        )

    def test_strip_transform_prefix(self):
        self.assertEqual(strip_transform_prefix("clr_F1", "clr"), "F1")
        self.assertEqual(strip_transform_prefix("pa_g__Blautia", "pa"), "g__Blautia")
        self.assertEqual(strip_transform_prefix("rank_F1", "rank"), "F1")
        self.assertEqual(strip_transform_prefix("alr_F1", "alr"), "F1")
        self.assertEqual(strip_transform_prefix("F1", None), "F1")
        # non-transform names pass through untouched even when a transform is set
        self.assertEqual(
            strip_transform_prefix("shannon_entropy", "clr"), "shannon_entropy"
        )


class TestBuildProvenanceMap(unittest.TestCase):
    def setUp(self):
        self.tax = pd.DataFrame(
            {
                "Taxon": [
                    "d__A; p__B; c__C; o__D; f__L; g__Blautia",
                    "d__A; p__B; c__C; o__D; f__L; g__Blautia",
                    "d__A; p__B; c__C; o__D; f__M; g__Dorea",
                ]
            },
            index=["F1", "F2", "F3"],
        )

    def test_raw_features_no_aggregation(self):
        tmodel = _make_tmodel(
            {"data_aggregation": None, "data_transform": "clr"},
            self.tax,
            {"t0": ["F1", "F3"]},
            ["clr_F1", "clr_F3", "shannon_entropy"],
        )
        prov = build_provenance_map(tmodel)
        self.assertIsInstance(prov["clr_F1"], FeatureProvenance)
        self.assertEqual(prov["clr_F1"].kind, "taxon")
        self.assertEqual(prov["clr_F1"].otus, frozenset({"F1"}))
        self.assertEqual(prov["shannon_entropy"].kind, "diversity")

    def test_aggregated_features(self):
        tmodel = _make_tmodel(
            {"data_aggregation": "tax_genus", "data_transform": None},
            self.tax,
            {"t0": ["g__Blautia", "g__Dorea"]},
            ["g__Blautia", "g__Dorea"],
        )
        prov = build_provenance_map(tmodel)
        self.assertEqual(prov["g__Blautia"].otus, frozenset({"F1", "F2"}))
        self.assertEqual(prov["g__Dorea"].otus, frozenset({"F3"}))

    def test_lumped_column_members(self):
        tmodel = _make_tmodel(
            {
                "data_aggregation": "tax_genus",
                "data_selection": "abundance_topi",
                "data_transform": None,
            },
            self.tax,
            {"t0": ["g__Blautia", "F_low_abun"]},
            ["g__Blautia", "F_low_abun"],
        )
        prov = build_provenance_map(tmodel)
        self.assertEqual(prov["F_low_abun"].kind, "lumped")
        # everything in the aggregation universe that did not survive selection
        self.assertEqual(prov["F_low_abun"].otus, frozenset({"F3"}))

    def test_lumped_without_aggregation_uses_taxonomy_universe(self):
        tmodel = _make_tmodel(
            {
                "data_aggregation": None,
                "data_selection": "abundance_topi",
                "data_transform": None,
            },
            self.tax,
            {"t0": ["F1", "F_low_abun"]},
            ["F1", "F_low_abun"],
        )
        prov = build_provenance_map(tmodel)
        self.assertEqual(prov["F_low_abun"].otus, frozenset({"F2", "F3"}))

    def test_ilr_and_metadata(self):
        tmodel = _make_tmodel(
            {"data_aggregation": None, "data_transform": "ilr"},
            self.tax,
            {"t0": ["F1", "F2", "F3"]},
            ["ilr_0", "ilr_1", "age", "site_bern"],
        )
        prov = build_provenance_map(tmodel)
        self.assertEqual(prov["ilr_0"].kind, "balance")
        self.assertEqual(prov["ilr_0"].otus, frozenset())
        self.assertEqual(prov["age"].kind, "metadata")

    def test_snapshot_suffix(self):
        tmodel = _make_tmodel(
            {"data_aggregation": None, "data_transform": "clr"},
            self.tax,
            {"t0": ["F1"], "t-1": ["F1"]},
            ["clr_F1", "clr_F1__t-1"],
        )
        prov = build_provenance_map(tmodel)
        self.assertEqual(prov["clr_F1"].snapshot, "t0")
        self.assertEqual(prov["clr_F1__t-1"].snapshot, "t-1")
        self.assertEqual(prov["clr_F1__t-1"].otus, frozenset({"F1"}))


def _fp(column, kind, otus, snapshot="t0"):
    return FeatureProvenance(column, kind, snapshot, frozenset(otus))


class TestSameFeatureSpace(unittest.TestCase):
    def test_identical_and_differing(self):
        a = {
            "data_aggregation": "tax_genus",
            "data_transform": "clr",
            "data_selection": None,
            "data_selection_i": None,
            "data_selection_q": None,
            "data_selection_t": None,
        }
        b = dict(a)
        self.assertTrue(same_feature_space(a, b))
        b["data_transform"] = "pa"
        self.assertFalse(same_feature_space(a, b))


class TestMatchProvenance(unittest.TestCase):
    def test_identical_name(self):
        target = _fp("clr_F1", "taxon", {"F1"})
        trial = {"clr_F1": _fp("clr_F1", "taxon", {"F1"})}
        result = match_provenance(target, trial, identical_space=False)
        self.assertIsInstance(result, MatchResult)
        self.assertEqual(result.kind, "identical")
        self.assertEqual(result.columns, ("clr_F1",))

    def test_same_name_different_members_is_not_identical(self):
        # F_low_abun/F_low_var hold whatever each trial's selection
        # discarded, so the same column name in two trials can describe
        # disjoint feature sets.
        target = _fp("F_low_abun", "lumped", {"F2", "F3"})
        trial = {
            "F_low_abun": _fp("F_low_abun", "lumped", {"F1"}),
            "clr_F3": _fp("clr_F3", "taxon", {"F3"}),
        }
        result = match_provenance(target, trial, identical_space=False)
        self.assertNotEqual(result.kind, "identical")

    def test_same_name_same_members_is_identical(self):
        target = _fp("F_low_abun", "lumped", {"F2", "F3"})
        trial = {"F_low_abun": _fp("F_low_abun", "lumped", {"F2", "F3"})}
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "identical")

    def test_equal_otus_under_different_transform_is_identical(self):
        # data_transform is a tuned categorical, so a band routinely holds
        # trials differing only by prefix: clr_F1 and pa_F1 are the same
        # taxon, not one contained in a coarser other.
        target = _fp("clr_F1", "taxon", {"F1"})
        trial = {"pa_F1": _fp("pa_F1", "taxon", {"F1"})}
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "identical")
        self.assertEqual(result.columns, ("pa_F1",))

    def test_target_without_known_otus_is_not_attributable(self):
        # a lumped column built without a taxonomy has empty otus; the
        # empty set is a subset of everything, so every comparison would
        # match vacuously and order-dependently.
        target = _fp("F_low_abun", "lumped", set())
        trial = {
            "clr_F1": _fp("clr_F1", "taxon", {"F1"}),
            "clr_F2": _fp("clr_F2", "taxon", {"F2"}),
        }
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "not_attributable")
        self.assertEqual(result.columns, ())

    def test_contained_in_coarser_aggregate(self):
        target = _fp("g__Blautia", "taxon", {"F1", "F2"})
        trial = {"f__L": _fp("f__L", "taxon", {"F1", "F2", "F9"})}
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "contained")
        self.assertEqual(result.columns, ("f__L",))

    def test_split_into_finer_parts(self):
        target = _fp("f__L", "taxon", {"F1", "F2"})
        trial = {
            "g__A": _fp("g__A", "taxon", {"F1"}),
            "g__B": _fp("g__B", "taxon", {"F2"}),
        }
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "split")
        self.assertEqual(set(result.columns), {"g__A", "g__B"})

    def test_lumped(self):
        target = _fp("g__Rare", "taxon", {"F7"})
        trial = {"F_low_abun": _fp("F_low_abun", "lumped", {"F7", "F8"})}
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "lumped")

    def test_absent(self):
        target = _fp("g__Gone", "taxon", {"F9"})
        trial = {"clr_F1": _fp("clr_F1", "taxon", {"F1"})}
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "absent")
        self.assertEqual(result.columns, ())

    def test_ilr_trial_not_attributable_for_taxa(self):
        target = _fp("clr_F1", "taxon", {"F1"})
        trial = {"ilr_0": _fp("ilr_0", "balance", set())}
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "not_attributable")

    def test_metadata_matches_by_name_even_in_ilr_trial(self):
        target = _fp("shannon_entropy", "diversity", set())
        trial = {
            "ilr_0": _fp("ilr_0", "balance", set()),
            "shannon_entropy": _fp("shannon_entropy", "diversity", set()),
        }
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "identical")

    def test_balance_target_needs_identical_space(self):
        target = _fp("ilr_0", "balance", set())
        trial = {"ilr_0": _fp("ilr_0", "balance", set())}
        self.assertEqual(
            match_provenance(target, trial, identical_space=True).kind, "identical"
        )
        self.assertEqual(
            match_provenance(target, trial, identical_space=False).kind,
            "not_attributable",
        )

    def test_snapshot_must_match(self):
        # defensive: snapshot is derived from column, so build_provenance_map
        # can never produce a same-column/different-snapshot pair -- this
        # pins the guard for hand-built maps only.
        target = _fp("clr_F1", "taxon", {"F1"}, snapshot="t-1")
        trial = {"clr_F1": _fp("clr_F1", "taxon", {"F1"}, snapshot="t0")}
        result = match_provenance(target, trial, identical_space=False)
        self.assertEqual(result.kind, "absent")

    def test_trac_trial_is_never_comparable(self):
        # trac labels coefficients by log-contrast clade, not by
        # design-matrix column, so it shares no namespace -- even a
        # colliding column name must not read as a match.
        target = _fp("clr_F1", "taxon", {"F1"})
        trial = {"clr_F1": _fp("clr_F1", "opaque", set())}
        result = match_provenance(target, trial, identical_space=True)
        self.assertEqual(result.kind, "not_attributable")


class TestTracProvenance(unittest.TestCase):
    def test_trac_model_maps_every_column_to_opaque(self):
        tmodel = _make_tmodel(
            {"data_aggregation": None, "data_transform": "clr"},
            None,
            {"t0": ["F1"]},
            ["clr_F1", "clr_F2"],
            model_type="trac",
        )
        prov = build_provenance_map(tmodel)
        self.assertEqual({p.kind for p in prov.values()}, {"opaque"})


class TestMatchedOtuCoverage(unittest.TestCase):
    def test_full_and_partial_coverage(self):
        target = _fp("f__L", "taxon", {"F1", "F2", "F3"})
        trial = {
            "g__A": _fp("g__A", "taxon", {"F1"}),
            "g__B": _fp("g__B", "taxon", {"F2"}),
        }
        self.assertAlmostEqual(
            matched_otu_coverage(target, trial, ("g__A", "g__B")), 2 / 3
        )
        self.assertAlmostEqual(matched_otu_coverage(target, trial, ("g__A",)), 1 / 3)
        self.assertAlmostEqual(matched_otu_coverage(target, trial, ()), 0.0)

    def test_coverage_is_nan_without_known_otus(self):
        target = _fp("F_low_abun", "lumped", set())
        self.assertTrue(np.isnan(matched_otu_coverage(target, {}, ())))
