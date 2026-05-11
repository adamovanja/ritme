import os
import unittest
from unittest.mock import patch

import biom
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from parameterized import parameterized
from scipy.stats.mstats import gmean
from skbio import TreeNode
from skbio.stats.composition import ilr

from ritme.feature_space._process_trac_specific import (
    _create_identity_matrix_for_leaves,
    _create_matrix_for_internal_nodes,
    _get_internal_nodes,
    _get_leaves_and_index_map,
    _preprocess_taxonomy_aggregation,
    create_matrix_from_tree,
)
from ritme.feature_space._process_train import (
    KFoldEngineered,
    _encode_target,
    process_train,
    process_train_kfold,
)
from ritme.feature_space.aggregate_features import (
    agg_microbial_fts_taxonomy,
    aggregate_ft_by_taxonomy,
    aggregate_microbial_features,
    extract_taxonomic_entity,
)
from ritme.feature_space.enrich_features import (
    compute_shannon_diversity,
    enrich_features,
)
from ritme.feature_space.select_features import (
    find_features_to_group_by_abundance_ith,
    find_features_to_group_by_abundance_quantile,
    find_features_to_group_by_abundance_threshold,
    find_features_to_group_by_abundance_topi,
    find_features_to_group_by_variance_ith,
    find_features_to_group_by_variance_quantile,
    find_features_to_group_by_variance_threshold,
    find_features_to_group_by_variance_topi,
    select_microbial_features,
)
from ritme.feature_space.transform_features import (
    PSEUDOCOUNT,
    _find_most_nonzero_feature_idx,
    alr,
    presence_absence,
    transform_microbial_features,
)
from ritme.feature_space.utils import (
    _biom_to_df,
    _df_to_biom,
    _extract_time_labels,
    _time_label_sort_key,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        super().setUp()
        idx_ls = ["Sample1", "Sample2", "Sample3"]
        self.true_df = pd.DataFrame(
            {"F0": [10.0, 20.0, 50.0], "F1": [20.0, 30.0, 5.0]}, index=idx_ls
        )

        ft_biom_array = np.array([[10.0, 20.0, 50.0], [20.0, 30.0, 5.0]])
        self.true_biom_table = biom.table.Table(
            ft_biom_array,
            observation_ids=["F0", "F1"],
            sample_ids=idx_ls,
        )

    def test_biom_to_df(self):
        obs_df = _biom_to_df(self.true_biom_table)
        assert_frame_equal(self.true_df, obs_df)

    def test_df_to_biom(self):
        obs_biom_table = _df_to_biom(self.true_df)
        assert obs_biom_table == self.true_biom_table

    def test_time_label_sort_key_ordering(self):
        labels = ["t-2", "t0", "t-10", "foo", "t-1", "t-x", "t2"]
        sorted_labels = sorted(labels, key=_time_label_sort_key)
        # Expected ordering:
        # - t0 first
        # - then numeric t-<n> ascending
        # - then others (non-conforming)
        self.assertEqual(sorted_labels[:4], ["t0", "t-1", "t-2", "t-10"])  # numeric
        # The rest are non-conforming and should follow (order among them not specified)
        self.assertCountEqual(sorted_labels[4:], ["foo", "t-x", "t2"])

    def test_extract_time_labels_basic(self):
        cols = [
            "F1",  # t0 unsuffixed
            "F2__t-2",
            "F3__t-1",
            "age",  # metadata, not prefixed with F
            "age__t-1",
            "misc",
        ]
        self.assertEqual(_extract_time_labels(cols, "F"), ["t0", "t-1", "t-2"])

    def test_extract_time_labels_custom_prefix(self):
        cols = ["G1", "G2__t-3", "F1__t-1"]
        self.assertEqual(_extract_time_labels(cols, "G"), ["t0", "t-3"])

    def test_extract_time_labels_no_suffix(self):
        cols = ["F1", "F2", "meta"]
        # Unsuffixed F columns represent t0
        self.assertEqual(_extract_time_labels(cols, "F"), ["t0"])

    def test_extract_time_labels_malformed(self):
        cols = ["F1__t-x", "F2__t-2", "F3__t1", "F4"]
        # F4 is unsuffixed -> t0, F2__t-2 -> t-2, plus malformed labels captured
        labels = _extract_time_labels(cols, "F")
        self.assertEqual(labels[:2], ["t0", "t-2"])
        self.assertCountEqual(labels[2:], ["t-x", "t1"])


class TestTransformMicrobialFeatures(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.ft = pd.DataFrame({"F0": [10.0, 20.0, 50.0], "F1": [20.0, 30.0, 5.0]})
        self.ft_zero = pd.DataFrame({"F0": [10.0, 20.0, 0.0], "F1": [20.0, 0.0, 5.0]})

    def _relative_abundances(self, ft):
        """Calculate relative frequency from absolute frequency"""
        return ft.div(ft.sum(axis=1), axis=0)

    def _clr_from_scratch(self, ft):
        """Calculate clr from scratch"""
        ft_gmean = gmean(ft, axis=1)
        ft_trans = pd.DataFrame(index=ft.index, columns=ft.columns, dtype=float)
        for i in ft.index:
            ft_trans.loc[i] = np.log(ft.loc[i, :] / ft_gmean[i])
        return ft_trans

    def test_find_most_nonzero_feature_idx_with_nonzero_feature(self):
        data = pd.DataFrame(
            {"F1": [0.1, 0.2, 0.7], "F2": [0, 0, 0], "F3": [0.9, 0.8, 0.3]}
        )
        expected_idx = 0
        self.assertEqual(_find_most_nonzero_feature_idx(data), expected_idx)

    def test_find_most_nonzero_feature_idx_with_all_zero_features(self):
        data = pd.DataFrame({"F1": [0, 0, 0], "F2": [0, 0, 0], "F3": [0, 0, 0]})
        with self.assertRaises(ValueError):
            _find_most_nonzero_feature_idx(data)

    def test_find_most_nonzero_feature_idx_with_empty_dataframe(self):
        data = pd.DataFrame()
        with self.assertRaises(ValueError):
            _find_most_nonzero_feature_idx(data)

    def test_alr(self):
        """Tests alr function"""
        # expected
        ft = self.ft.replace(0.0, PSEUDOCOUNT)
        denom = ft.iloc[:, 1]
        exp_ft = ft.div(denom, axis=0).drop(ft.columns[1], axis=1)

        # observed
        obs_ft = alr(ft, 1)

        assert_frame_equal(exp_ft, obs_ft)

    def test_presence_absence(self):
        """Tests presence_absence function"""
        # expected
        exp_ft = self.ft.copy()
        exp_ft[exp_ft > 0] = 1

        # observed
        obs_ft = presence_absence(self.ft)

        np.array_equal(exp_ft.values, obs_ft)

    def test_transform_presence_absence(self):
        """Tests presence_absence transformation"""
        # expected
        exp_ft = self.ft.copy()
        exp_ft[exp_ft > 0] = 1
        exp_ft = exp_ft.add_prefix("pa_")

        # observed
        obs_ft = transform_microbial_features(self.ft, "pa")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_clr(self):
        """Tests default clr transformation"""
        # expected
        ft = self._relative_abundances(self.ft)
        exp_ft = self._clr_from_scratch(ft)
        exp_ft = exp_ft.add_prefix("clr_")

        # observed
        obs_ft = transform_microbial_features(self.ft, "clr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_clr_pseudocounts(self):
        """Tests clr transformation with pseudocounts introduced"""
        # expected
        ft = self.ft_zero.replace(0.0, PSEUDOCOUNT)
        ft = self._relative_abundances(ft)
        exp_ft = self._clr_from_scratch(ft)
        exp_ft = exp_ft.add_prefix("clr_")

        # observed
        obs_ft = transform_microbial_features(self.ft_zero, "clr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_alr(self):
        """Tests alr transformation"""
        # expected
        ft = self.ft.replace(0.0, PSEUDOCOUNT)
        exp_ft = alr(ft, 0)
        exp_ft = exp_ft.add_prefix("alr_")

        # observed
        obs_ft = transform_microbial_features(self.ft, "alr", 0)

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_ilr(self):
        """Tests ilr transformation"""
        # expected
        ft = self.ft.replace(0.0, PSEUDOCOUNT)
        ft = self._relative_abundances(ft)
        exp_ft = pd.DataFrame(
            ilr(ft),
            columns=[f"ilr_{i}" for i in range(ft.shape[1] - 1)],
            index=ft.index,
        )

        # observed
        obs_ft = transform_microbial_features(self.ft, "ilr")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_rank(self):
        """Tests rank feature transformation"""
        # expected
        self.ft = pd.DataFrame({"F0": [10.0, 20.0, 50.0], "F1": [20.0, 30.0, 5.0]})
        exp_ft = pd.DataFrame({"rank_F0": [2.0, 2.0, 1.0], "rank_F1": [1.0, 1.0, 2.0]})

        # observed
        obs_ft = transform_microbial_features(self.ft, "rank")

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_none(self):
        """Tests no transformation"""
        # expected
        exp_ft = self.ft

        # observed
        obs_ft = transform_microbial_features(self.ft, None)

        assert_frame_equal(exp_ft, obs_ft)

    def test_transform_features_error(self):
        """Tests error when invalid method is provided"""
        with self.assertRaisesRegex(
            ValueError, "Method FancyTransform is not implemented yet."
        ):
            transform_microbial_features(self.ft, "FancyTransform")


class TestAggregateMicrobialFeatures(unittest.TestCase):
    def setUp(self):
        super().setUp()
        current_dir = os.path.dirname(__file__)
        ft_w_md = pd.read_csv(
            os.path.join(current_dir, "data/example_feature_table.tsv"),
            sep="\t",
            index_col=0,
        )
        self.ft = ft_w_md.drop(columns=["md2"])

        self.tax = pd.read_csv(
            os.path.join(current_dir, "data/example_taxonomy.tsv"),
            sep="\t",
            index_col=0,
        )
        self.tax_dict_class = {
            "F1": "c__Clostridia",
            "F2": "c__Clostridia",
            "F3": "c__Clostridia",
            "F4": "c__Bacilli",
            "F5": "c__Clostridia",
            "F6": "c__Bacilli",
        }
        self.tax_dict_species = {
            "F1": "s__unknown",
            "F2": "s__unknown",
            "F3": "s__uncultured_Dorea",
            "F4": "s__unknown",
            "F5": "s__Clostridium_scindens",
            "F6": "s__unknown",
        }

    def test_extract_taxonomic_entity_no_unknowns(self):
        obs_tax_dict = extract_taxonomic_entity(self.tax, "class")

        self.assertDictEqual(self.tax_dict_class, obs_tax_dict)

    def test_extract_taxonomic_entity_w_unknowns(self):
        # observed
        obs_tax_dict = extract_taxonomic_entity(self.tax, "species")

        self.assertDictEqual(self.tax_dict_species, obs_tax_dict)

    def test_aggregate_ft_by_taxonomy(self):
        # expected
        exp_ft = self.ft.copy()
        exp_ft = exp_ft.groupby(self.tax_dict_class, axis=1).sum()
        # observed
        obs_ft = aggregate_ft_by_taxonomy(self.ft, self.tax_dict_class)

        assert_frame_equal(exp_ft, obs_ft)

    def test_aggregate_ft_by_taxonomy_more_ft_than_tax(self):
        tax_dict = {f"F{i}": self.tax_dict_class[f"F{i}"] for i in range(1, 5)}

        with self.assertWarnsRegex(Warning, r".*are hence disregarded: \['F5', 'F6'\]"):
            aggregate_ft_by_taxonomy(self.ft, tax_dict)

    def test_agg_microbial_fts_taxonomy(self):
        # Define the expected feature table columns - no feature dim reduction
        # here only resorting
        exp_ft_cols = sorted(
            [
                "s__unkn_g__Subdoligranulum",
                "s__unkn_g__Ruminococcus_torques_group",
                "s__uncultured_Dorea",
                "s__unkn_g__Streptococcus",
                "s__Clostridium_scindens",
                "s__unkn_g__Granulicatella",
            ]
        )
        exp_ft = self.ft[["F5", "F3", "F6", "F2", "F4", "F1"]].copy()

        obs_ft = agg_microbial_fts_taxonomy(self.ft, "species", self.tax)

        self.assertEqual(exp_ft_cols, obs_ft.columns.tolist())
        assert_array_equal(exp_ft.values, obs_ft.values)

    def test_aggregate_microbial_features_tax_class(self):
        exp_ft = pd.DataFrame()
        exp_ft["c__Bacilli"] = self.ft[["F4", "F6"]].sum(axis=1)
        exp_ft["c__Clostridia"] = self.ft[["F1", "F2", "F3", "F5"]].sum(axis=1)

        obs_ft = aggregate_microbial_features(self.ft, "tax_class", self.tax)

        assert_frame_equal(exp_ft, obs_ft)

    def test_aggregate_microbial_features_none(self):
        exp_ft = self.ft.copy()

        obs_ft = aggregate_microbial_features(self.ft, None, self.tax)

        assert_frame_equal(exp_ft, obs_ft)

    def test_aggregate_microbial_features_method_not_available(self):
        with self.assertRaisesRegex(
            ValueError, "Method FancyMethod is not implemented yet."
        ):
            aggregate_microbial_features(self.ft, "FancyMethod", self.tax)


class TestSelectMicrobialFeatures(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.ft = pd.DataFrame(
            {
                "F1": [1, 2, 5],
                "F2": [2, 5, 0],
                "F3": [9, 9, 9],
                "F4": [10, 10, 10],
            }
        )

    @parameterized.expand(
        [(1, ["F3", "F1", "F2"]), (2, ["F1", "F2"]), (3, ["F2"]), (4, [])]
    )
    def test_find_features_to_group_abundance_topi(self, i, expected_features):
        features_to_group = find_features_to_group_by_abundance_topi(self.ft, i)
        self.assertEqual(features_to_group, expected_features)

    @parameterized.expand(
        [(1, ["F1", "F3", "F4"]), (2, ["F3", "F4"]), (3, ["F4"]), (4, [])]
    )
    def test_find_features_to_group_variance_topi(self, i, expected_features):
        features_to_group = find_features_to_group_by_variance_topi(self.ft, i)
        self.assertEqual(features_to_group, expected_features)

    @parameterized.expand(
        [(1, ["F2", "F1"]), (2, ["F2", "F1"]), (3, ["F2"]), (4, ["F2"])]
    )
    def test_find_features_to_group_abundance_ith(self, i, expected_features):
        features_to_group = find_features_to_group_by_abundance_ith(self.ft, i)
        self.assertEqual(features_to_group, expected_features)
        # for IDE test verification or debugging:
        # def test_find_features_to_group_abundance(self):
        #     i = 4
        #     features_to_group = find_features_to_group_by_abundance(self.ft, i)

        #     # Assert
        #     self.assertEqual(features_to_group, [])

    @parameterized.expand(
        [
            (1, ["F1", "F3", "F4"]),
            (2, ["F1", "F3", "F4"]),
            (3, ["F3", "F4"]),
            (4, ["F3", "F4"]),
        ]
    )
    def test_find_features_to_group_variance_ith(self, i, expected_features):
        features_to_group = find_features_to_group_by_variance_ith(self.ft, i)

        self.assertEqual(sorted(features_to_group), expected_features)

        # def test_find_features_to_group_variance_ith_test(self):
        #     i = 1
        #     features_to_group = find_features_to_group_by_variance_ith(self.ft, i)

        #     # Assert
        #     self.assertEqual(features_to_group, ["F4", "F3", "F1"])

    @parameterized.expand([(0.5, ["F1", "F2"]), (0.9, ["F1", "F2", "F3"])])
    def test_find_features_to_group_abundance_quantile(self, q, expected_features):
        features_to_group = find_features_to_group_by_abundance_quantile(self.ft, q)
        self.assertEqual(features_to_group, expected_features)

    @parameterized.expand([(0.5, ["F3", "F4"]), (0.9, ["F1", "F3", "F4"])])
    def test_find_features_to_group_variance_quantile(self, q, expected_features):
        features_to_group = find_features_to_group_by_variance_quantile(self.ft, q)
        self.assertEqual(features_to_group, expected_features)

    @parameterized.expand([(10, ["F1", "F2"]), (30, ["F1", "F2", "F3"])])
    def test_find_features_to_group_abundance_threshold(self, t, expected_features):
        features_to_group = find_features_to_group_by_abundance_threshold(self.ft, t)
        self.assertEqual(features_to_group, expected_features)

    @parameterized.expand([(3, ["F3", "F4"]), (5, ["F1", "F3", "F4"])])
    def test_find_features_to_group_variance_threshold(self, t, expected_features):
        features_to_group = find_features_to_group_by_variance_threshold(self.ft, t)
        self.assertEqual(features_to_group, expected_features)

    def test_select_microbial_features_method_none(self):
        config = {"data_selection": None}
        obs_ft = select_microbial_features(self.ft, config, "F")
        assert_frame_equal(self.ft, obs_ft)

    def test_select_microbial_features_none_grouped_one_selected(self):
        with self.assertWarnsRegex(Warning, r".* Returning original feature table."):
            config = {"data_selection": "abundance_ith", "data_selection_i": 4}
            obs_ft = select_microbial_features(self.ft, config, "F")
        assert_frame_equal(self.ft, obs_ft)

    def test_select_microbial_features_none_grouped_zero_selected(self):
        with self.assertWarnsRegex(Warning, r".* Returning original feature table."):
            config = {"data_selection": "abundance_threshold", "data_selection_t": 0.5}
            obs_ft = select_microbial_features(self.ft, config, "F")
        assert_frame_equal(self.ft, obs_ft)

    def test_select_microbial_features_unknown_method(self):
        with self.assertRaisesRegex(ValueError, r"Unknown method: FancyMethod."):
            config = {"data_selection": "FancyMethod"}
            select_microbial_features(self.ft, config, "F")

    def test_select_microbial_features_abundance_ith(self):
        # expected
        exp_ft = self.ft.copy()
        exp_ft["F_low_abun"] = self.ft[["F1", "F2"]].sum(axis=1)
        exp_ft.drop(columns=["F1", "F2"], inplace=True)

        # observed
        config = {"data_selection": "abundance_ith", "data_selection_i": 2}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_variance_ith(self):
        # expected
        exp_ft = self.ft.copy()
        ls_group = ["F3", "F4", "F1"]

        exp_ft["F_low_var"] = self.ft[ls_group].sum(axis=1)
        exp_ft.drop(columns=ls_group, inplace=True)

        # observed
        config = {"data_selection": "variance_ith", "data_selection_i": 1}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_abundance_topi(self):
        # expected
        exp_ft = self.ft.copy()
        ls_group = ["F1", "F2"]
        exp_ft["F_low_abun"] = self.ft[ls_group].sum(axis=1)
        exp_ft.drop(columns=ls_group, inplace=True)

        # observed
        config = {"data_selection": "abundance_topi", "data_selection_i": 2}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_variance_topi(self):
        # expected
        exp_ft = self.ft.copy()
        ls_group = ["F1", "F3", "F4"]

        exp_ft["F_low_var"] = self.ft[ls_group].sum(axis=1)
        exp_ft.drop(columns=ls_group, inplace=True)

        # observed
        config = {"data_selection": "variance_topi", "data_selection_i": 1}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_abundance_quantile(self):
        # expected
        exp_ft = self.ft.copy()
        ls_group = ["F1", "F2"]
        exp_ft["F_low_abun"] = self.ft[ls_group].sum(axis=1)
        exp_ft.drop(columns=ls_group, inplace=True)

        # observed
        config = {"data_selection": "abundance_quantile", "data_selection_q": 0.5}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_variance_quantile(self):
        # expected
        exp_ft = self.ft.copy()
        ls_group = ["F3", "F4"]

        exp_ft["F_low_var"] = self.ft[ls_group].sum(axis=1)
        exp_ft.drop(columns=ls_group, inplace=True)

        # observed
        config = {"data_selection": "variance_quantile", "data_selection_q": 0.5}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_abundance_threshold(self):
        # expected
        exp_ft = self.ft.copy()
        ls_group = ["F1", "F2"]
        exp_ft["F_low_abun"] = self.ft[ls_group].sum(axis=1)
        exp_ft.drop(columns=ls_group, inplace=True)

        # observed
        config = {"data_selection": "abundance_threshold", "data_selection_t": 10}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_variance_threshold(self):
        # expected
        exp_ft = self.ft.copy()
        ls_group = ["F3", "F4"]

        exp_ft["F_low_var"] = self.ft[ls_group].sum(axis=1)
        exp_ft.drop(columns=ls_group, inplace=True)

        # observed
        config = {"data_selection": "variance_threshold", "data_selection_t": 3}
        obs_ft = select_microbial_features(self.ft, config, "F")

        assert_frame_equal(exp_ft, obs_ft)

    def test_select_microbial_features_i_too_large(self):
        with self.assertWarnsRegex(
            Warning, r"Selected i=1000 is larger than number of features.*"
        ):
            config = {"data_selection": "abundance_ith", "data_selection_i": 1000}
            select_microbial_features(self.ft, config, "F")


class TestEnrichFeatures(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.train_val = pd.DataFrame(
            {
                "md1": ["a", "b", "c"],
                "md2": [1, 1, 2],
                "F1": [0.1, 0.0, 0.1],
                "F2": [0.2, 0.0, 0.1],
                "F3": [0.2, 0.6, 0.5],
                "F4": [0.5, 0.4, 0.3],
                "target": [1, 2, 1],
            },
            index=["SR1", "SR2", "SR3"],
        )

        self.train_val_t = self.train_val.copy()
        self.train_val_t["F_low_abun"] = self.train_val[["F1", "F2"]].sum(axis=1)
        self.train_val_t.drop(columns=["F1", "F2"], inplace=True)

    def test_compute_shannon_diversity(self):
        exp_entropy = pd.Series(
            [1.2206072605530174, 0.6730116650092564, 1.1682824461765626],
            index=self.train_val.index,
        )
        obs_entropy = compute_shannon_diversity(
            self.train_val, ["F1", "F2", "F3", "F4"]
        )
        assert_series_equal(exp_entropy, obs_entropy)

    def test_enrich_features_shannon(self):
        microbial_fts = ["F1", "F2", "F3", "F4"]
        # expected
        exp_df = self.train_val_t.copy()
        exp_df["shannon_entropy"] = compute_shannon_diversity(
            self.train_val, microbial_fts
        )

        # observed
        config = {"data_enrich": "shannon"}
        obs_other_ft_ls, obs_df = enrich_features(
            self.train_val, microbial_fts, self.train_val_t, config
        )

        assert_frame_equal(exp_df, obs_df)
        self.assertEqual(["shannon_entropy"], obs_other_ft_ls)

    def test_enrich_features_metadata_only_categorical(self):
        microbial_fts = ["F1", "F2", "F3", "F4"]
        enrich_w_col = ["md1"]
        config = {"data_enrich": "metadata_only", "data_enrich_with": enrich_w_col}

        exp_df = self.train_val_t.copy()
        exp_df["md1_b"] = [0.0, 1.0, 0.0]
        exp_df["md1_c"] = [0.0, 0.0, 1.0]
        exp_other_ft_ls = ["md1_b", "md1_c"]

        obs_other_ft_ls, obs_df = enrich_features(
            self.train_val, microbial_fts, self.train_val_t, config
        )

        assert_frame_equal(exp_df, obs_df)
        self.assertEqual(exp_other_ft_ls, obs_other_ft_ls)

    def test_enrich_features_metadata_only_float(self):
        microbial_fts = ["F1", "F2", "F3", "F4"]
        enrich_w_col = ["md2"]
        config = {"data_enrich": "metadata_only", "data_enrich_with": enrich_w_col}

        exp_df = self.train_val_t.copy()
        exp_df["md2"] = exp_df["md2"].astype(float)

        obs_other_ft_ls, obs_df = enrich_features(
            self.train_val, microbial_fts, self.train_val_t, config
        )

        assert_frame_equal(exp_df, obs_df)
        self.assertEqual(enrich_w_col, obs_other_ft_ls)

    def test_enrich_features_metadata_float_n_categorical(self):
        microbial_fts = ["F1", "F2", "F3", "F4"]
        enrich_w_col = ["md1", "md2"]
        config = {"data_enrich": "metadata_only", "data_enrich_with": enrich_w_col}

        exp_df = self.train_val_t.copy()
        exp_df["md1_b"] = [0.0, 1.0, 0.0]
        exp_df["md1_c"] = [0.0, 0.0, 1.0]
        exp_df["md2"] = exp_df["md2"].astype(float)
        exp_other_ft_ls = ["md1_b", "md1_c", "md2"]

        obs_other_ft_ls, obs_df = enrich_features(
            self.train_val, microbial_fts, self.train_val_t, config
        )

        assert_frame_equal(exp_df, obs_df)
        self.assertEqual(exp_other_ft_ls, obs_other_ft_ls)

    def test_enrich_features_metadata_not_fount(self):
        microbial_fts = ["F1", "F2", "F3", "F4"]
        config = {"data_enrich": "metadata_only", "data_enrich_with": ["md3"]}

        with self.assertRaisesRegex(
            ValueError, r"Feature {'md3'} not found in the training data."
        ):
            enrich_features(self.train_val, microbial_fts, self.train_val_t, config)

    def test_enrich_features_shannon_n_metadata(self):
        microbial_fts = ["F1", "F2", "F3", "F4"]
        config = {
            "data_enrich": "shannon_and_metadata",
            "data_enrich_with": ["md2"],
        }
        # expected
        exp_df = self.train_val_t.copy()
        exp_df["md2"] = exp_df["md2"].astype(float)
        exp_df["shannon_entropy"] = compute_shannon_diversity(
            self.train_val, microbial_fts
        )
        # observed
        obs_other_ft_ls, obs_df = enrich_features(
            self.train_val, microbial_fts, self.train_val_t, config
        )

        assert_frame_equal(exp_df, obs_df)
        self.assertEqual(["shannon_entropy", "md2"], obs_other_ft_ls)


class TestProcessTrain(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = {
            "data_transform": None,
            "data_aggregation": None,
            "data_selection": None,
            "data_selection_i": None,
            "data_selection_q": None,
            "data_selection_t": None,
            "data_enrich": None,
            "data_enrich_with": None,
        }
        # Single-snapshot (t0) training table with unsuffixed columns
        self.train_val = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "target": [1, 2, 1, 2],
                "covariate": [0, 1, 0, 1],
                "F0": [0.12, 0.23, 0.33, 0.44],
                "F1": [0.1, 0.2, 0.3, 0.4],
            },
            index=["SR1", "SR2", "SR3", "SR4"],
        )
        self.target = "target"
        self.host_id = "host_id"
        self.seed_data = 0
        self.tax = pd.DataFrame()
        self.strat_cols = ["covariate"]

    def _assert_called_with_df(self, mock_method, *expected_args):
        actual_args = mock_method.call_args[0]
        for expected, actual in zip(expected_args, actual_args):
            if isinstance(expected, pd.DataFrame):
                assert_frame_equal(expected, actual)
            else:
                assert expected == actual, f"Expected {expected}, but got {actual}"

    @patch("ritme.feature_space._process_train.enrich_features")
    @patch("ritme.feature_space._process_train.aggregate_microbial_features")
    @patch("ritme.feature_space._process_train.select_microbial_features")
    @patch("ritme.feature_space._process_train.transform_microbial_features")
    @patch("ritme.feature_space._process_train._split_data_grouped")
    def test_process_train_no_feature_engineering(
        self,
        mock_split_data_grouped,
        mock_transform_features,
        mock_select_features,
        mock_aggregate_features,
        mock_enrich_features,
    ):
        # With unsuffixed t0 input, snapshot slice returns columns as-is.
        raw_unsuffixed = pd.DataFrame(
            {
                "F0": self.train_val["F0"].values,
                "F1": self.train_val["F1"].values,
            },
            index=self.train_val.index,
        )
        mock_aggregate_features.return_value = raw_unsuffixed
        mock_select_features.return_value = raw_unsuffixed
        mock_transform_features.return_value = raw_unsuffixed

        # Build expected unsuffixed snapshot for enrichment call
        snap_all_unsuff = self.train_val.copy()

        snap_md_unsuff = snap_all_unsuff.drop(
            columns=["F0", "F1"]
        )  # host_id, target & covariate
        transf_plus_md_unsuff = raw_unsuffixed.join(snap_md_unsuff)

        # Enrichment returns unsuffixed snapshot; _add_suffix("t0") is a no-op
        enriched_unsuff = self.train_val.copy()
        mock_enrich_features.return_value = ([], enriched_unsuff)

        # After accumulation, split should be called on train_val_accum.
        mock_split_data_grouped.return_value = (
            self.train_val.iloc[:2, :],
            self.train_val.iloc[2:, :],
        )

        X_train, y_train, X_val, y_val = process_train(
            self.config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
        )

        # Assert
        self._assert_called_with_df(
            mock_aggregate_features, raw_unsuffixed, None, self.tax
        )
        self._assert_called_with_df(
            mock_select_features, raw_unsuffixed, self.config, "F"
        )
        self._assert_called_with_df(mock_transform_features, raw_unsuffixed, None)
        self._assert_called_with_df(
            mock_enrich_features,
            snap_all_unsuff,
            ["F0", "F1"],
            transf_plus_md_unsuff,
            self.config,
        )
        # Expect split called with unsuffixed enriched snapshot (t0 stays unsuffixed)
        enriched_suff_exp = pd.DataFrame(
            {
                "host_id": self.train_val["host_id"].values,
                "target": self.train_val["target"].values,
                "covariate": self.train_val["covariate"].values,
                "F0": self.train_val["F0"].values,
                "F1": self.train_val["F1"].values,
            },
            index=self.train_val.index,
        )
        self._assert_called_with_df(
            mock_split_data_grouped,
            enriched_suff_exp,
            "host_id",
            0.8,
            0,
        )

    @patch("ritme.feature_space._process_train.enrich_features")
    @patch("ritme.feature_space._process_train.aggregate_microbial_features")
    @patch("ritme.feature_space._process_train.select_microbial_features")
    @patch("ritme.feature_space._process_train.transform_microbial_features")
    @patch("ritme.feature_space._process_train._split_data_grouped")
    def test_process_train_one_ft_selected_no_ft_transformed(
        self,
        mock_split_data_grouped,
        mock_transform_features,
        mock_select_features,
        mock_aggregate_features,
        mock_enrich_features,
    ):
        # goal is to ensure that data_transform is set to None if only 1 feature
        # is selected
        raw_unsuffixed = pd.DataFrame(
            {
                "F0": self.train_val["F0"].values,
                "F1": self.train_val["F1"].values,
            },
            index=self.train_val.index,
        )
        one_ft_config = self.config.copy()
        one_ft_config["data_transform"] = "ilr"
        mock_aggregate_features.return_value = raw_unsuffixed
        mock_select_features.return_value = raw_unsuffixed[["F0"]]
        mock_transform_features.return_value = raw_unsuffixed[["F0"]]
        enriched_snapshot = self.train_val[
            [
                "F0",
            ]
        ].copy()
        mock_enrich_features.return_value = ([], enriched_snapshot)
        mock_split_data_grouped.return_value = (
            self.train_val.loc[self.train_val.index[:2], ["F0", "target"]],
            self.train_val.loc[self.train_val.index[2:], ["F0", "target"]],
        )

        X_train, y_train, X_val, y_val = process_train(
            one_ft_config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
        )

        # Assert ilr -> None
        mock_transform_features.assert_not_called()

    def test_process_train_alr_multi_snapshot(self):
        # Build multi-snapshot dataset (t0 unsuffixed, t-1 suffixed)
        df = pd.DataFrame(
            {
                "host_id": ["a", "b", "c", "a"],
                "host_id__t-1": ["a", "b", "c", "a"],
                "target": [1, 0, 1, 0],
                "F0": [10, 20, 30, 40],
                "F1": [5, 6, 7, 8],
                "F2": [1, 2, 3, 4],
                "F0__t-1": [11, 21, 31, 41],
                "F1__t-1": [6, 7, 8, 9],
                "F2__t-1": [2, 3, 4, 5],
            },
            index=["S1", "S2", "S3", "S4"],
        )
        cfg = self.config.copy()
        cfg["data_transform"] = "alr"
        X_train, y_train, X_val, y_val = process_train(
            cfg, df, "target", "host_id", self.tax, self.seed_data
        )
        # Expect 2 transformed microbial features per snapshot (3 -> 2 after ALR)
        # Total features: 4
        self.assertEqual(X_train.shape[1], 4)
        self.assertIn("data_alr_denom_idx_map", cfg)
        self.assertEqual(sorted(cfg["data_alr_denom_idx_map"].keys()), ["t-1", "t0"])
        # Indices should be valid within original (unsuffixed) 3-column snapshot
        for v in cfg["data_alr_denom_idx_map"].values():
            self.assertTrue(v in [0, 1, 2])

    @patch("ritme.feature_space._process_train._split_data_grouped")
    @patch("ritme.feature_space._process_train.transform_microbial_features")
    @patch("ritme.feature_space._process_train.select_microbial_features")
    @patch("ritme.feature_space._process_train.aggregate_microbial_features")
    @patch("ritme.feature_space._process_train.enrich_features")
    def test_process_train_stratify_by_passed_verbatim(
        self,
        mock_enrich_features,
        mock_aggregate_features,
        mock_select_features,
        mock_transform_features,
        mock_split_data_grouped,
    ):
        # Mock stages to keep focus on split invocation
        raw_unsuffixed = pd.DataFrame(
            {
                "F0": self.train_val["F0"].values,
                "F1": self.train_val["F1"].values,
            },
            index=self.train_val.index,
        )
        mock_aggregate_features.return_value = raw_unsuffixed
        mock_select_features.return_value = raw_unsuffixed
        mock_transform_features.return_value = raw_unsuffixed
        mock_enrich_features.return_value = ([], self.train_val)
        mock_split_data_grouped.return_value = (
            self.train_val.iloc[:2, :],
            self.train_val.iloc[2:, :],
        )
        process_train(
            self.config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
            stratify_by=self.strat_cols,
        )
        _args, _kwargs = mock_split_data_grouped.call_args
        self.assertEqual(_kwargs.get("stratify_by"), self.strat_cols)

    def test_process_train_nan_rows_preserved_in_multi_snapshot(self):
        """NaN rows in past snapshot should pass through feature engineering
        and appear as NaN in the output arrays (for XGBoost compatibility)."""
        df = pd.DataFrame(
            {
                "host_id": ["a", "a", "b", "b"],
                "target": [1, 2, 3, 4],
                "F0": [0.6, 0.7, 0.2, 0.5],
                "F1": [0.4, 0.3, 0.8, 0.5],
                # t-1: second and fourth rows have real data; first and third
                # are NaN (missing past observations)
                "host_id__t-1": ["a", "a", "b", "b"],
                "F0__t-1": [np.nan, 0.6, np.nan, 0.2],
                "F1__t-1": [np.nan, 0.4, np.nan, 0.8],
            },
            index=["S1", "S2", "S3", "S4"],
        )
        cfg = self.config.copy()
        X_train, y_train, X_val, y_val = process_train(
            cfg, df, "target", "host_id", self.tax, self.seed_data
        )
        # Output should contain NaN values from the missing past observations
        # (XGBoost handles these natively)
        all_X = np.concatenate([X_train, X_val])
        self.assertTrue(np.isnan(all_X).any(), "Expected NaN values in output")
        # Non-NaN values should be finite
        self.assertTrue(np.isfinite(all_X[~np.isnan(all_X)]).all())

    def test_process_train_clr_with_nan_rows(self):
        """CLR transform should work with NaN rows in past snapshot by
        skipping NaN rows during transformation."""
        df = pd.DataFrame(
            {
                "host_id": ["a", "a", "b", "b"],
                "target": [1, 2, 3, 4],
                "F0": [0.6, 0.7, 0.2, 0.5],
                "F1": [0.4, 0.3, 0.8, 0.5],
                "host_id__t-1": ["a", "a", "b", "b"],
                "F0__t-1": [np.nan, 0.6, np.nan, 0.2],
                "F1__t-1": [np.nan, 0.4, np.nan, 0.8],
            },
            index=["S1", "S2", "S3", "S4"],
        )
        cfg = self.config.copy()
        cfg["data_transform"] = "clr"
        # Should not raise (CLR on NaN data would fail without the NaN-row fix)
        X_train, y_train, X_val, y_val = process_train(
            cfg, df, "target", "host_id", self.tax, self.seed_data
        )
        all_X = np.concatenate([X_train, X_val])
        # NaN values should still be present for missing past rows
        self.assertTrue(np.isnan(all_X).any())
        # Non-NaN values (CLR-transformed) should be finite
        self.assertTrue(np.isfinite(all_X[~np.isnan(all_X)]).all())


class TestProcessTrainKFold(unittest.TestCase):
    """Direct unit tests for :func:`process_train_kfold` (the K-fold variant
    of :func:`process_train`). End-to-end K-fold trainable tests mock this
    function out, so without these tests the partition / shape contract is
    only exercised via the full Ray Tune integration path.
    """

    def setUp(self):
        super().setUp()
        # No engineering -> the produced ``X_full`` should equal the input
        # F-columns (modulo dtype and engineering side-effects).
        self.config = {
            "data_transform": None,
            "data_aggregation": None,
            "data_selection": None,
            "data_selection_i": None,
            "data_selection_q": None,
            "data_selection_t": None,
            "data_enrich": None,
            "data_enrich_with": None,
        }
        rng = np.random.default_rng(0)
        n_rows = 30
        # 10 hosts x 3 rows: groups are large enough for n_splits=3.
        host_ids = np.repeat(np.arange(10), 3)
        self.train_val = pd.DataFrame(
            {
                "host_id": host_ids,
                "target": rng.uniform(size=n_rows),
                "F0": rng.uniform(size=n_rows),
                "F1": rng.uniform(size=n_rows),
                "F2": rng.uniform(size=n_rows),
            },
            index=[f"SR{i}" for i in range(n_rows)],
        )
        self.tax = pd.DataFrame([])

    def test_returns_kfold_engineered_namedtuple(self):
        out = process_train_kfold(
            self.config.copy(),
            self.train_val,
            "target",
            "host_id",
            self.tax,
            seed_data=0,
            n_splits=3,
        )
        self.assertIsInstance(out, KFoldEngineered)
        # Row counts match across X_full, y_full, and the input.
        self.assertEqual(out.X_full.shape[0], len(self.train_val))
        self.assertEqual(out.y_full.shape, (len(self.train_val),))
        # Column count matches the recorded feature list.
        self.assertEqual(out.X_full.shape[1], len(out.ft_ls_used))

    def test_fold_indices_partition_input_rows(self):
        out = process_train_kfold(
            self.config.copy(),
            self.train_val,
            "target",
            "host_id",
            self.tax,
            seed_data=0,
            n_splits=3,
        )
        n = len(self.train_val)
        self.assertEqual(len(out.fold_indices), 3)
        all_val = []
        for tr_idx, va_idx in out.fold_indices:
            # Train and val of the same fold are disjoint and cover all rows.
            self.assertEqual(set(tr_idx).intersection(va_idx), set())
            self.assertEqual(set(tr_idx).union(va_idx), set(range(n)))
            all_val.extend(va_idx.tolist())
        # Across the K folds, every row appears in exactly one val set.
        self.assertEqual(sorted(all_val), list(range(n)))

    def test_seed_determinism(self):
        a = process_train_kfold(
            self.config.copy(),
            self.train_val,
            "target",
            "host_id",
            self.tax,
            seed_data=42,
            n_splits=3,
        )
        b = process_train_kfold(
            self.config.copy(),
            self.train_val,
            "target",
            "host_id",
            self.tax,
            seed_data=42,
            n_splits=3,
        )
        # Same seed -> identical fold partition.
        for (tr_a, va_a), (tr_b, va_b) in zip(a.fold_indices, b.fold_indices):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(va_a, va_b)
        # And identical engineered design matrix / target.
        np.testing.assert_array_equal(a.X_full, b.X_full)
        np.testing.assert_array_equal(a.y_full, b.y_full)


class TestEncodeTargetReuse(unittest.TestCase):
    """``_encode_target`` is called once per slice during K-fold dispatch (via
    ``process_train_kfold`` -> ``_encode_target``) and twice during single-split
    dispatch (train slice + val slice). The second call must reuse the encoder
    fit on the first call so train and val share a single label-to-int map.
    """

    def test_numeric_target_returns_floats_without_encoder(self):
        config: dict = {}
        train_val = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
        out = _encode_target(config, train_val, "target", train_val["target"])
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0, 4.0]))
        # Numeric path does NOT install an encoder.
        self.assertNotIn("_label_encoder", config)

    def test_non_numeric_target_fits_encoder_and_reuses(self):
        config: dict = {}
        train_val = pd.DataFrame({"target": ["cat", "dog", "cat", "bird"]})
        first = _encode_target(config, train_val, "target", train_val["target"])
        # An encoder is now stashed in ``config`` for downstream reuse.
        self.assertIn("_label_encoder", config)
        le_first = config["_label_encoder"]
        # Second call (e.g. on a val slice) must reuse the same encoder
        # instance -- never refit -- so train and val share a label map.
        val_slice = pd.Series(["dog", "bird"])
        second = _encode_target(config, train_val, "target", val_slice)
        self.assertIs(config["_label_encoder"], le_first)
        # Encoded values round-trip through the shared encoder.
        np.testing.assert_array_equal(
            first.astype(int), le_first.transform(train_val["target"].to_numpy())
        )
        np.testing.assert_array_equal(
            second.astype(int), le_first.transform(val_slice.to_numpy())
        )


class TestProcessTracSpecific(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tree = self._build_example_tree()
        self.tax = self._build_example_taxonomy()

    def _build_example_taxonomy(self):
        tax = pd.DataFrame(
            {
                "Taxon": [
                    "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; "
                    "f__SBR1031; g__SBR1031; s__anaerobic_digester",
                    "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; "
                    "f__SBR1031; g__SBR1031; s__uncultured_bacterium",
                    "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031",
                ],
                "Confidence": [0.9, 0.9, 0.9],
            }
        )
        tax.index = ["F1", "F2", "F3"]
        tax.index.name = "Feature ID"
        return tax

    def _build_example_tree(self):
        # Create the tree nodes with lengths
        n1 = TreeNode(name="node1")
        f1 = TreeNode(name="F1", length=1.0)
        f2 = TreeNode(name="F2", length=1.0)
        n2 = TreeNode(name="node2")
        f3 = TreeNode(name="F3", length=1.0)

        # Build the tree structure with lengths
        n1.extend([f1, f2])
        n2.extend([n1, f3])
        n1.length = 1.0
        n2.length = 1.0

        # n2 is the root of this tree
        tree = n2

        return tree

    def _build_collinear_example_tree(self):
        # Create the tree nodes with lengths
        n1 = TreeNode(name="node1")
        f1 = TreeNode(name="F1", length=1.0)
        f2 = TreeNode(name="F2", length=1.0)
        n2 = TreeNode(name="node2")
        f3 = TreeNode(name="F3", length=1.0)
        n3 = TreeNode(name="node3")
        n4 = TreeNode(name="node4")

        # Build the tree structure with lengths
        n1.extend([f1, f2])
        n2.extend([f3])
        n3.extend([n2])
        n4.extend([n1, n3])
        n1.length = 1.0
        n2.length = 1.0
        n3.length = 1.0
        n4.length = 1.0

        tree = n4

        return tree

    def test_get_leaves_and_index_map(self):
        leaves, leaf_index_map = _get_leaves_and_index_map(self.tree)
        self.assertEqual(len(leaves), 3)
        self.assertEqual(leaf_index_map, {"F1": 0, "F2": 1, "F3": 2})

    def test_get_internal_nodes(self):
        internal_nodes = _get_internal_nodes(self.tree)
        self.assertEqual(len(internal_nodes), 1)
        self.assertEqual(internal_nodes[0].name, "node1")

    def test_create_identity_matrix_for_leaves(self):
        import scipy.sparse as sp

        leaves = list(self.tree.tips())
        A1, a1_node_names = _create_identity_matrix_for_leaves(3, self.tax, leaves)
        self.assertTrue(sp.issparse(A1))
        np.testing.assert_array_equal(A1.toarray(), np.eye(3))
        self.assertEqual(
            a1_node_names,
            [
                "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; "
                "f__SBR1031; g__SBR1031; s__anaerobic_digester; otu__F1",
                "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; "
                "f__SBR1031; g__SBR1031; s__uncultured_bacterium; otu__F2",
                "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; otu__F3",
            ],
        )

    def test_create_matrix_for_internal_nodes(self):
        import scipy.sparse as sp

        leaves, leaf_index_map = _get_leaves_and_index_map(self.tree)
        internal_nodes = _get_internal_nodes(self.tree)
        A2, a2_node_names = _create_matrix_for_internal_nodes(
            3, internal_nodes, leaf_index_map, self.tax
        )
        self.assertTrue(sp.issparse(A2))
        np.testing.assert_array_equal(A2.toarray(), np.array([[1], [1], [0]]))
        self.assertEqual(
            a2_node_names,
            [
                "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; "
                "f__SBR1031; g__SBR1031"
            ],
        )

    def test_create_matrix_for_internal_nodes_collinear_cols(self):
        """Tests that collinear columns are collapsed"""
        tree_collinear = self._build_collinear_example_tree()
        leaves, leaf_index_map = _get_leaves_and_index_map(tree_collinear)
        internal_nodes = _get_internal_nodes(tree_collinear)

        A2, a2_node_names = _create_matrix_for_internal_nodes(
            3, internal_nodes, leaf_index_map, self.tax
        )
        np.testing.assert_array_equal(A2.toarray(), np.array([[1, 0], [1, 0], [0, 1]]))
        self.assertEqual(
            a2_node_names,
            [
                "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; "
                "f__SBR1031; g__SBR1031",
                "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031",
            ],
        )

    def test_create_matrix_from_tree(self):
        ma_exp = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
        )
        node_taxon_names = [
            "d__Bacteria; p__Chloroflexi; c__Anaerolineae; o__SBR1031; "
            "f__SBR1031; g__SBR1031"
        ]
        leaf_names = (self.tax["Taxon"] + "; otu__" + self.tax.index).values.tolist()
        ft_names = ["F1", "F2", "F3"]
        ma_act = create_matrix_from_tree(self.tree, self.tax)

        self.assertEqual(list(ma_act.columns), leaf_names + node_taxon_names)
        self.assertEqual(list(ma_act.index), ft_names)
        # Result must remain a sparse-dtype DataFrame (zero-storage backing).
        self.assertTrue(all(isinstance(dt, pd.SparseDtype) for dt in ma_act.dtypes))
        np.testing.assert_array_equal(ma_act.sparse.to_dense().to_numpy(), ma_exp)

    def test_preprocess_taxonomy_aggregation(self):
        # Create sample input data
        x = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
        A = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])

        # Define expected output
        X_expected = np.log(PSEUDOCOUNT + x)
        nleaves_expected = np.array([1.0, 1.0, 1.0, 2.0])
        log_geom_expected = X_expected.dot(A) / nleaves_expected

        # Call the function
        log_geom_actual, nleaves_actual = _preprocess_taxonomy_aggregation(x, A)

        # Assert the expected output
        np.testing.assert_array_equal(log_geom_actual, log_geom_expected)
        np.testing.assert_array_equal(nleaves_actual, nleaves_expected)

    def test_preprocess_taxonomy_aggregation_sparse_dataframe(self):
        # A sparse-dtype DataFrame (the actual return type of
        # `create_matrix_from_tree`) must produce the same result as the
        # equivalent dense matrix - without densifying internally.
        x = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
        A_dense = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
        )
        A_df = pd.DataFrame(A_dense).astype(pd.SparseDtype("float", 0))

        log_geom_dense, nleaves_dense = _preprocess_taxonomy_aggregation(x, A_dense)
        log_geom_sparse, nleaves_sparse = _preprocess_taxonomy_aggregation(x, A_df)

        np.testing.assert_allclose(log_geom_sparse, log_geom_dense)
        np.testing.assert_array_equal(nleaves_sparse, nleaves_dense)

    def test_create_matrix_from_tree_high_dim_stays_sparse(self):
        # Builds a balanced binary-ish tree with 256 leaves to exercise the
        # high-dimensional path. The result must stay sparse-dtype - the
        # original implementation densified A (~num_leaves^2) here.
        import scipy.sparse as sp

        num_leaves = 256
        leaves = [TreeNode(name=f"L{i}", length=1.0) for i in range(num_leaves)]

        # Balanced binary internal-node structure.
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    parent = TreeNode(name=f"n_{len(next_level)}_{i}", length=1.0)
                    parent.extend([current_level[i], current_level[i + 1]])
                    next_level.append(parent)
                else:
                    next_level.append(current_level[i])
            current_level = next_level
        tree = current_level[0]

        tax = pd.DataFrame(
            {
                "Taxon": [f"d__Bacteria; p__P; c__C{i}" for i in range(num_leaves)],
                "Confidence": [0.9] * num_leaves,
            },
            index=[f"L{i}" for i in range(num_leaves)],
        )
        tax.index.name = "Feature ID"

        a_df = create_matrix_from_tree(tree, tax)
        # Public contract: sparse-dtype DataFrame with leaves as the row index.
        self.assertTrue(all(isinstance(dt, pd.SparseDtype) for dt in a_df.dtypes))
        self.assertEqual(a_df.shape[0], num_leaves)

        # The sparse representation must round-trip through scipy.sparse with
        # nnz << num_leaves**2 (the cost the original dense path was paying).
        a_coo = a_df.sparse.to_coo()
        self.assertTrue(sp.issparse(a_coo))
        self.assertLess(a_coo.nnz, num_leaves * num_leaves)

        # Aggregation must work on the sparse-dtype DataFrame directly.
        x = np.random.RandomState(0).rand(4, num_leaves)
        log_geom, nleaves = _preprocess_taxonomy_aggregation(x, a_df)
        self.assertEqual(log_geom.shape, (4, a_df.shape[1]))
        self.assertEqual(nleaves.shape, (a_df.shape[1],))
