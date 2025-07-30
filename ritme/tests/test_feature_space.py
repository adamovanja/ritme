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
from ritme.feature_space._process_train import process_train
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
from ritme.feature_space.utils import _biom_to_df, _df_to_biom


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
        self.train_val = pd.DataFrame(
            {
                "host_id": ["c", "b", "c", "a"],
                "target": [1, 2, 1, 2],
                "F0": [0.12, 0.23, 0.33, 0.44],
                "F1": [0.1, 0.2, 0.3, 0.4],
            },
            index=["SR1", "SR2", "SR3", "SR4"],
        )
        self.target = "target"
        self.host_id = "host_id"
        self.seed_data = 0
        self.tax = pd.DataFrame()

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
        # only no_feature_engineering is tested here since all individual
        # functions were tested above
        ls_ft = ["F0", "F1"]
        ft = self.train_val[ls_ft]
        mock_aggregate_features.return_value = ft
        mock_select_features.return_value = ft
        mock_transform_features.return_value = ft

        mock_enrich_features.return_value = ([], self.train_val)
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
        self._assert_called_with_df(mock_aggregate_features, ft, None, self.tax)
        self._assert_called_with_df(mock_select_features, ft, self.config, "F")
        self._assert_called_with_df(mock_transform_features, ft, None)
        self._assert_called_with_df(
            mock_enrich_features,
            self.train_val,
            ft.columns.tolist(),
            self.train_val,
            self.config,
        )
        self._assert_called_with_df(
            mock_split_data_grouped,
            self.train_val,
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
        ls_ft = ["F0", "F1"]
        ft = self.train_val[ls_ft]
        one_ft_config = self.config.copy()
        one_ft_config["data_transform"] = "ilr"

        mock_aggregate_features.return_value = ft
        mock_select_features.return_value = ft[["F0"]]
        mock_transform_features.return_value = ft[["F0"]]
        mock_enrich_features.return_value = ([], self.train_val)
        mock_split_data_grouped.return_value = (
            self.train_val.iloc[:2, :],
            self.train_val.iloc[2:, :],
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
        self._assert_called_with_df(mock_transform_features, ft[["F0"]], None)


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
        leaves = list(self.tree.tips())
        A1, a1_node_names = _create_identity_matrix_for_leaves(3, self.tax, leaves)
        np.testing.assert_array_equal(A1, np.eye(3))
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
        leaves, leaf_index_map = _get_leaves_and_index_map(self.tree)
        internal_nodes = _get_internal_nodes(self.tree)
        A2, a2_node_names = _create_matrix_for_internal_nodes(
            3, internal_nodes, leaf_index_map, self.tax
        )
        np.testing.assert_array_equal(A2, np.array([[1], [1], [0]]))
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
        np.testing.assert_array_equal(A2, np.array([[1, 0], [1, 0], [0, 1]]))
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
        ma_exp = pd.DataFrame(
            ma_exp,
            columns=leaf_names + node_taxon_names,
            index=ft_names,
        )
        ma_exp = ma_exp.astype(pd.SparseDtype("float", 0))
        ma_act = create_matrix_from_tree(self.tree, self.tax)

        assert_frame_equal(ma_exp, ma_act)

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
