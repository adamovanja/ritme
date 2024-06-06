from unittest.mock import patch

import biom
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase
from scipy.stats.mstats import gmean
from skbio import TreeNode
from skbio.stats.composition import ilr

from q2_ritme.feature_space._process_trac_specific import (
    _create_identity_matrix_for_leaves,
    _create_matrix_for_internal_nodes,
    _get_internal_nodes,
    _get_leaves_and_index_map,
    create_matrix_from_tree,
)
from q2_ritme.feature_space._process_train import process_train
from q2_ritme.feature_space.aggregate_features import (
    agg_microbial_fts_taxonomy,
    aggregate_ft_by_taxonomy,
    aggregate_microbial_features,
    extract_taxonomic_entity,
)
from q2_ritme.feature_space.transform_features import (
    PSEUDOCOUNT,
    alr,
    presence_absence,
    transform_microbial_features,
)
from q2_ritme.feature_space.utils import _biom_to_df, _df_to_biom


class TestUtils(TestPluginBase):
    package = "q2_ritme.tests"

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


class TestTransformMicrobialFeatures(TestPluginBase):
    package = "q2_ritme.tests"

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


class TestAggregateMicrobialFeatures(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.ft = pd.read_csv(
            self.get_data_path("example_feature_table.tsv"), sep="\t", index_col=0
        )
        self.tax = pd.read_csv(
            self.get_data_path("example_taxonomy.tsv"), sep="\t", index_col=0
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


class TestProcessTrain(TestPluginBase):
    package = "q2_ritme.tests"

    def setUp(self):
        super().setUp()
        self.config = {
            "data_transform": None,
            "data_alr_denom_idx": False,
            "data_aggregation": None,
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

    @patch("q2_ritme.feature_space._process_train.aggregate_microbial_features")
    @patch("q2_ritme.feature_space._process_train.transform_microbial_features")
    @patch("q2_ritme.feature_space._process_train.split_data_by_host")
    def test_process_train(
        self, mock_split_data_by_host, mock_transform_features, mock_aggregate_features
    ):
        # Arrange
        ls_ft = ["F0", "F1"]
        ft = self.train_val[ls_ft]
        mock_aggregate_features.return_value = ft
        mock_transform_features.return_value = ft
        mock_split_data_by_host.return_value = (
            self.train_val.iloc[:2, :],
            self.train_val.iloc[2:, :],
        )

        # Act
        X_train, y_train, X_val, y_val, ft_col = process_train(
            self.config,
            self.train_val,
            self.target,
            self.host_id,
            self.tax,
            self.seed_data,
        )

        # Assert
        self._assert_called_with_df(mock_aggregate_features, ft, None, self.tax)
        self._assert_called_with_df(mock_transform_features, ft, None, False)
        self._assert_called_with_df(
            mock_split_data_by_host,
            self.train_val[[self.host_id, self.target] + ls_ft],
            "host_id",
            0.8,
            0,
        )


class TestProcessTracSpecific(TestPluginBase):
    package = "q2_ritme.tests"

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
        ma_act = create_matrix_from_tree(self.tree, self.tax)

        assert_frame_equal(ma_exp, ma_act)
