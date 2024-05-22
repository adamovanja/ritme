import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase
from skbio import TreeNode

from q2_ritme.feature_space._process_train import create_matrix_from_tree


class TestProcessTrain(TestPluginBase):
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
