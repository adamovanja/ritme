import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from qiime2.plugin.testing import TestPluginBase
from skbio import TreeNode

from q2_ritme.feature_space._process_train import (
    _create_matrix_from_tree,
    derive_matrix_a,
)


class TestProcessTrain(TestPluginBase):
    package = "q2_ritme.test"

    def setUp(self):
        super().setUp()
        self.tree = self._build_example_tree()

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
        ma_act, a_names_act = _create_matrix_from_tree(self.tree)

        assert_array_equal(ma_exp, ma_act)
        self.assertEqual(a_names_act, ["n0"])

    def test_derive_matrix_a(self):
        ft_act = ["F1", "F2", "F3"]
        tax_act = ["tax1", "tax2", "tax3"]
        tax = pd.DataFrame(
            {"Feature ID": ft_act, "Taxon": tax_act, "Confidence": 3 * [0.9]}
        )
        a_act = derive_matrix_a(self.tree, tax, ft_act)

        self.assertEqual(a_act.columns.tolist(), tax_act + ["n0"])
