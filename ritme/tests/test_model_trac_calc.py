import unittest

import numpy as np
from numpy import linalg

from ritme.model_space._model_trac_calc import (
    min_least_squares_solution,
    solve_unpenalized_least_squares,
)


class TestModelTracCalc(unittest.TestCase):
    def setUp(self):
        """Initialize variables used across multiple tests."""
        self.A1 = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
        )
        self.X = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
        )
        self.y = np.array([1.0, 2.0, 3.0])
        self.C1 = np.zeros((0, self.A1.shape[1]))

    def test_solve_unpenalized_ls_no_intercept(self):
        cmatrices = (self.A1, self.C1, self.y)
        beta = solve_unpenalized_least_squares(cmatrices, intercept=False)

        # Compare with numpy.linalg.lstsq
        expected_beta = linalg.lstsq(self.A1, self.y, rcond=None)[0]
        np.testing.assert_allclose(beta, expected_beta, atol=1e-6)

    def test_solve_unpenalized_ls_with_intercept(self):
        cmatrices = (self.A1, self.C1, self.y)
        beta = solve_unpenalized_least_squares(cmatrices, intercept=True)

        # Build augmented A for comparison
        A = np.concatenate([np.ones((len(self.A1), 1)), self.A1], axis=1)
        expected_beta = linalg.lstsq(A, self.y, rcond=None)[0]
        np.testing.assert_allclose(beta, expected_beta, atol=1e-6)

    def test_min_least_squares_solution_no_intercept(self):
        matrices = (self.X, self.C1, self.y)
        selected = np.array([True, False, True, False])
        beta = min_least_squares_solution(matrices, selected, intercept=False)

        # Expected: zeros except at selected indices
        X_selected = self.X[:, selected]
        expected_beta_selected = linalg.lstsq(X_selected, self.y, rcond=None)[0]
        expected_beta = np.zeros(self.X.shape[1])
        expected_beta[selected] = expected_beta_selected
        np.testing.assert_allclose(beta, expected_beta, atol=1e-6)

    def test_min_least_squares_solution_with_intercept(self):
        matrices = (self.X, self.C1, self.y)
        selected = np.array([True, True, False, True, False])
        beta = min_least_squares_solution(matrices, selected, intercept=selected[0])

        # Expected: zeros except at selected indices
        expected_beta = np.zeros(len(selected))
        expected_beta[selected] = solve_unpenalized_least_squares(
            (self.X[:, selected[1:]], self.C1[:, selected[1:]], self.y),
            intercept=selected[0],
        )
        np.testing.assert_allclose(beta, expected_beta, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
