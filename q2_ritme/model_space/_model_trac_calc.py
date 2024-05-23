import numpy as np
from numpy import linalg


def solve_unpenalized_least_squares(cmatrices, intercept=False):
    # adapted from classo > misc_functions.py > unpenalised
    if intercept:
        A1, C1, y = cmatrices
        A = np.concatenate([np.ones((len(A1), 1)), A1], axis=1)
        C = np.concatenate([np.zeros((len(C1), 1)), C1], axis=1)
    else:
        A, C, y = cmatrices

    k = len(C)
    d = len(A[0])
    M1 = np.concatenate([A.T.dot(A), C.T], axis=1)
    M2 = np.concatenate([C, np.zeros((k, k))], axis=1)
    M = np.concatenate([M1, M2], axis=0)
    b = np.concatenate([A.T.dot(y), np.zeros(k)])
    sol = linalg.lstsq(M, b, rcond=None)[0]
    beta = sol[:d]
    return beta


def min_least_squares_solution(matrices, selected, intercept=False):
    """Minimum Least Squares solution for selected features."""
    # adapted from classo > misc_functions.py > min_LS
    X, C, y = matrices
    beta = np.zeros(len(selected))

    if intercept:
        beta[selected] = solve_unpenalized_least_squares(
            (X[:, selected[1:]], C[:, selected[1:]], y), intercept=selected[0]
        )
    else:
        beta[selected] = solve_unpenalized_least_squares(
            (X[:, selected], C[:, selected], y), intercept=False
        )

    return beta
