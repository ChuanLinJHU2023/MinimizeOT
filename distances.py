import pulp
import numpy as np
from scipy.spatial import distance
import ot


def calculate_causal_distance(Matrix1, Matrix2, costs, options=None):
    """
    Calculate Wasserstein distance between two discrete distributions using linear programming.
    Note that we transport from P(X^, Y^)  to P(X, Y)
    In other words, Matrix 1 is P(X^, Y^) while Matrix 2 is P(X, Y)
    All elements in Matrix 1 and Matrix 2 must be non-negative. But their sum can be not 1

    Parameters:
    - Matrix1: probabilities for the first distribution (size M x I) where M = |X^| and I = |Y^|
    - Matrix2: probabilities for the second distribution (size N x J) where N = |X| and J = |Y|
    - costs: 4D list or array of costs/cost matrix between points (size M x I x N x J)
    - options: options for solver For example: {"msg":False, "gapRel":0.25}

    Returns:
    - causal_distance: the minimal cost (scalar)
    - transport_plan: the optimal transportation plan matrix (numpy array)
    """
    assert np.all(Matrix1 >= 0)
    assert np.all(Matrix2 >= 0)
    Matrix1 = np.array(Matrix1)
    Matrix2 = np.array(Matrix2)
    costs = np.array(costs)
    Matrix1 = Matrix1 / np.sum(Matrix1)
    Matrix2 = Matrix2 / np.sum(Matrix2)
    M, I = Matrix1.shape
    N, J = Matrix2.shape
    assert (M, I, N, J) == costs.shape

    # Step 1: Initialize Problem
    prob = pulp.LpProblem("Causal_Distance", pulp.LpMinimize)

    # Step 2: Initialize Variables
    T = {}
    for m in range(M):
        for i in range(I):
            # This is a speed up trick. If P(X^, Y^)==0, P~(X^, Y^, X, Y) in transport plan must be 0. We don't need a variable for it
            if Matrix1[m,i]==0:
                continue
            for n in range(N):
                for j in range(J):
                    if Matrix2[n,j]==0:
                        continue
                    T[(m, i, n, j)] = pulp.LpVariable(f"T_{m}_{i}_{n}_{j}", lowBound=0)

    # Step 3: Initialize Objective---minimize total transportation cost
    prob += pulp.lpSum([costs[m, i, n, j] * T[(m, i, n, j)]
                        for m in range(M) for i in range(I) for n in range(N) for j in range(J)
                        if Matrix1[m,i]>0 and Matrix2[n,j]>0])

    # Step 4: Initialize Constraints---two marginal constrains and one causal constrain
    for m in range(M):
        for i in range(I):
            if Matrix1[m,i]==0:
                continue
            # Marginal Constrain 1: P~(X^ = m , Y^ = i) == P(X^ = m, Y^ = i)
            prob += (pulp.lpSum([T[(m, i, n, j)] for n in range(N) for j in range(J) if Matrix2[n,j]>0])
                     == Matrix1[m, i]
                     , f"marginal_constrain1_{m}_{i}")
    for n in range(N):
        for j in range(J):
            if Matrix2[n,j]==0:
                continue
            # Marginal Constrain 2: P~(X = n , Y = j) == P(X = n, Y = j)
            prob += (pulp.lpSum([T[(m, i, n, j)] for m in range(M) for i in range(I) if Matrix1[m,i]>0])
                     == Matrix2[n, j]
                     , f"marginal_constrain2_{n}_{j}")
    for m in range(M):
        for i in range(I):
            if Matrix1[m,i]==0:
                continue
            conditional_prob_of_i_given_m = Matrix1[m, i] / np.sum(Matrix1[m])
            for n in range(N):
                # Causal Constrain: P~(X^ = m , Y^ = i, X = n) == P~(X^ = m, X = n) * P(Y^ = i | X^ = m)
                # This is the equivalent expression of causal independence ( X ind Y^ | X^ )
                # Given X^, X is independent of Y
                prob += (pulp.lpSum([T[(m, i, n, j)] for j in range(J) if Matrix2[n,j]>0])
                    == pulp.lpSum([T[(m, i_, n, j)] for i_ in range(I) for j in range(J) if Matrix2[n,j]>0 and Matrix1[m,i_]>0]) * conditional_prob_of_i_given_m
                    , f"causality_constrain_{m}_{i}_{n}")

    # Step 4: Solve Linear Programming
    if options:
        prob.solve(pulp.GUROBI(**options))
    else:
        prob.solve(pulp.GUROBI())

    # Step5: Retrieve Results
    transport_plan = np.zeros((M, I, N, J))
    for m in range(M):
        for i in range(I):
            if Matrix1[m,i]==0:
                continue
            for n in range(N):
                for j in range(J):
                    if Matrix2[n,j]==0:
                        continue
                    transport_plan[m, i, n, j] = pulp.value(T[(m, i, n, j)])
    causal_distance = pulp.value(prob.objective)
    return causal_distance, transport_plan


def calculate_causal_distance_between_images(image1, image2, scaling_parameter_c=128, options=None):
    def get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c):
        assert m < H * W
        assert n < H * W
        assert i < C
        assert j < C
        h1 = m // W
        w1 = m % W
        h2 = n // W
        w2 = n % W
        c1 = i
        c2 = j
        res = np.abs(h1 - h2) + np.abs(w1 - w2) + (c1 != c2) * scaling_parameter_c
        return res
    assert image1.shape == image2.shape
    H, W, C = image1.shape
    Matrix1 = image1.reshape(-1, C)
    Matrix2 = image2.reshape(-1, C)
    costs = np.array([get_cost_from_minj(m, i, n, j, H, W, C, scaling_parameter_c)
                      for m in range(len(Matrix1)) for i in range(len(Matrix1[0]))
                      for n in range(len(Matrix2)) for j in range(len(Matrix2[0]))])
    costs = costs.reshape((len(Matrix1), len(Matrix1[0]), len(Matrix2), len(Matrix2[0])))
    return calculate_causal_distance(Matrix1, Matrix2, costs, options=options)


def calculate_causal_distance_between_datasets(X1, y1, X2, y2, class_number_n=2, order_parameter_p=2, scaling_parameter_c=2, options=None):
    def reduce_redundant_transport_matrix(redundant_transport_matrix, y1, y2):
        """
        Extracts a reduced transportation matrix from a higher-dimensional redundant matrix
        based on the provided index mappings y1 and y2.
        The (i,j)-th element of return transport plan is the (y1[i],i,y2[j],j)-th element of the redundant transport plan

        Parameters:
        - redundant_transport_matrix: numpy.ndarray
            A 4D array with shape (M, I, N, J), representing the redundant transport data.
        - y1: numpy.ndarray
            1D array of shape (I,), containing index mappings for the second dimension.
        - y2: numpy.ndarray
            1D array of shape (J,), containing index mappings for the fourth dimension.

        Returns:
        - new_transport_matrix: numpy.ndarray
            A 2D array of shape (I, J) representing the reduced transport matrix.
        """
        M, I, N, J = redundant_transport_matrix.shape
        assert y1.shape == (I,)
        assert y2.shape == (J,)
        i_indices = np.arange(I)
        j_indices = np.arange(J)
        new_transport_matrix = redundant_transport_matrix[
            y1[:, np.newaxis], i_indices[:, np.newaxis], y2[np.newaxis, :], j_indices[np.newaxis, :]]
        assert new_transport_matrix.shape == (I, J)
        return new_transport_matrix

    assert np.all(y1<class_number_n) and np.all(y2<class_number_n)
    M = class_number_n
    I = X1.shape[0]
    N = class_number_n
    J = X2.shape[0]
    assert y1.shape == (I,)
    assert y2.shape == (J,)
    redundant_Matrix1 = np.row_stack([y1 == class_i for class_i in range(class_number_n)])
    redundant_Matrix2 = np.row_stack([y2 == class_i for class_i in range(class_number_n)])
    costs_X = distance.cdist(X1, X2, metric='minkowski', p=order_parameter_p)**order_parameter_p
    costs_Y = (1 - np.eye(class_number_n)) * scaling_parameter_c ** order_parameter_p
    redundant_costs = costs_X.reshape(1, I, 1, J) + costs_Y.reshape(M, 1, N, 1)
    causal_distance, redundant_transport_plan = calculate_causal_distance(redundant_Matrix1, redundant_Matrix2, redundant_costs, options=options)
    return causal_distance, reduce_redundant_transport_matrix(redundant_transport_plan, y1, y2)


def calculate_wasserstein_distance(vector1, vector2, costs):
    marginal_prob1 = vector1 / vector1.sum()
    marginal_prob2 = vector2 / vector2.sum()
    ot_distance = ot.emd2(marginal_prob1, marginal_prob2, costs)
    plan = None
    return ot_distance, plan


def calculate_wasserstein_distance_between_images(image1, image2, scaling_parameter_c=128):
    def get_cost_from_ij(i, j, H, W, C, scaling_parameter_c=128):
        def get_hwc_from_i(i, H, W, C):
            assert i < H * W * C
            c = i % C
            i //= C
            w = i % W
            h = i // W
            return h, w, c
        h1, w1, c1 = get_hwc_from_i(i, H, W, C)
        h2, w2, c2 = get_hwc_from_i(j, H, W, C)
        res = np.abs(h1 - h2) + np.abs(w1 - w2) + (c1 != c2) * scaling_parameter_c
        return res
    assert image1.shape == image2.shape
    (H, W, C) = image1.shape
    vector1 = image1.flatten()
    vector2 = image2.flatten()
    costs = np.array([get_cost_from_ij(i, j, H, W, C, scaling_parameter_c) for i in range(len(vector1)) for j in range(len(vector2))])
    costs = costs.reshape((len(vector1), len(vector2)))
    return calculate_wasserstein_distance(vector1, vector2, costs)