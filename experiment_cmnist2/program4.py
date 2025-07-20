import numpy as np
from PIL import Image
import numpy as np
import gurobipy as grb
import os
import pandas as pd
from utils import *


def calculate_causal_distance_between_images(image1, image2, scaling_parameter_c = 4):
    H, W, C = image1.shape
    assert image1.shape == image2.shape
    assert C == 3
    assert H == W
    image1 = image1.reshape((-1,C))
    image2 = image2.reshape((-1,C))
    image1 = image1 / image1.sum()
    image2 = image2 / image2.sum()
    pixel = H
    pp = pixel ** 2
    cost = np.zeros((pp, 3, pp, 3), dtype=int)
    for i in range(pp):
        xi = i % pixel
        yi = i // pixel
        for ci in range(3):
            for j in range(pp):
                xj = j % pixel
                yj = j // pixel
                for cj in range(3):
                    cost[i, ci, j, cj] = abs(xi - xj) + abs(yi - yj) + (0 if ci == cj else pixel) * scaling_parameter_c

    def transport_distance(P_hat: np.ndarray, P_new: np.ndarray, causal=True):
        mod = grb.Model()
        gamma = mod.addMVar((pp, 3, pp, 3))
        ones = np.ones(pp)
        mod.addConstrs(
            sum(gamma[i, j, :, k] @ ones for k in range(3)) == P_hat[i, j] for i in range(pp) for j in
            range(3))
        mod.addConstrs(
            sum(gamma[:, k, i, j] @ ones for k in range(3)) == P_new[i, j] for i in range(pp) for j in
            range(3))
        if causal:
            for i in range(pp):
                c0, c1, c2 = P_hat[i, :]
                mod.addConstr(
                    c1 * sum(gamma[i, 0, :, k] for k in range(3)) == c0 * sum(gamma[i, 1, :, k] for k in range(3)))
                mod.addConstr(
                    c2 * sum(gamma[i, 0, :, k] for k in range(3)) == c0 * sum(gamma[i, 2, :, k] for k in range(3)))

        loss = sum(cost[i, j, :, k] @ gamma[i, j, :, k] for i in range(pp) for j in range(3) for k in range(3))
        mod.setObjective(loss, grb.GRB.MINIMIZE)
        mod.optimize()
        return mod.ObjVal
    P_hat = image1
    P_new = image2
    return transport_distance(P_hat, P_new), None



img1 = np.load("output_arrays/Three1.npy")
img2 = np.load("output_arrays/Four1.npy")
downsample_factor = 2
scaling_parameter = 16
img1 = downsample_image(img1, downsample_factor)
img2 = downsample_image(img2, downsample_factor)
dist, _ = calculate_causal_distance_between_images(img1, img2, scaling_parameter)
