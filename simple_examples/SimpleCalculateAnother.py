import numpy as np
from sklearn import datasets
from distances import *

def create_domain_adaptation_problem(n_samples=300, noise_level=0.1):
    X_s, y_s = datasets.make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
    X_t, y_t = datasets.make_moons(n_samples=n_samples, noise=noise_level, random_state=43)
    theta = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    X_t = np.dot(X_t, rotation_matrix)
    X_t = X_t * 1.5
    return X_s, y_s, X_t, y_t

X_source, y_source, X_target, y_target = create_domain_adaptation_problem(n_samples=500, noise_level=0.1)
options = {"msg": True}
calculate_causal_distance_between_datasets(X_source, y_source, X_target, y_target, class_number_n=2, order_parameter_p=2, scaling_parameter_c=2, options=options)