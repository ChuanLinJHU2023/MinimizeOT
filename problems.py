from sklearn import datasets
import numpy as np

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