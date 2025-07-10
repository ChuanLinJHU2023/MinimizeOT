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

def make_moons_with_ratio(n_samples=100, ratio_for_positive=0.7, noise_level=0.1):
    X, y = datasets.make_moons(n_samples=int(n_samples*2*ratio_for_positive), noise=noise_level, random_state=42)
    X_positive = X[y==1]
    y_positive = y[y==1]
    X, y = datasets.make_moons(n_samples=int(n_samples*2*(1-ratio_for_positive)), noise=noise_level, random_state=42)
    X_negative = X[y==0]
    y_negative = y[y==0]
    X = np.concatenate((X_positive, X_negative), axis=0)
    y = np.concatenate((y_positive, y_negative), axis=0)
    # new_indices=np.random.shuffle(np.arange(len(X)))
    # print(new_indices.shape)
    # return X[new_indices], y[new_indices]
    return X, y

# X, y = make_moons_with_ratio(n_samples=100, ratio_for_positive=0.9, noise_level=0.1)
# visualize_domains([X,], [y,],
#                   [f'Domain ',],
#                   x_limit=(-3, 3), y_limit=(-3, 3), with_model=None)
# print(len(X), len(y))
# print(len(X[y==0]), len(X[y==1]))


def create_domain_adaptation_problem_with_label_shift(n_samples=300, noise_level=0.1, source_ratio=0.8, target_ratio=0.2):
    X_s, y_s = make_moons_with_ratio(n_samples, source_ratio, noise_level)
    X_t, y_t = make_moons_with_ratio(n_samples, target_ratio, noise_level)
    theta = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    X_t = np.dot(X_t, rotation_matrix)
    X_t = X_t * 1.5
    return X_s, y_s, X_t, y_t
