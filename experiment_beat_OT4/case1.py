import numpy as np
from models import *
from problems import *
from distances import *
from scipy.spatial import distance
from visualizations import *


# Step 1: Get data
n_samples=100
X_source, y_source, X_target, y_target = create_domain_adaptation_problem(n_samples=n_samples,  theta=0, noise_level=0.1,
                                                                          horizontally_stretch=1.5,
                                                                          pos_x_interval_target=(-0.2, 1.7),
                                                                          pos_y_interval_target=(-np.inf, np.inf),
                                                                          neg_x_interval_target=(-0.2, 1.7),
                                                                          neg_y_interval_target=(-np.inf, np.inf),
                                                                          pos_x_interval_source=(1.3, np.inf),
                                                                          pos_y_interval_source=(-np.inf, np.inf),
                                                                          neg_x_interval_source=(-np.inf, 0.2),
                                                                          neg_y_interval_source=(-np.inf, np.inf),
                                                                          )




# Step 2: Visualize
visualize_domains([X_source, X_target], [y_source, y_target],
                  ['Source Domain', "Target Domain"],
                  x_limit=(-2.5, 3.5), y_limit=(-3, 3))



