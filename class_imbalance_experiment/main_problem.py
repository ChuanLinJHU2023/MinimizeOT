from visualizations import *
from problems import *
import numpy as np
from models import *
from problems import *
from distances import *
from scipy.spatial import distance
from visualizations import *


# Step 1: Get data
X_source, y_source, X_target, y_target = create_domain_adaptation_problem_with_label_shift(n_samples=100,
                                                                noise_level=0.1, source_ratio=0.7, target_ratio=0.3)


print(len(y_source==1))
print(len(y_target==1))

visualize_domains([X_source, X_target], [y_source, y_target],
                  [f'Source Domain ',
                   f"Target Domain"],
                  x_limit=(-3, 3), y_limit=(-3, 3), with_model=None)
