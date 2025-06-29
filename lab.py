import numpy as np
from scipy.spatial import distance

# First set of points
set1 = np.array([
    [1, 2],
    [3, 4]
])

# Second set of points
set2 = np.array([
    [5, 6],
    [7, 8]
])

# Compute all pairwise Euclidean distances between set1 and set2
distances = distance.cdist(set1, set2, metric='minkowski', p=2)**2

print(distances)