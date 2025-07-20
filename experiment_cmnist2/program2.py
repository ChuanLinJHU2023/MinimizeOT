import numpy as np

from distances import *
from utils import *
img1 = np.load("output_arrays/Three1.npy")
img2 = np.load("output_arrays/Three3.npy")
downsample_factor = 2
scaling_parameter = 32
img1 = downsample_image(img1, downsample_factor)
img2 = downsample_image(img2, downsample_factor)
dist, _ = calculate_causal_distance_between_images(img1, img2, scaling_parameter, options=None)
