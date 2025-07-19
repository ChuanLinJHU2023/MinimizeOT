from distances import *
from utils import *
img1="Three1.png"
img2="Three3.png"
downsample_factor = 2
scaling_parameter = 32
img1 = plt.imread(img1)
img2 = plt.imread(img2)
img1 = downsample_image(img1, downsample_factor)
img2 = downsample_image(img2, downsample_factor)
dist, _ = calculate_causal_distance_between_images(img1, img2, scaling_parameter, options=None)
