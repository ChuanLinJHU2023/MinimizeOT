import matplotlib.pyplot as plt
from distances import *
from utils import *

scaling_parameter_c = 128
options = [('MIPGap', 0.2), ('TimeLimit', 30), ('MIPFocus', 1)]

img = plt.imread('../image_cats/A32.jpeg')
img = downsample_image(img,2)
img_A = np.array(img)

img = plt.imread('../image_cats/B32.jpeg')
img = downsample_image(img,2)
img_B = np.array(img)

dist, _ = calculate_causal_distance_between_images(img_A, img_B, scaling_parameter_c, options=options)
print(dist)

