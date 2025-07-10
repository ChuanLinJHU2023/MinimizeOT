import argparse
import matplotlib.pyplot as plt
from distances import *
from utils import *


imgs = list()
scaling_parameter_c = 32
for index in range(5):
    img = plt.imread(f'../image_coffees/{index}_32.png')
    img = downsample_image(img, 2)
    imgs.append(img)
print(f"IMAGE SHAPE: {imgs[0].shape}")
print(f"SCALING FACTOR: {scaling_parameter_c}")

for i in range(0,4):
    dist, _ = calculate_wasserstein_distance_between_images(imgs[i], imgs[4], scaling_parameter_c)
    print(f"DISTANCE BETWEEN {i} AND 4: {dist}")


for i in range(1, 5):
    dist, _ = calculate_wasserstein_distance_between_images(imgs[i], imgs[0], scaling_parameter_c)
    print(f"DISTANCE BETWEEN 0 AND {i}: {dist}")