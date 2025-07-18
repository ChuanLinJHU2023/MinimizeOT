import argparse
import matplotlib.pyplot as plt
from distances import *
from utils import *

def main(i, j):
    imgs = list()
    scaling_parameter_c = 32
    options = {"msg": True}

    # Load and process images
    for index in range(5):
        img = plt.imread(f'../image_coffees/{index}_32.png')
        img = downsample_image(img, 2)
        imgs.append(img)

    print(f"IMAGE SHAPE: {imgs[0].shape}")
    print(f"SCALING FACTOR: {scaling_parameter_c}")

    # Check for index validity
    if i >= len(imgs) or j >= len(imgs):
        print(f"Error: Index out of range. List contains {len(imgs)} images.")
        return

    dist, _ = calculate_causal_distance_between_images(imgs[i], imgs[j], scaling_parameter_c, options=options)
    print(f"DISTANCE BETWEEN {i} AND {j}: {dist}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate causal distance between images.')
    parser.add_argument('params', nargs='+', help='Parameters in the form i=0 j=5')
    args = parser.parse_args()

    # Parse parameters
    params = {}
    for param in args.params:
        key, value = param.split('=')
        params[key] = int(value)

    i = params.get('i', 0)
    j = params.get('j', 0)

    main(i, j)