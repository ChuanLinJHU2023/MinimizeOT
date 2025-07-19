# program1.py
from distances import *
import matplotlib.pyplot as plt
import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Calculate causal distance between images.')
    parser.add_argument('--img1', type=str, required=True, help='Path to first image')
    parser.add_argument('--img2', type=str, required=True, help='Path to second image')
    parser.add_argument('--scaling_param', type=int, required=True, help='Scaling parameter c')
    args = parser.parse_args()

    options = {"msg": True}

    img1 = plt.imread(args.img1)
    img2 = plt.imread(args.img2)
    img1 = downsample_image(img1, 2)
    img2 = downsample_image(img2, 2)
    img1 += 0.5
    img2 += 0.5
    # print(img1.shape)
    # print(np.max(img1))
    # print(np.min(img1))
    # print(img2.shape)
    # print(np.max(img2))
    # print(np.min(img2))
    dist, _ = calculate_causal_distance_between_images(img1, img2, args.scaling_param, options=options)
    print(dist)

if __name__ == "__main__":
    main()