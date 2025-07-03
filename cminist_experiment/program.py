# program.py
from distances import *
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Calculate causal distance between images.')
    parser.add_argument('--img1', type=str, required=True, help='Path to first image')
    parser.add_argument('--img2', type=str, required=True, help='Path to second image')
    parser.add_argument('--scaling_param', type=int, required=True, help='Scaling parameter c')
    args = parser.parse_args()

    options = {"msg": False}

    img1 = plt.imread(args.img1)
    img2 = plt.imread(args.img2)

    dist, _ = calculate_causal_distance_between_images(img1, img2, args.scaling_param, options=options)
    print(dist)

if __name__ == "__main__":
    main()