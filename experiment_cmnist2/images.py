import numpy as np
import os
from utils import generate_colored_digit  # Assuming your function is in utils
from PIL import Image

# Create a directory to save the images and arrays if it doesn't exist
os.makedirs("output_images", exist_ok=True)
os.makedirs("output_arrays", exist_ok=True)

# List of parameters for each call
params = [
    (3, (255, 255, 25), (150, 150, 150), "Three1"),
    (3, (50, 255, 255), (255, 255, 50), "Three2"),
    (3, (128, 60, 128), (255, 165, 60), "Three3"),
    (4, (255, 255, 25), (150, 150, 150), "Four1"),
    (4, (50, 255, 255), (255, 255, 50), "Four2"),
    (4, (128, 60, 128), (255, 165, 60), "Four3"),
    (5, (255, 255, 25), (150, 150, 150), "Five1"),
    (5, (50, 255, 255), (255, 255, 50), "Five2"),
    (5, (128, 60, 128), (255, 165, 60), "Five3")
]

for num, color1, color2, title in params:
    # Generate the numpy array
    array = generate_colored_digit(num, color1, color2, save_title=title)

    # Save the numpy array as a .npy file
    np.save(f"output_arrays/{title}.npy", array)

    # Convert the numpy array to an image and save it
    # Assuming array is in a format compatible with Image.fromarray
    image = Image.fromarray(array)
    image.save(f"output_images/{title}.png")