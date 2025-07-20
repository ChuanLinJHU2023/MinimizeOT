import numpy as np
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

def generate_colored_digit(number, bg_color, digit_color, save_title=None):
    # Load MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Filter images of the specified digit
    digit_images = x_train[y_train == number]

    # Randomly select an image
    img = digit_images[np.random.randint(len(digit_images))]

    # Normalize the image
    img_norm = img / 255.0

    # Create a colored background
    colored_img = np.ones((28, 28, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)

    # Create a mask for the digit pixels
    mask = img_norm > 0.5  # Threshold to distinguish digit pixels

    # Apply digit color to the digit pixels
    for c in range(3):
        colored_img[:, :, c][mask] = digit_color[c]
    return colored_img


def downsample_image(image, factor):
    # Calculate the new size
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    # Resize the image
    downsampled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return np.array(downsampled_image)


def print_format_string(text, total_length):
    text_length = len(text)
    if text_length > total_length:
        print(text[:total_length])
        return

    # Calculate the number of "#" symbols on each side
    side_length = (total_length - text_length) // 2

    # Handle odd total length when the difference isn't even
    left_hashes = "#" * side_length
    right_hashes = "#" * (total_length - text_length - side_length)

    # Construct and print the final string
    result = left_hashes + text + right_hashes
    print(result)


def combine_images(
        image_directory,
        image_titles=None,
        layout=(1, 3),
        combine_image_name="combined.png",
        font_size=36
):
    """
    Combines images into a grid with titles overlaid, using matplotlib.

    Parameters:
        image_directory (str): Path to the directory containing images.
        image_titles (list or None): List of titles for each image. If None, filenames are used.
        layout (tuple): Tuple (rows, cols) defining grid layout.
        combine_image_name (str): Output filename for the combined image.
        font_size (int): Font size for the titles.
    """
    # List all image files in the directory
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(valid_exts)]
    image_files.sort()

    rows, cols = layout
    total_slots = rows * cols

    if len(image_files) < total_slots:
        raise ValueError("Not enough images to fill the specified layout.")

    # Prepare titles
    if image_titles is None:
        image_titles = [os.path.splitext(f)[0] for f in image_files]
    else:
        if len(image_titles) < len(image_files):
            # Fill remaining titles with filenames
            remaining = len(image_files) - len(image_titles)
            image_titles.extend([os.path.splitext(f)[0] for f in image_files[len(image_titles):]])
        elif len(image_titles) > len(image_files):
            # Truncate to match images
            image_titles = image_titles[:len(image_files)]

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    for idx, ax in enumerate(axes_flat):
        if idx >= len(image_files):
            ax.axis('off')
            continue
        img_path = os.path.join(image_directory, image_files[idx])
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')
        # Add title overlay with adjustable font size
        title = image_titles[idx]
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
        ax.text(
            0.05, 0.95, title,
            fontsize=font_size,
            va='top', ha='left',
            bbox=bbox_props,
            transform=ax.transAxes
        )

    plt.tight_layout()
    plt.savefig(combine_image_name, dpi=300)
    plt.close()
    print(f"Combined image saved as {combine_image_name}")
