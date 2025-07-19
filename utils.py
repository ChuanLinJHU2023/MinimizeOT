import numpy as np
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

    # Save or show the image based on save_title
    if save_title and isinstance(save_title, str):
        # Save the image to a file
        plt.imsave(save_title+".png", colored_img)
    else:
        # Display the image
        plt.imshow(colored_img)
        plt.axis('off')
        plt.show()
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