import numpy as np
import cv2


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