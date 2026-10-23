"""Image preprocessing utilities."""
from PIL import Image
import numpy as np


def load_grayscale(image):
    # load and image (file path or PIL image) and return a 2d grayscale numpy array
    if isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(
            f"Expected a file path or PIL image, got {type(image).__name__}"
        )
    gray = image.convert("L")
    array = np.asarray(gray)

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Image is empty (zero height or width)")

    return array


def preprocess_image(image):
    """Prepare an input image for recognition.

    The preprocessing pipeline will be implemented here.
    """
    return image
