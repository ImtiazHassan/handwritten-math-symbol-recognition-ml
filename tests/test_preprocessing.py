# Tests for image preprocessing.

import numpy as np
import pytest
from PIL import Image, ImageDraw

from support.preprocessing import preprocess_image, load_grayscale, to_ink_mask


def make_line_image():
    # White 100x60 image with a black diagonal line, used by several tests.
    img = Image.new("RGB", (100, 60), "white")
    ImageDraw.Draw(img).line([(10, 10), (90, 50)], fill="black", width=3)
    return img


# Placeholder (until preprocess_image is implemented)
def test_preprocess_image_preserves_placeholder_input():
    image = object()
    assert preprocess_image(image) is image


# step 1: load_grayscale

def test_load_grayscale_returns_2d_uint8():
    # Pillow size is (width, height) ; numpy shape is (height, width)
    arr = load_grayscale(make_line_image())
    assert arr.shape == (60, 100)
    assert arr.dtype == np.uint8
    assert arr.min() == 0       # black line
    assert arr.max() == 255     # white bg


def test_load_grayscale_rejects_wrong_type():
    with pytest.raises(TypeError):
        load_grayscale(42)


def test_load_grayscale_rejects_empty_image():
    with pytest.raises(ValueError):
        load_grayscale(Image.new("RGB", (0, 0)))


# step 2: to_ink_mask

def test_to_ink_mask_marks_dark_pixels_as_ink():
    mask = to_ink_mask(load_grayscale(make_line_image()))
    assert mask.shape == (60, 100)
    assert mask.dtype == bool
    assert not mask[0, 0]      # corner is white background
    assert mask[10, 10]        # start of the line is ink
    assert mask.sum() > 0      # some ink was found


def test_to_ink_mask_respects_threshold():
    # A mid-grey pixel (100) is ink at threshold 128, background at threshold 50
    grey = np.full((5, 5), 100, dtype=np.uint8)
    assert to_ink_mask(grey, threshold=128).all()
    assert not to_ink_mask(grey, threshold=50).any()