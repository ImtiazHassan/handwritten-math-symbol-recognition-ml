"""Tests for image preprocessing."""

from src.preprocessing import preprocess_image


def test_preprocess_image_preserves_placeholder_input():
    image = object()
    assert preprocess_image(image) is image
