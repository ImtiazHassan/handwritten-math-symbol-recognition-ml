"""Tests for mathematical symbol recognition."""




"""Tests for mathematical symbol recognition."""

import numpy as np
import pytest

from support.math_recognizer import recognize_math, INPUT_SIZE


def test_valid_image():
    image = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    result = recognize_math(image)

    assert isinstance(result, dict)
    assert isinstance(result["label"], str)
    assert isinstance(result["confidence"], float)


def test_invalid_input():
    with pytest.raises(TypeError):
        recognize_math([[0, 1], [1, 0]])


def test_invalid_shape():
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        recognize_math(image)
