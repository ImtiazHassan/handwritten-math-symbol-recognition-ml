"""Tests for the dependency-free text recognizer."""

import pytest

from src.text_recognizer import predict, predict_character, predict_text


SAMPLES = {
    "0": "01110\n10001\n10011\n10101\n11001\n10001\n01110",
    "1": "00100\n01100\n00100\n00100\n00100\n00100\n01110",
    "2": "01110\n10001\n00001\n00010\n00100\n01000\n11111",
    "5": "11111\n10000\n10000\n11110\n00001\n00001\n11110",
    "8": "01110\n10001\n10001\n01110\n10001\n10001\n01110",
}


@pytest.mark.parametrize("expected,image", SAMPLES.items())
def test_predict_recognizes_sample_digit(expected, image):
    assert predict(image) == expected


def test_predict_character_reports_confidence():
    result = predict_character(SAMPLES["8"])
    assert result.character == "8"
    assert 0.0 <= result.confidence <= 1.0


def test_predict_text_recognizes_multiple_samples():
    assert predict_text([SAMPLES["1"], SAMPLES["2"], SAMPLES["5"]]) == "125"


def test_predict_rejects_blank_image():
    with pytest.raises(ValueError, match="blank"):
        predict(".....\n.....")
