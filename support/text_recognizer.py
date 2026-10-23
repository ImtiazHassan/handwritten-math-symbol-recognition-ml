"""A very simple handwritten digit recognizer.

Important: this file does not train a machine-learning model yet. Instead, it
uses a collection of example digit shapes written by hand in the code. The
program compares a new image with those shapes and chooses the closest one.

The process is:

1. Accept an image.
2. Convert the image into a grid of dark and light cells.
3. Crop away empty space around the digit.
4. Resize the digit to the same 5-by-7 size as our examples.
5. Compare every cell with each known digit shape.
6. Return the digit with the most matching cells.

This simple approach makes the code easy to understand, but a real
handwriting system would normally use a trained machine-learning model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# A Bitmap is a grid of True/False values.
# True means "this cell contains ink"; False means "this cell is background".
Bitmap = list[list[bool]]

# These are the digit examples used for comparison.
#
# Each digit has 7 strings, and each string has 5 characters. Together they
# form a 5-by-7 grid. In each string:
#   - "1" means the digit is drawn in that position.
#   - "0" means that position should be empty.
#
# For example, the first and last rows of 0 are "01110", which draws the
# rounded top and bottom of a zero.
_DIGIT_TEMPLATES = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}


@dataclass(frozen=True)
class CharacterPrediction:
    """The answer and a simple measure of how closely it matched.

    ``character`` is the digit we selected. ``confidence`` is not a
    probability from a trained model. It is simply the percentage of the
    35 grid cells that matched the selected built-in digit shape.
    """

    character: str
    confidence: float


def _to_bitmap(image: object) -> Bitmap:
    """Turn an input image into a grid of True/False values.

    The rest of the program should not need to know whether the input was
    text, a PIL image, or a list of numbers. This function gives all of those
    input types the same format before recognition begins.

    True represents a dark pixel, which is part of the character. False
    represents a light pixel, which is background.
    """
    if isinstance(image, str):
        # A string can be used as a tiny drawing, for example:
        # "01110\n10001\n01110".
        # Dots, spaces, and zeroes are treated as empty background cells.
        rows = [row for row in image.splitlines() if row.strip()]
        if not rows:
            return []
        return [[cell not in " .0" for cell in row] for row in rows]

    if hasattr(image, "convert") and hasattr(image, "size"):
        # A PIL image stores real image pixels. Convert it to grayscale so
        # every pixel has one brightness value from 0 (black) to 255 (white).
        grayscale = image.convert("L")
        width, height = grayscale.size
        pixels = list(grayscale.getdata())
        # A pixel darker than 128 is considered part of the handwriting.
        return [
            [pixels[row * width + column] < 128 for column in range(width)]
            for row in range(height)
        ]

    try:
        # A list such as [[0, 1], [1, 0]] is another possible image format.
        rows = [list(row) for row in image]  # type: ignore[union-attr]
    except TypeError as error:
        raise TypeError("image must be ASCII art, a PIL-like image, or a 2-D sequence") from error

    if not rows or not all(rows):
        return []
    values = [pixel for row in rows for pixel in row]
    numeric_values = [pixel for pixel in values if isinstance(pixel, (int, float))]
    # If the largest number is 1, we assume the image already uses only 0 and
    # 1. Otherwise, we assume normal grayscale values from 0 to 255.
    binary = bool(numeric_values) and max(numeric_values) <= 1
    bitmap = []
    for row in rows:
        converted = []
        for pixel in row:
            if isinstance(pixel, (tuple, list)):
                # For a color pixel such as (red, green, blue), use its
                # average brightness.
                pixel = sum(pixel) / len(pixel)
            if binary:
                # In a binary image, 1 means ink and 0 means background.
                converted.append(pixel > 0)
            else:
                # In a grayscale image, values below 128 are dark ink.
                converted.append(pixel < 128)
        bitmap.append(converted)
    return bitmap


def _normalize(bitmap: Bitmap, width: int = 5, height: int = 7) -> Bitmap:
    """Make different-sized drawings comparable.

    A person might draw the same digit at different sizes or leave different
    amounts of empty space around it. This function:

    1. Finds the smallest rectangle containing all dark cells.
    2. Removes the empty space outside that rectangle.
    3. Resizes the remaining drawing to a 5-by-7 grid.
    """
    ink = [
        (row_index, column_index)
        for row_index, row in enumerate(bitmap)
        for column_index, value in enumerate(row)
        if value
    ]
    if not ink:
        # No dark cells means the image is blank.
        return []
    top = min(row for row, _ in ink)
    bottom = max(row for row, _ in ink)
    left = min(column for _, column in ink)
    right = max(column for _, column in ink)
    # Removing the empty border means that the digit's position in the
    # original image does not matter as much.
    cropped = [row[left : right + 1] for row in bitmap[top : bottom + 1]]
    source_height = len(cropped)
    source_width = len(cropped[0])
    # Select a source cell for each cell in the new 5-by-7 grid. This is a
    # small, simple resize operation; it does not use a neural network.
    return [
        [
            cropped[min(source_height - 1, row * source_height // height)][
                min(source_width - 1, column * source_width // width)
            ]
            for column in range(width)
        ]
        for row in range(height)
    ]


def _template_bitmap(template: Sequence[str]) -> Bitmap:
    """Turn a digit's strings into the True/False format used for images."""
    return [[cell == "1" for cell in row] for row in template]


def predict_character(image: object) -> CharacterPrediction:
    """Recognize one isolated handwritten digit.

    This is the main recognition function. It does not learn from the input;
    it only compares the input with the ten digit templates above.

    The input can be ASCII art, a PIL image, or a rectangular 2-D sequence of
    grayscale/binary pixels. The confidence is the fraction of the 35 grid
    cells that match the selected digit template.
    """
    # First convert the image and put it into the standard 5-by-7 shape.
    normalized = _normalize(_to_bitmap(image))
    if not normalized:
        raise ValueError("cannot recognize a blank image")

    scores = {}
    for character, template in _DIGIT_TEMPLATES.items():
        expected = _template_bitmap(template)
        # Compare the input with this digit one cell at a time.
        # There are 5 * 7 = 35 cells in total.
        matches = sum(
            actual == target
            for actual_row, expected_row in zip(normalized, expected)
            for actual, target in zip(actual_row, expected_row)
        )
        scores[character] = matches / 35

    # Pick the digit whose template has the highest matching score.
    character = max(scores, key=scores.get)
    return CharacterPrediction(character, round(scores[character], 4))


def predict(image: object) -> str:
    """Return only the predicted digit.

    Use this shorter function when the caller needs an answer such as "8" and
    does not need the matching score.
    """
    return predict_character(image).character


def predict_text(images: Iterable[object]) -> str:
    """Recognize several separate character images and join the answers.

    The function expects one image per character. For example, three images
    containing 1, 2, and 5 return the string "125". It does not yet split a
    full photograph of a handwritten word into separate characters.
    """
    return "".join(predict(image) for image in images)
