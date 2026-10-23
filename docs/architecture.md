# Architecture

Owner: Imtiaz (preprocessing and integration). Agreed by the team on Oct 23, 2026.

## Data flow

```
User uploads an image (main.py, Streamlit)          ← Muntaha
        │  raw image (file path, uploaded file, or PIL image)
        ▼
run_pipeline(image, mode)                            ← Imtiaz, support/pipeline.py
        │  picks the recognizer for "text" or "math"
        │  and asks it for its INPUT_SIZE
        ▼
preprocess_image(image, size)                        ← Imtiaz, support/preprocessing.py
        │  processed array (format below)
   ┌────┴──────────────────┐
   ▼                       ▼
recognize_text(array)   recognize_math(array)
Ahnaf, EMNIST           Sonali, CROHME
   └────┬──────────────────┘
        ▼
{"text" or "label": str, "confidence": float}  →  shown in the app
```

**Rule:** the raw image goes into `run_pipeline` only. The models only ever
receive the processed array. A model never opens or converts an image itself.

## Agreed image format

Every recognizer receives exactly this:

| Property | Value | Why |
|---|---|---|
| Type | NumPy array | Models work with numbers. PyTorch converts NumPy with `torch.from_numpy`, and scikit-learn baselines use NumPy directly. |
| Shape | `(size, size)` — one grayscale channel | Colour adds no information for handwriting; one value per pixel is enough. |
| Dtype | `float32` | The default number type in PyTorch. |
| Value range | `0.0` to `1.0` | Neural networks train more smoothly on small input values than on 0–255. |
| Colours | Ink = `1.0` (white), background = `0.0` (black) | Matches EMNIST (white ink on black). CROHME images are rendered the same way. A blank image is all zeros. |
| Position | Character centred, padded to a square, small margin | Training images are centred; inputs must look the same. |

A normal photo (dark pen on white paper) is therefore **inverted** by
`preprocess_image`.

## Function interfaces

| Function | File | Input | Output |
|---|---|---|---|
| `preprocess_image(image, size)` | `support/preprocessing.py` | raw image, target size (int) | array in the agreed format, shape `(size, size)` |
| `recognize_text(image)` | `support/text_recognizer.py` | processed array | `{"text": str, "confidence": float}` |
| `recognize_math(image)` | `support/math_recognizer.py` | processed array | `{"label": str, "confidence": float}` |
| `run_pipeline(image, mode)` | `support/pipeline.py` | raw image, `"text"` or `"math"` | the recognizer's dictionary; `ValueError` for an unknown mode |

Each recognizer module declares its own input size as a constant, for example
`INPUT_SIZE = 28` in `support/text_recognizer.py`. The pipeline reads that
constant and passes it to `preprocess_image`.

`confidence` is a value from 0 to 1. Until real models exist, recognizers
return placeholder predictions; these are never reported as results.

## What each member needs to know

| Member | Responsibility under this format |
|---|---|
| **Imtiaz** | `preprocess_image` returns exactly the agreed format: grayscale, threshold, crop to ink, pad to square, resize, invert if needed, scale to 0–1. Validates input and raises clear errors. |
| **Ahnaf** | `recognize_text` receives a `(28, 28)` array. Add the batch and channel dimensions PyTorch needs (`(1, 1, 28, 28)`) inside the module. Declare `INPUT_SIZE = 28`. **Note:** EMNIST images loaded through torchvision are stored rotated/flipped — fix the orientation when loading the training data, otherwise real handwriting will not match. Display a few images to confirm. |
| **Sonali** | Render CROHME strokes as **white strokes on a black background**, centred, at a chosen size. Declare that size as `INPUT_SIZE` in `support/math_recognizer.py`. |
| **Muntaha** | Pass the uploaded image straight to `run_pipeline(image, mode)`. Do not preprocess in the app. Display the returned dictionary. |

## Changing this format

This format is a contract between four modules. Any change must be agreed by
the whole team and updated in this file first.
