# Handwritten Text and Mathematical Symbol Recognition

A machine learning system that recognizes handwritten English characters and
mathematical symbols from images.

**Course:** CSE445 Machine Learning, North South University, Section 7

> **Status:** early development (Week 1 of 8). The image preprocessing module
> and the shared interface are in progress. No recognition model has been trained
> yet, so this repository reports **no accuracy results**.

---

## Overview

Handwritten input looks very different from the clean images that recognition
models learn from. Photos have uneven backgrounds, different pen colours and
stroke widths, and symbols that are not centred. This project builds a pipeline
that standardizes each input image and classifies it with a trained model.

The user uploads an image and chooses a mode:

| Mode | Recognizes | Training data |
|---|---|---|
| **Text** | Handwritten English letters and digits | [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) |
| **Math** | Handwritten mathematical symbols (`+ − × ÷ =` …) | [CROHME](https://www.isical.ac.in/~crohme/) |

The main goal is recognizing **isolated** characters and symbols. Reading full
equations (fractions, superscripts, roots) is a stretch goal.

## How it works

```
Image upload (main.py)
        │
        ▼
run_pipeline(image, mode)          support/pipeline.py
        │
        ▼
preprocess_image(image, size)      support/preprocessing.py
        │
   ┌────┴─────────────┐
   ▼                  ▼
recognize_text()   recognize_math()
(EMNIST model)     (CROHME model)
   └────┬─────────────┘
        ▼
Prediction + confidence
```

Every model receives images in one agreed format: a NumPy `float32` array of
shape `(size, size)` with values from 0 to 1, white ink (1.0) on a black
background (0.0). The full interface and the reasons behind it are in
[docs/architecture.md](docs/architecture.md).

## Current status

| Module | Owner | File(s) | Status |
|---|---|---|---|
| Repository setup, plan, interface | Mohammad Imtiaz Hassan | `plan.md`, `docs/architecture.md`, `requirements.txt` | Done |
| Image preprocessing | Mohammad Imtiaz Hassan | `support/preprocessing.py` | In progress: grayscale loading with input validation (`load_grayscale`) and thresholding (`to_ink_mask`) done and tested; cropping, padding, resizing and normalization next |
| Pipeline / integration | Mohammad Imtiaz Hassan | `support/pipeline.py` | Planned (Week 5) |
| Text recognition | Ahnaf Akib Ahmed | `support/text_recognizer.py` | Prototype: digit recognizer that compares inputs with hand-coded 5×7 templates (not a trained model yet); EMNIST model planned |
| Math symbol recognition | Tamanna Islam Sonali | `support/math_recognizer.py` | Planned |
| Web application | Shidratul Muntaha | `main.py` | Planned (Streamlit) |

Detailed timeline: [plan.md](plan.md).

## Repository structure

```
├── main.py              # Entry point (Streamlit app, planned)
├── requirements.txt     # Python dependencies (Python 3.12)
├── plan.md              # 8-week project plan and responsibilities
├── data/                # Dataset download instructions (datasets are not committed)
├── support/             # Project modules: preprocessing, recognizers, pipeline
├── tests/               # Automated tests (pytest)
├── models/              # Trained model instructions (model files are not committed)
├── examples/            # Sample inputs and recorded results
├── docs/                # Architecture and progress documents
├── others/              # Reports, presentations and demo video
└── .github/workflows/   # Runs the tests automatically on every push
```

## Setup

Requires **Python 3.12** (PyTorch does not yet support newer versions).

```bash
git clone https://github.com/ImtiazHassan/handwritten-math-symbol-recognition-ml.git
cd handwritten-math-symbol-recognition-ml
py -3.12 -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
pip install -r requirements.txt
```

## Running the tests

```bash
python -m pytest -v
```

The tests also run automatically on GitHub for every push and pull request.

## Team

| Member | ID | Role |
|---|---|---|
| Shidratul Muntaha | 2312328042 | Web application and documentation |
| Mohammad Imtiaz Hassan | 2321196642 | Image preprocessing and integration |
| Tamanna Islam Sonali | 2312735042 | Mathematical symbol recognition |
| Ahnaf Akib Ahmed | 2312304642 | Handwritten text recognition |

## Documents

- [Project plan](plan.md)
- [Architecture and interfaces](docs/architecture.md)
- Reports and presentations: [`others/`](others/)

## References

- G. Cohen, S. Afshar, J. Tapson, and A. van Schaik, "EMNIST: Extending MNIST to
  handwritten letters," *IJCNN*, 2017.
- M. Mahdavi et al., "ICDAR 2019 CROHME + TFD: Competition on recognition of
  handwritten mathematical expressions and typeset formula detection," *ICDAR*, 2019.
- AI tools (ChatGPT, Claude) were used for project planning, explanations,
  code review and documentation drafting.
