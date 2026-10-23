# Project Plan — Handwritten Text & Mathematical Symbol Recognition

**Course:** CSE445 Machine Learning, North South University
**Team:** Ahnaf · Sonali · Imtiaz · Muntaha
**Duration:** 8 weeks (Oct 21 – Dec 13, 2026)
**Update 1:** Oct 24, 1:00 pm · **Final submission:** around Dec 13–17

---

## 1. Project summary

We build a machine learning application that takes an image of handwriting and
recognizes it. The user picks one of two modes:

| Mode | What it recognizes | Model trained on |
|---|---|---|
| **Text** | Handwritten English letters and digits | EMNIST |
| **Math** | Handwritten mathematical symbols (`+ − × ÷ =`, `√`, `∫`, …) | CROHME |

The app shows the prediction, a confidence score, and lets the user correct mistakes.

### Scope

**Minimum viable project (must have)**
- Upload an image of handwriting.
- Preprocess the image (grayscale, threshold, crop, resize).
- Recognize single handwritten letters and digits (text mode).
- Recognize a defined set of single mathematical symbols (math mode).
- Show the result with a confidence score and allow correction.
- Evaluate both models on a separate test set that was never used for training.

**Stretch goals (only if time allows)**
- Simple one-line equations, e.g. `x + 5 = 10`, read left to right.
- Short words, e.g. `Hello`.
- LaTeX output for simple expressions, e.g. `a^2 + b^2 = c^2`.

**Out of scope**
- Fractions, nested roots, matrices and other 2-D equation layouts.
- Solving equations, handwriting-to-speech, mobile apps.

> Recognizing one symbol is much easier than understanding a full equation. A full
> equation also needs the positions of symbols (superscripts, fractions, roots).
> We say this clearly in the report and the demo.

---

## 2. Team and responsibilities

Every member writes and tests their own module, pushes their own commits, and
presents their own work.

| Member | Role | Module / files | Dataset | End deliverable |
|---|---|---|---|---|
| **Ahnaf** | Text recognition (ML lead) | `support/text_recognizer.py` | EMNIST | Trained character model, training code, dataset description, evaluation results |
| **Sonali** | Math symbol recognition (math ML lead) | `support/math_recognizer.py` | CROHME | Trained symbol model, symbol label mapping, training code, evaluation results |
| **Imtiaz** | Image processing and integration | `support/preprocessing.py`, `support/pipeline.py` | — | Preprocessing pipeline, integration code, shared interface, pipeline tests |
| **Muntaha** | Application, testing and documentation | `main.py` (Streamlit app), `README.md` | — | Working web app, system tests, screenshots, README, demo |

Rules:
- Every member tests their own module; Muntaha coordinates the final whole-system tests.
- Imtiaz helps with integration but must not become a bottleneck — everyone fixes their
  own interface problems.
- Report and slides are written by **all four**: each member writes the sections about
  their own module.

---

## 3. Architecture and shared interfaces

```
Image upload (main.py, Streamlit)
        │
        ▼
preprocess_image(image)            ← support/preprocessing.py   (Imtiaz)
        │
        ▼
run_pipeline(image, mode)          ← support/pipeline.py        (Imtiaz)
        │
   ┌────┴─────────────┐
   ▼                  ▼
recognize_text()   recognize_math()
(Ahnaf, EMNIST)    (Sonali, CROHME)
   └────┬─────────────┘
        ▼
Result shown in the app: prediction · confidence · correction field
```

### Function interfaces (agree before coding)

| Function | File | Input | Output |
|---|---|---|---|
| `preprocess_image(image)` | `support/preprocessing.py` | raw image | processed image |
| `recognize_text(image)` | `support/text_recognizer.py` | processed image | `{"text": str, "confidence": float}` |
| `recognize_math(image)` | `support/math_recognizer.py` | processed image | `{"label": str, "confidence": float}` |
| `run_pipeline(image, mode)` | `support/pipeline.py` | raw image, `"text"` or `"math"` | the recognizer's dictionary; `ValueError` for an unknown mode |

`main.py` calls only `run_pipeline()`. It never loads models directly.

Until the real models are ready, each function returns **placeholder (mock)
predictions** so others can build against it. Placeholder values are never
reported as results.

### Image format — TO DECIDE (owner: Imtiaz, before Update 1)

- [ ] Functions pass a **PIL image** or a **NumPy array**?
- [ ] Grayscale only? Pixel values **0–255** or **0–1**?
- [ ] Ink colour convention (white ink on black, or black on white) — must match each dataset.
- [ ] Who resizes? EMNIST images are **28×28**; CROHME render size is chosen by Sonali.
      Either `preprocess_image` takes a size, or each recognizer resizes itself.

---

## 4. Repository structure

Course-required structure (the plan's `src/` is our `support/`, its `app.py` is our `main.py`).

```
handwritten-math-symbol-recognition-ml/
├── main.py              # Entry point: Streamlit app  →  streamlit run main.py
├── README.md            # Description, setup, how to run, method, results
├── requirements.txt     # Python libraries (Python 3.12)
├── plan.md              # This plan
├── .gitignore
├── data/                # Dataset download instructions only (no datasets committed)
├── support/             # All supporting Python modules
│   ├── preprocessing.py     # Imtiaz
│   ├── pipeline.py          # Imtiaz
│   ├── text_recognizer.py   # Ahnaf
│   ├── math_recognizer.py   # Sonali
│   └── reconstruction.py    # Stretch goal: equation reconstruction
├── tests/               # pytest tests, one file per module
├── models/              # Model download/saving instructions (no large model files committed)
├── examples/            # Sample input images and recorded results
├── docs/                # architecture.md, weekly-progress.md, screenshots/
├── others/              # Required deliverables (see section 9)
└── .github/workflows/   # Automatic tests on every push / pull request
```

Never commit: datasets, trained model files, `.venv/`, `__pycache__/`.

---

## 5. Git workflow

- Each member works **only on their own branch**: `Ahnaf`, `Sonali`, `Imtiaz`, `Muntaha`
  (type the names with exactly these capital letters).
- Start of every work session:
  ```
  git fetch
  git merge origin/main
  ```
- End of a task:
  ```
  git add <files>
  git commit -m "Short, meaningful message"
  git push
  ```
- Code reaches `main` **only through a pull request**, reviewed by one teammate,
  with the automatic tests passing (green ✓).
- Avoid editing someone else's file. Shared files (`requirements.txt`, `main.py`,
  `support/pipeline.py`) — tell the owner first.

---

## 6. Environment setup

Everyone uses **Python 3.12**.

```
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
pip install -r requirements.txt
python -m pytest                     # all tests should pass
```

---

## 7. Timeline (8 weeks)

Each week ends with something that can be shown, not only research.

### Week 1 — Oct 21–27: Planning, setup, interfaces
| Member | Tasks |
|---|---|
| All | Repo, branches, Python 3.12 environment, agree on interfaces |
| Ahnaf | Choose the EMNIST split, load it, look at samples, text module skeleton |
| Sonali | Choose supported symbol classes, study CROHME format, math module skeleton |
| Imtiaz | Define and share the interface + image format; first version of `preprocess_image` |
| Muntaha | Streamlit skeleton (upload, preview, mode selection), first README |

**Update 1 — Oct 24, 1:00 pm.** Each member presents their own part:
- Ahnaf: EMNIST choice, sample images, planned model.
- Sonali: CROHME choice, chosen symbols, planned model.
- Imtiaz: preprocessing before/after images, the shared interface.
- Muntaha: Streamlit page with upload and mock prediction, README.

### Week 2 — Oct 28–Nov 3: Datasets and baselines
| Member | Tasks |
|---|---|
| Ahnaf | Character-recognition baseline on EMNIST |
| Sonali | Convert CROHME symbols to images; symbol baseline |
| Imtiaz | Image validation, grayscale, thresholding, resizing + tests |
| Muntaha | App shows mock predictions and error messages |

### Week 3 — Nov 4–10: Model development
| Member | Tasks |
|---|---|
| Ahnaf | Train the character model (CNN), record accuracy and common errors |
| Sonali | Train the symbol model (CNN), record accuracy and common errors |
| Imtiaz | Compare raw vs preprocessed images on both models |
| Muntaha | App connected to Imtiaz's mock pipeline |

### Week 4 — Nov 11–17: Improve recognition
| Member | Tasks |
|---|---|
| Ahnaf / Sonali | Tune settings, test confusing pairs (`−` vs `=`, `x` vs `×`, `1` vs `l`), freeze model v1 |
| Imtiaz | Fix image-quality problems; tilted / low-contrast images |
| Muntaha | Output display, correction field, LaTeX rendering |

### Week 5 — Nov 18–24: Integration
| Member | Tasks |
|---|---|
| Ahnaf / Sonali | Expose final `recognize_text` / `recognize_math`, support integration |
| Imtiaz | Connect both models to the pipeline |
| Muntaha | Connect the app to the real pipeline |

### Week 6 — Nov 25–Dec 1: Complete the MVP
| All | End-to-end image → output, test with real handwriting from people outside the team, fix integration bugs. Start stretch goals only if the MVP works. |
|---|---|

### Week 7 — Dec 2–8: Evaluation and report
| All | Evaluate on held-out test sets, error analysis, charts, screenshots, limitations, draft final report and slides |
|---|---|

### Week 8 — Dec 9–13: Final submission
| All | Code freeze, finish README, final report PDF, final PPTX, 1-minute demo video, rehearse. **Submit Dec 13–17.** |
|---|---|

---

## 8. Evaluation

| Metric | Measures |
|---|---|
| Character accuracy | % of letters/digits recognized correctly |
| Symbol accuracy | % of math symbols classified correctly |
| Precision, recall, F1 (per class) | Which classes the model struggles with |
| Confusion matrix | Which symbols are mixed up with each other |
| Expression exact-match | % of whole equations correct (only if the stretch goal is reached) |
| Processing time | Time to return one prediction |
| Robustness | Performance on real handwriting and different image quality |

Rules:
- The **test set is never used** for training or for choosing settings.
- Report only results we actually measured. No promised accuracy.
- Placeholder / mock outputs are never presented as results.

---

## 9. Deliverables checklist

**Repository**
- [ ] `main.py` runs the app
- [ ] `README.md`: title, problem, objectives, features, architecture, datasets (with citations
      and licenses), setup and run instructions, screenshots, results, team responsibilities,
      limitations, future work
- [ ] `requirements.txt` complete
- [ ] `data/README.md` explains how to download EMNIST and CROHME
- [ ] Tests pass on GitHub

**`others/` folder**
- [ ] Update presentation (PPTX)
- [ ] Update report (PDF)
- [ ] Final presentation (PPTX)
- [ ] Final report (PDF)
- [ ] 1-minute demo video

**Per member**
- [ ] Ahnaf: character model, training code, dataset description, evaluation results
- [ ] Sonali: symbol model, symbol mapping, training code, evaluation results
- [ ] Imtiaz: preprocessing pipeline, integration code, tests
- [ ] Muntaha: web app, system tests, screenshots, README

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CROHME is stored as InkML pen strokes, not images — conversion takes time | Start conversion in Week 2; limit to the chosen symbol classes |
| Full equations need segmentation and layout — hard | Keep equations as a stretch goal; MVP is single characters/symbols |
| Different Python / library versions on different laptops | Everyone uses Python 3.12 and `requirements.txt` |
| Real photos look different from training images | Preprocessing matches each dataset's format; test with real handwriting |
| Waiting for other members' modules | Agreed interfaces + mock functions from Week 1 |
| Merge conflicts | Own branch, own files, daily `git merge origin/main`, small pull requests |
| Large files pushed to GitHub | `.gitignore` covers data, models, venv |

---

## 11. Progress tracker

Update every week (or after each pull request).

| Member | Current task | Status | Blocker |
|---|---|---|---|
| Ahnaf | | | |
| Sonali | | | |
| Imtiaz | Interface + `preprocess_image` | In progress | |
| Muntaha | | | |
