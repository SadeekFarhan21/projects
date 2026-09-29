# SignifyAI (blog copy)

Real-time ASL finger-spelling, hand-gesture and facial-emotion recognition as a PyQt5 screen overlay (screen captured with `mss`, three small Keras CNNs, OpenCV DNN face detector, MediaPipe hands).

This is a trimmed copy of https://github.com/SadeekFarhan21/SignifyAI (a fork of https://github.com/jalenfran/SignifyAI, originally named HackAI2025), taken at commit 6a1090c. It is a team project by three authors; the repo has no LICENSE file, so all rights remain with the authors.

## Who wrote what (from git history / git blame)

- Jalen Francis: model training scripts (`model_builders/`), the original README, cleanup.
- Jayson Clark: the desktop app (`app/`).
- Farhan Sadeek: the research paper (`reseach_paper/research_paper.tex`, `sample.bib`, figures), the early FER2013 notebook, `data/dataset.py` and the 2026 reproducibility cleanup (29-class ASL in code, real-PNG confusion images, fixed script paths, recompiled PDF; the confusion matrices themselves are unchanged from 2025).

## Not copied

Model checkpoints (`*.keras`), the Caffe face detector files, the compiled paper PDF (tracked upstream), `.claude/settings.local.json` (tracked upstream), Kaggle datasets, and scratch test scripts. Datasets come from Kaggle via `python data/dataset.py`; pretrained models are described in `README.upstream.md`.

## Results in this folder

- `results/param_counts.txt`: parameter counts, by rebuilding the layer stacks from `model_builders/*.py` (TensorFlow 2.21, Python 3.12) via `results/count_params.py`. Measured.
- `results/confusion_arithmetic.txt`: accuracies recomputed from the committed confusion-matrix images (`reseach_paper/figures/confusion_1.png`, `confusion_2.png`, `confusion_3.png`): ASL (old 36-class matrix, approximate) about 493/503 = 98.0%, gesture 2795/2800 = 99.82%, emotion 2448/3589 = 68.21%. Counts were read by eye from the images, not machine-extracted.
- Paper table (committed PDF, not copied): ASL 98.2%, gesture 99.2%, emotion 67.6%. The paper's prose gives different numbers and its model descriptions differ from the code, so treat the paper as unreliable where it disagrees with code.
- Caveats: the ASL "test" set is the validation set (`test_ds = val_ds`); the paper's ASL confusion matrix is from an older 36-class dataset while the code now uses 29 classes. No training was run for this copy.

The original README follows in `README.upstream.md`.
