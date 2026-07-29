# SkinCare ML

Training and export pipeline for the machine-learning component of the
SkinCare Android application.

[![Kaggle](https://img.shields.io/badge/Kaggle-daniolaetafaria-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/daniolaetafaria)
[![Android app](https://img.shields.io/badge/Android-SkinCareApp-3DDC84?logo=android&logoColor=white)](https://github.com/sogekingdabest/SkinCareApp)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

The project trains a multimodal binary classifier based on EfficientNetV2-B0.
It combines a dermoscopic image with three metadata values: approximate age,
sex and anatomical site. Training is designed to run in a Kaggle Notebook and
the resulting Keras model can be exported to TensorFlow Lite.

> [!WARNING]
> This is an educational and research portfolio project. It is not a medical
> device and must not be used to diagnose, rule out or treat melanoma.

## Relationship with the Android app

This repository owns the training, evaluation and export workflow. The Android
application consumes only a validated TensorFlow Lite release. The current
notebook model is **not** the model deployed by the Android app:

| Component | Notebook model | Android model |
| --- | --- | --- |
| Image input | `256 x 256 x 3` | `224 x 224 x 3` |
| Metadata input | Age, sex, anatomical site | None |
| Output | Sigmoid probability | Sigmoid probability |
| Artifact | `skincare_multimodal_256.tflite` | `melanoma_efficientnet_dynamic.tflite` |
| Status | Experimental | Currently deployed |

Metrics reported in this repository apply only to the notebook model. See
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the evaluation context and known
limitations.

## Project structure

```text
.
├── artifacts/
│   ├── README.md
│   └── manifest.json
├── docs/
│   ├── ANDROID_INTEGRATION.md
│   ├── DATASETS.md
│   └── MODEL_CARD.md
├── notebooks/
│   └── skincare_training.ipynb
├── .gitignore
├── kernel-metadata.json
├── LICENSE
├── README.md
└── requirements.txt
```

## Run on Kaggle

1. Authenticate once with `kaggle auth login`.
2. Push the repository root with `kaggle kernels push -p .`.
3. Follow the run with
   `kaggle kernels status daniolaetafaria/skincare-ai-multimodal-melanoma-classification`.
4. Download `metrics.json`, the resolution-specific reports and the approved
   model artifact with `kaggle kernels output`.
5. Register the final artifact and SHA-256 in `artifacts/manifest.json`.

The notebook currently targets the Kaggle Python 3 environment and records
Python `3.12.12` in its metadata. Kaggle images change over time, so record the
Kaggle notebook version and Docker image identifier for every published model.

## Notebook hygiene

The notebook committed to Git intentionally contains no execution outputs or
Kaggle runtime metadata at cell level. This keeps reviews readable and avoids
committing progress bars, CUDA warnings and large TensorFlow conversion traces.
The historical baseline remains documented in
[docs/MODEL_CARD.md](docs/MODEL_CARD.md). Each successful Kaggle run also
exports machine-readable `metrics_256.json`, `metrics_384.json` and
`metrics.json`.

After downloading an executed notebook from Kaggle, prepare it for Git with:

```bash
python scripts/prepare_notebook.py
```

This command repairs legacy text encoding, clears outputs and rebuilds the
canonical split, evaluation, parity and export cells. It does not execute or
retrain the model.

## Local environment

Kaggle is the reference execution environment. A local environment can be
created for inspection and lightweight tests:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Dataset paths in the notebook point to `/kaggle/input`. Local execution
requires either equivalent paths or a future configurable dataset root.

Kaggle profile: <https://www.kaggle.com/daniolaetafaria>

Android repository: <https://github.com/sogekingdabest/SkinCareApp>

## Model release policy

Large model files are intentionally excluded from Git. Publish approved
artifacts as versioned release assets, together with:

- SHA-256 checksum;
- source commit;
- Kaggle notebook version;
- input and output signatures;
- preprocessing contract;
- validation metrics and decision threshold;
- dataset versions and licenses.

Only an artifact that passes the parity procedure in
[docs/ANDROID_INTEGRATION.md](docs/ANDROID_INTEGRATION.md) should be copied into
the Android application.

## License

The code is distributed under GPL-3.0. Dataset images and trained artifacts may
be subject to additional terms described in [docs/DATASETS.md](docs/DATASETS.md).
