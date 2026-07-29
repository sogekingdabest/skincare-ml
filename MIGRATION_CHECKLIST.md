# Standalone Repository Migration

## Prepared

- [x] Repository structure created.
- [x] Current notebook copied without modifying the source.
- [x] Dataset dependencies documented.
- [x] Experimental model card created.
- [x] Android compatibility mismatch documented.
- [x] Existing Keras, TFLite and Android asset hashes recorded.
- [x] Large training artifacts excluded from Git.
- [x] GPL-3.0 license copied from the Android repository.

## Before the first public release

- [x] Choose the final repository name and public description.
- [x] Replace `YOUR_KAGGLE_USERNAME` in the Kaggle metadata.
- [ ] Decide whether the Kaggle notebook will be public.
- [ ] Verify and record the exact licenses of all four attached dataset versions.
- [ ] Add the required ISIC and dataset citations.
- [ ] Decide whether to preserve the original notebook's Git history.
- [ ] Review notebook outputs and remove noisy runtime traces.
- [ ] Pin the exact package versions from a successful Kaggle run.

## First model release

- [ ] Confirm which model should be the canonical Android model.
- [ ] Run a clean Kaggle execution from top to bottom.
- [ ] Verify patient-level train/validation separation.
- [ ] Save the notebook version and Kaggle Docker image identifier.
- [ ] Export the matching Keras and TensorFlow Lite artifacts.
- [ ] Update `artifacts/manifest.json`.
- [ ] Add an inference/parity fixture.
- [ ] Compare Python, TFLite and Android predictions.
- [ ] Publish model binaries as release assets.

## Android repository cleanup

Perform this only after the standalone repository has been created and
verified:

- [ ] Link the ML repository from the Android README.
- [ ] Add a metadata manifest next to the deployed Android `.tflite`.
- [ ] Record the deployed model version and hash with every analysis result.
- [ ] Remove `NoteBook-Model-AI` from the Android repository.
- [ ] Confirm that Android still builds and loads its existing model.
