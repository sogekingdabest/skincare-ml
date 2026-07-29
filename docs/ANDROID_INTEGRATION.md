# Android Integration Contract

## Current incompatibility

The notebook export and the Android production asset are different models.
They must not be swapped by filename alone.

| Contract | Notebook export | Current Android asset |
| --- | --- | --- |
| Filename | `skincare_multimodal_256.tflite` | `melanoma_efficientnet_dynamic.tflite` |
| Image shape | `[1, 256, 256, 3]` | `[1, 224, 224, 3]` |
| Additional input | Three metadata values | None |
| Image scaling | `[0, 1]` | To be verified |
| Decision threshold | 0.2002 from notebook validation | Must be recalculated for deployed model |

## Release gate

Do not place a new model in `app/src/main/assets` until all of these checks pass:

1. Freeze a versioned Keras checkpoint.
2. Export TensorFlow Lite from that exact checkpoint.
3. Inspect and record every input/output tensor name, shape and dtype.
4. Record image colour order, resizing, scaling and metadata encoding.
5. Freeze the decision threshold from the matching validation run.
6. Run a Python/TFLite parity test on representative samples.
7. Run the same samples through the Android preprocessing and interpreter.
8. Compare probabilities within a documented tolerance.
9. Run Android unit/instrumentation tests for invalid inputs and model loading.
10. Update the artifact manifest and user-facing model version.

## Suggested parity fixture

Create a small, legally redistributable test fixture containing:

- synthetic or separately licensed test images;
- input metadata vectors;
- expected preprocessed tensors;
- expected model probabilities;
- model version and SHA-256;
- numerical tolerance.

Do not commit patient images merely to create a parity test.

## Delivery options

For this portfolio project, the simplest reliable approach is:

1. Publish the approved `.tflite` as a versioned release asset in the ML
   repository.
2. Verify its SHA-256 after download.
3. Copy it deliberately into the Android assets directory.
4. Commit the Android asset together with its small JSON metadata manifest.

Automating the download in Gradle can be added later, but a network-dependent
build is not necessary for the initial portfolio cleanup.

