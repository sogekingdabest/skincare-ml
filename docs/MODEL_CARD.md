# Model Card: SkinCare Multimodal Classifier

## Summary

Experimental binary skin-lesion classifier created as part of the SkinCare
portfolio project. The model uses transfer learning with EfficientNetV2-B0 and
combines image features with basic metadata.

This model is not the classifier currently deployed in the Android app.

## Intended use

- Educational demonstration of an end-to-end ML pipeline.
- Research and portfolio experimentation with imbalanced image classification.
- TensorFlow Lite export and mobile-integration testing.

## Out-of-scope use

- Clinical diagnosis or screening.
- Replacing a dermatologist or other healthcare professional.
- Providing treatment recommendations.
- Use on populations, devices or capture conditions not represented in the
  validation data.

## Architecture

| Property | Value |
| --- | --- |
| Image backbone | EfficientNetV2-B0, ImageNet weights |
| Image input | Float32 RGB, `256 x 256 x 3` |
| Image scaling | Pixel values divided by `255.0` |
| Metadata input | Three float values |
| Metadata encoding | Age `/100`, encoded sex, anatomical site `/7` |
| Fusion | Concatenated image and metadata branches |
| Head | Dense 128, ReLU, dropout 0.3, sigmoid output |
| Loss | Focal loss, gamma 2.0, alpha 0.75 |
| Optimizer | Adam |
| Export | Dynamically optimised TensorFlow Lite |

## Training

Training has two principal phases:

1. Initial training with a cosine-decay learning-rate schedule.
2. Fine-tuning of the final 50 EfficientNetV2-B0 layers at a lower learning
   rate.

The notebook also contains an experimental 384-pixel progressive-resizing
phase. The exported artifact currently comes from the best 256-pixel model.

## Recorded validation result

The current notebook output reports the following values for its 256-pixel
validation split:

| Metric | Value |
| --- | ---: |
| Decision threshold | 0.2002 |
| ROC AUC | 0.8541 |
| Average precision | 0.1368 |
| Sensitivity | 85.47% |
| Specificity | 69.94% |
| Precision | 4.91% |
| Negative predictive value | 99.62% |
| F1 score | 0.0929 |
| Matthews correlation coefficient | 0.1585 |
| Balanced accuracy | 77.71% |

These are experimental validation metrics, not externally validated clinical
performance. The low precision and average precision are material limitations
and must not be hidden behind the high accuracy or negative predictive value.

## Known limitations

- Severe class imbalance makes accuracy a misleading headline metric.
- The validation split is based on sorted TFRecord shards; patient-level
  separation has not yet been demonstrated.
- There is no independent test cohort or external validation.
- Demographic and acquisition-device performance have not been stratified.
- Metadata encodings are simplistic and include sentinel values that need
  explicit documentation.
- The output probability has not been clinically calibrated.
- The notebook model requires metadata that the current Android flow does not
  provide to its deployed classifier.
- The Android app currently uses a different 224-pixel, single-input model.
- Performance on ordinary phone photographs cannot be inferred directly from
  performance on dermoscopic datasets.

## Reproducibility

The notebook records:

- Python 3.12.12;
- Kaggle GPU execution;
- NumPy seed 42;
- TensorFlow seed 42;
- Kaggle dataset-version metadata.

For a release-quality run, also record exact package versions, the Kaggle
Docker image, Python's random seed and deterministic-operation settings where
supported.

## Ethical and safety considerations

False negatives can delay medical evaluation, while false positives can cause
unnecessary anxiety. Any user-facing integration must state that the result is
informational, avoid diagnostic language and direct users to qualified medical
professionals when concerned.

