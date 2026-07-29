# Datasets

The training notebook expects preprocessed TFRecord datasets attached to a
Kaggle Notebook. Dataset files are not redistributed in this repository.

## Kaggle inputs

| Purpose | Kaggle handle/path used by the notebook | Resolution |
| --- | --- | --- |
| SIIM-ISIC 2020 training, validation and test | `cdeotte/melanoma-256x256` | 256 |
| ISIC 2019 additional training data | `cdeotte/isic2019-256x256` | 256 |
| SIIM-ISIC 2020 progressive resizing | `cdeotte/melanoma-384x384` | 384 |
| ISIC 2019 progressive resizing | `cdeotte/isic2019-384x384` | 384 |

The notebook metadata also records the attached Kaggle dataset version IDs.
Keep those IDs when publishing the notebook so a future run can identify the
same snapshots.

## Split used by the current notebook

- The 2020 TFRecord filenames are sorted and shuffled with seed `42`.
- 70% of the 2020 shards are used for training.
- 15% are used only to select the decision threshold.
- 15% form a held-out test split used for final metrics.
- All 2019 shards are added only to training.
- The same deterministic shard indices are used at 256 and 384 pixels.

The notebook reads `patient_id` when the field is available and aborts if a
patient appears in more than one split. If the source TFRecords omit that
field, `metrics.json` records `patient_id_unavailable`; in that case the split
is independent at shard level but patient-level separation remains unverified.
This internal test is not a substitute for external or clinical validation.

## Licensing and attribution

Kaggle hosts the processed datasets, but the original images come from ISIC
collections. Before publishing a model release:

1. Open the exact Kaggle Data Card for every attached dataset version.
2. Record its displayed license and version.
3. Verify the underlying ISIC image or collection licenses.
4. Add the requested academic citations and attributions.
5. Confirm whether the trained artifact may be redistributed.

The ISIC Archive explains that permission can vary per image and may be CC0,
CC-BY or CC-BY-NC. A Kaggle listing alone should not be treated as a substitute
for checking the source collection.

Useful references:

- Kaggle datasets: <https://www.kaggle.com/datasets>
- ISIC licensing FAQ: <https://www.isic-archive.com/blank-1>
- ISIC Archive: <https://www.isic-archive.com/>

## Privacy

Do not commit source images, patient identifiers, Kaggle credentials or private
dataset exports. Only use properly de-identified data under its applicable
license and terms.
