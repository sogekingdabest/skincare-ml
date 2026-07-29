# Datasets

The training notebook expects preprocessed TFRecord datasets attached to a
Kaggle Notebook. Dataset files are not redistributed in this repository.

## Kaggle inputs

| Purpose | Kaggle handle/path used by the notebook | Resolution |
| --- | --- | --- |
| SIIM-ISIC 2020 training and validation | `cdeotte/melanoma-256x256` | 256 |
| ISIC 2019 additional training data | `cdeotte/isic2019-256x256` | 256 |
| SIIM-ISIC 2020 progressive resizing | `cdeotte/melanoma-384x384` | 384 |
| ISIC 2019 progressive resizing | `cdeotte/isic2019-384x384` | 384 |

The notebook metadata also records the attached Kaggle dataset version IDs.
Keep those IDs when publishing the notebook so a future run can identify the
same snapshots.

## Split used by the current notebook

- The 2020 TFRecord filenames are sorted.
- The first 80% of the 2020 TFRecord shards are used for training.
- The last 20% of the 2020 shards are used for validation.
- All 2019 shards are added to training.

This is a shard-level split, not an explicitly verified patient-level split.
Before treating the reported metrics as a reliable generalisation estimate,
confirm that a patient cannot appear in more than one split.

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

