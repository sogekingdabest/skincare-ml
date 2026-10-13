"""Versioned metadata encoding and validation-only experimental operating points."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_curve


ENCODING_VERSION = "cdeotte-isic2020-metadata-v1"
SEX_CODES = {"male": 0, "female": 1, "unspecified": -1}
SITE_CODES = {
    "head_neck": 0,
    "upper_extremity": 1,
    "lower_extremity": 2,
    "torso": 3,
    "palms_soles": 4,
    "oral_genital": 5,
    "unspecified": -1,
}
SCENARIOS = ("recorded_context", "site_only", "no_context")
SENSITIVITY_TARGETS = (0.85, 0.90, 0.95)


def scenario_metadata(age: int, sex: int, site: int, scenario: str) -> list[float]:
    from lesion_model_contract import encode_metadata

    if scenario not in SCENARIOS:
        raise ValueError("Unknown metadata scenario")
    if scenario == "no_context":
        age, sex, site = -1, -1, -1
    elif scenario == "site_only":
        age, sex = -1, -1
    return encode_metadata(age, sex, site)


def select_operating_point(labels, probabilities, minimum_sensitivity: float, *, split: str = "validation") -> dict:
    if split != "validation":
        raise ValueError("Operating points must be selected on validation only")
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or set(y) != {0, 1}:
        raise ValueError("Validation needs equally sized arrays with both classes")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid model scores")
    if not math.isfinite(minimum_sensitivity) or not 0 < minimum_sensitivity <= 1:
        raise ValueError("Invalid sensitivity target")
    false_positive, true_positive, thresholds = roc_curve(y, p, drop_intermediate=False)
    eligible = [i for i, threshold in enumerate(thresholds)
                if math.isfinite(threshold) and true_positive[i] >= minimum_sensitivity]
    selected = min(eligible, key=lambda i: (false_positive[i], -thresholds[i]))
    return {
        "threshold": float(thresholds[selected]),
        "minimum_sensitivity": minimum_sensitivity,
        "validation_sensitivity": float(true_positive[selected]),
        "validation_specificity": float(1 - false_positive[selected]),
        "selected_on": "validation",
        "clinical_validation": False,
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid interval counts")
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]
