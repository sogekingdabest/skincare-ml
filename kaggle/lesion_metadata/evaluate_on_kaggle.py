"""Study missing metadata and validation-only thresholds using the private bundle."""

from pathlib import Path
import subprocess
import sys

matches = list(Path("/kaggle/input").rglob("lesion-comparison-inputs.json"))
if len(matches) != 1:
    raise RuntimeError("Attach one private comparison bundle")
bundle = matches[0].parent
subprocess.run([
    sys.executable, str(bundle / "evaluate_lesion_metadata.py"),
    "--candidate-model", str(bundle / "skincare_multimodal_256_float16.tflite"),
    "--candidate-contract", str(bundle / "candidate.json"),
    "--legacy-model", str(bundle / "melanoma_efficientnet_dynamic.tflite"),
    "--legacy-contract", str(bundle / "legacy.json"),
    "--metrics", str(bundle / "metrics_256.json"),
    "--tfrecord-root", "/kaggle/input",
    "--output-dir", "/kaggle/working/metadata-policy",
], check=True)
