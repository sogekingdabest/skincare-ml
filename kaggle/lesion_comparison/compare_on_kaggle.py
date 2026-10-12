"""Run the frozen two-artifact comparison using a private input bundle."""

from pathlib import Path
import subprocess
import sys

input_root = Path("/kaggle/input")
matches = list(input_root.rglob("lesion-comparison-inputs.json"))
if len(matches) != 1:
    raise RuntimeError("Attach exactly one versioned private lesion comparison input bundle")
bundle = matches[0].parent
subprocess.run([sys.executable, str(bundle / "compare_lesion_models.py"),
    "--legacy-model", str(bundle / "melanoma_efficientnet_dynamic.tflite"),
    "--candidate-model", str(bundle / "skincare_multimodal_256_float16.tflite"),
    "--legacy-contract", str(bundle / "legacy.json"),
    "--candidate-contract", str(bundle / "candidate.json"),
    "--metrics", str(bundle / "metrics_256.json"),
    "--tfrecord-root", str(input_root), "--output-dir", "/kaggle/working/comparison"], check=True)
