"""Generate patient-free, fixed-resolution cross-runtime contract fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from lesion_model_contract import FloatLesionInterpreter, load_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy-model", "candidate-model", "legacy-contract", "candidate-contract", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases, timings = [], {}
    for label, model_path, contract_path in (("legacy", args.legacy_model, args.legacy_contract), ("candidate", args.candidate_model, args.candidate_contract)):
        contract = load_contract(contract_path)
        runtime = FloatLesionInterpreter(model_path, contract)
        _, height, width, _ = next(t["shape"] for t in contract["inputs"] if t["role"] == "image")
        y, x = np.indices((height, width))
        images = {"gray": np.full((height, width, 3), 128, dtype=np.uint8),
                  "rgb": np.stack((x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), (x + y) % 256), axis=-1).astype(np.uint8),
                  "shape": np.where(((x - width // 2) ** 2 + (y - height // 2) ** 2 < (width // 5) ** 2)[..., None],
                                    np.array([60, 35, 25], dtype=np.uint8), np.array([190, 155, 135], dtype=np.uint8)).astype(np.uint8)}
        meta = [0.45, 1.0, 4 / 7] if len(contract["inputs"]) == 2 else None
        model_times = []
        for name, pixels in images.items():
            filename = f"{label}-{name}.png"
            Image.fromarray(pixels).save(args.output_dir / filename)
            score = runtime.predict(pixels, meta)
            normalized = pixels.astype(np.float32)
            normalized = (normalized / np.float32(255)) if contract["imageNormalization"] == "ZERO_TO_ONE" else (normalized / np.float32(127.5) - np.float32(1))
            for _ in range(5):
                started = time.perf_counter()
                runtime.predict(pixels, meta)
                model_times.append((time.perf_counter() - started) * 1000)
            cases.append({"model": label, "image": filename, "imageInputSha256": hashlib.sha256(normalized.astype("<f4").tobytes()).hexdigest(),
                          "metadata": meta, "expectedScore": score, "absoluteTolerance": 0.002,
                          "modelSha256": contract["sha256"]})
        timings[label] = {"medianMs": float(np.median(model_times)), "p95Ms": float(np.percentile(model_times, 95)), "samples": len(model_times)}
    (args.output_dir / "fixtures.json").write_text(json.dumps({"schemaVersion": 1, "purpose": "synthetic integration parity; not accuracy evaluation", "cases": cases}, indent=2), encoding="utf-8")
    print(json.dumps({"syntheticDesktopTimings": timings, "accuracyEvaluated": False}))


if __name__ == "__main__":
    main()
