"""Stage evaluator source and a private Kaggle entry point without uploading data."""
import argparse
import hashlib
import json
from pathlib import Path


def stage(destination: Path, policy_sha: str, predictions_sha: str, report_sha: str, negative_count: int, docker_image: str) -> dict:
    for value in (policy_sha, predictions_sha, report_sha):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Expected pinned SHA-256 references")
    if negative_count < 1:
        raise ValueError("Need a positive negative-sample count")
    repo = Path(__file__).resolve().parents[1]
    source_path = repo / "scripts/evaluate_lesion_photo_stress.py"
    source = source_path.read_text(encoding="utf-8")
    entry = '\nif __name__ == "__main__":\n    main()\n'
    if not source.endswith(entry):
        raise ValueError("Evaluator entry point changed; review staging")
    wrapper = f'''
def one_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError("Expected one attached input " + name)
    return matches[0]

if __name__ == "__main__":
    sys.argv = [__file__,
        "--bundle-dir", str(one_input("lesion-comparison-inputs.json").parent),
        "--reference-dir", str(one_input("frozen-validation-policy.json").parent),
        "--tfrecord-root", "/kaggle/input", "--output-dir", "/kaggle/working/photo-stress",
        "--reference-policy-sha", "{policy_sha}",
        "--reference-predictions-sha", "{predictions_sha}",
        "--reference-report-sha", "{report_sha}",
        "--negative-count", "{negative_count}", "--seed", "20261013"]
    main()
'''
    body = source[:-len(entry)] + wrapper
    compile(body, "evaluate_on_kaggle.py", "exec")
    metadata = json.loads((repo / "kaggle/lesion_photo_stress/kernel-metadata.json").read_text())
    if not metadata["is_private"] or metadata["enable_internet"] or metadata["enable_gpu"]:
        raise ValueError("Expected private offline CPU evaluation")
    if docker_image:
        metadata["docker_image"] = docker_image
        metadata["docker_image_pinning_type"] = "original"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluate_on_kaggle.py").write_text(body, encoding="utf-8", newline="\n")
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    return {"source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256((destination / "evaluate_on_kaggle.py").read_bytes()).hexdigest(), "negative_count": negative_count,
        "private": True, "photos_or_models_uploaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--predictions-sha", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--negative-count", type=int, default=512)
    parser.add_argument("--docker-image", default="")
    args = parser.parse_args()
    print(json.dumps(stage(args.output_dir, args.policy_sha, args.predictions_sha, args.report_sha, args.negative_count, args.docker_image)))
