from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .runtime import DATASETS, ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a frozen formal Auto-CoT example library")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--model", default=os.getenv("LOCAL_MODEL_NAME", "qwen3.6:latest"))
    parser.add_argument("--base-url", default=os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reasoning-effort", default=os.getenv("RELOGIC_REASONING_EFFORT", "none"))
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    safe_model = args.model.replace("/", "_").replace(":", "_")
    destination = ROOT / "artifacts/autocot" / safe_model / args.dataset
    candidates = destination / "candidates"
    demo = destination / "demo.json"
    destination.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable, str(ROOT / "run_inference.py"),
            "--dataset", spec.task_name,
            "--method", "zero_shot_cot",
            "--dataset_path", str(spec.train_path),
            "--limit_dataset_size", str(args.train_limit),
            "--random_seed", str(args.seed),
            "--model", args.model,
            "--base_url", args.base_url,
            "--reasoning_effort", args.reasoning_effort,
            "--output_dir", str(candidates),
        ],
        cwd=ROOT,
        check=True,
    )
    prediction_file = candidates / f"{spec.task_name}.jsonl"
    subprocess.run(
        [
            sys.executable, str(ROOT / "run_demo.py"),
            "--task", spec.task_name,
            "--pred_file", str(prediction_file),
            "--demo_save_dir", str(demo),
            "--random_seed", str(args.seed),
        ],
        cwd=ROOT,
        check=True,
    )
    manifest = {
        "dataset_id": args.dataset,
        "model_id": args.model,
        "prompt_version": "relogic-formal-v1",
        "seed": args.seed,
        "candidate_count": args.train_limit,
        "train_sha256": sha256_file(spec.train_path),
        "demo_sha256": sha256_file(demo),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
