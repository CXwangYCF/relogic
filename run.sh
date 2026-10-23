#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset=${1:-}
stage=${2:-}
case "$dataset" in logic2text|spider) ;; *) echo '用法: bash run.sh {logic2text|spider} {zero|train|demo|auto|eval-zero|eval-auto}' >&2; exit 2;; esac
python_bin=${PYTHON:-python3}
task="relogic_${dataset}"
common=(--dataset "$task" --base_url "${LOCAL_MODEL_BASE_URL:-http://localhost:11434/v1}" --model "${LOCAL_MODEL_NAME:-qwen3.6:latest}" --reasoning_effort "${REASONING_EFFORT:-none}")
case "$stage" in
  zero)
    "$python_bin" run_inference.py "${common[@]}" --method zero_shot_cot --limit_dataset_size "${LIMIT:-0}" --output_dir "output/${dataset}_zero" ;;
  train)
    "$python_bin" run_inference.py "${common[@]}" --method zero_shot_cot --dataset_path "dataset/relogic_data/${dataset}/raw/train.json" --limit_dataset_size "${TRAIN_LIMIT:-256}" --output_dir "output/${dataset}_train" ;;
  demo)
    encoder_args=()
    if [[ -n "${ENCODER:-}" ]]; then encoder_args=(--encoder "$ENCODER"); fi
    "$python_bin" run_demo.py --task "$task" --pred_file "output/${dataset}_train/${task}.jsonl" --demo_save_dir "demos/${dataset}.json" "${encoder_args[@]}" ;;
  auto)
    "$python_bin" run_inference.py "${common[@]}" --method auto_cot --demo_path "demos/${dataset}.json" --limit_dataset_size "${LIMIT:-0}" --output_dir "output/${dataset}_auto" ;;
  eval-zero|eval-auto)
    method=${stage#eval-}
    "$python_bin" evaluate_blec.py --dataset "$dataset" --input "output/${dataset}_${method}/${task}.jsonl" ;;
  *) echo "未知步骤: $stage" >&2; exit 2 ;;
esac
