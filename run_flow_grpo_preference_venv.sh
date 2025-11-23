#!/usr/bin/env bash
set -euo pipefail

###############################################################
# Flow-GRPO Human Preference Alignment (PickScore) runner     #
# - Uses PickScore locally (NO reward server needed)          #
# - Auto-detects GPU count                                    #
# - Trains on ALL GPUs                                        #
###############################################################

# ----------- USER CONFIG -----------
FLOW_GRPO_DIR="flow_grpo-main"
TRAIN_VENV="Flow_env"      # venv folder name for flow_grpo training

TRAIN_SCRIPT="scripts/train_sd3_fast.py"
TRAIN_CONFIG="config/grpo.py:pickscore_sd3"

# Extra accelerate args (optional), e.g. ACCELERATE_ARGS="--mixed_precision bf16"
ACCELERATE_ARGS=""
# ----------------------------------

detect_gpu_count () {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
  else
    python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
  fi
}

make_all_gpus () {
  local n="$1"
  python - <<PY
n=$n
print(",".join(str(i) for i in range(n)))
PY
}

activate_venv () {
  local venv_dir="$1"
  if [ ! -d "$venv_dir" ]; then
    echo "[ERROR] venv directory '$venv_dir' not found."
    echo "        Please set TRAIN_VENV correctly."
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"
}

GPU_COUNT="$(detect_gpu_count)"
if [ "$GPU_COUNT" -lt 1 ]; then
  echo "[ERROR] No GPUs detected."
  exit 1
fi

TRAIN_GPUS="$(make_all_gpus "$GPU_COUNT")"
NUM_TRAIN_PROCS="$GPU_COUNT"

echo "[INFO] Detected GPUs: $GPU_COUNT"
echo "[INFO] Training GPUs: $TRAIN_GPUS (num_processes=$NUM_TRAIN_PROCS)"
echo "[INFO] PickScore reward is computed locally on each training GPU."

# ----------- launch training -----------
pushd "${FLOW_GRPO_DIR}" >/dev/null

activate_venv "../${TRAIN_VENV}"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"

accelerate launch \
  --num_processes "${NUM_TRAIN_PROCS}" \
  ${ACCELERATE_ARGS} \
  "${TRAIN_SCRIPT}" \
  --config "${TRAIN_CONFIG}" \
  "$@"

popd >/dev/null
echo "[DONE] Preference alignment training finished."
