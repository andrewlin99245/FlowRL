#!/usr/bin/env bash
set -euo pipefail

###############################################################
# Flow-GRPO compositional (GenEval) runner using python venvs #
# - Auto-detects GPU count                                    #
# - Uses GPU 0 for reward server                              #
# - Uses GPUs 1..N-1 for training                             #
###############################################################

# ----------- USER CONFIG -----------
REWARD_SERVER_DIR="reward-server-main"
FLOW_GRPO_DIR="flow_grpo-main"

REWARD_VENV="R_server"     # venv folder name for reward server
TRAIN_VENV="Flow_env"      # venv folder name for flow_grpo training

SERVER_APP='app_geneval:create_app()'  # GenEval app
SERVER_LOG="geneval_server.log"

TRAIN_SCRIPT="scripts/train_sd3_fast.py"
TRAIN_CONFIG="config/grpo.py:geneval_sd3"

# Extra accelerate args (optional), e.g. ACCELERATE_ARGS="--mixed_precision bf16"
ACCELERATE_ARGS=""
# ----------------------------------

detect_gpu_count () {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
  else
    # fallback to torch if nvidia-smi isn't present
    python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
  fi
}

make_train_gpus () {
  local n="$1"
  if [ "$n" -le 1 ]; then
    echo ""
    return
  fi
  # produce "1,2,3,...,n-1"
  python - <<PY
n=$n
print(",".join(str(i) for i in range(1, n)))
PY
}

activate_venv () {
  local venv_dir="$1"
  if [ ! -d "$venv_dir" ]; then
    echo "[ERROR] venv directory '$venv_dir' not found."
    echo "        Please set REWARD_VENV / TRAIN_VENV correctly."
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"
}

GPU_COUNT="$(detect_gpu_count)"
if [ "$GPU_COUNT" -lt 2 ]; then
  echo "[ERROR] Need at least 2 GPUs (1 for server + 1 for training). Detected: $GPU_COUNT"
  exit 1
fi

SERVER_GPU=0
TRAIN_GPUS="$(make_train_gpus "$GPU_COUNT")"
NUM_TRAIN_PROCS="$((GPU_COUNT - 1))"

echo "[INFO] Detected GPUs: $GPU_COUNT"
echo "[INFO] Reward server GPU: $SERVER_GPU"
echo "[INFO] Training GPUs: $TRAIN_GPUS (num_processes=$NUM_TRAIN_PROCS)"

# ----------- start GenEval reward server -----------
echo "[1/2] Starting GenEval reward server on GPU ${SERVER_GPU} ..."
pushd "${REWARD_SERVER_DIR}" >/dev/null

activate_venv "../${REWARD_VENV}"

export CUDA_VISIBLE_DEVICES="${SERVER_GPU}"

gunicorn -c gunicorn.conf.py -w 1 "${SERVER_APP}" \
  > "../${SERVER_LOG}" 2>&1 &

SERVER_PID=$!
echo "  -> Server PID: ${SERVER_PID}"
popd >/dev/null

cleanup () {
  echo
  echo "[CLEANUP] Stopping GenEval server (PID ${SERVER_PID}) ..."
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Give server time to load models
sleep 8

# Optional health check (won't fail if curl missing)
if command -v curl >/dev/null 2>&1; then
  if curl -s "http://127.0.0.1:18085/health" >/dev/null 2>&1; then
    echo "  -> Server health check OK."
  else
    echo "  -> Server health check not confirmed, continuing anyway."
  fi
else
  echo "  -> curl not found, skipping health check."
fi

# ----------- launch training -----------
echo "[2/2] Launching training ..."
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
echo "[DONE] Training finished. Server will be stopped automatically."
