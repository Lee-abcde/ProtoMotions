#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage one (joint Product-VQ posterior) on Hyperion: native IsaacLab environment,
# full AMASS + all 15 training OMOMO subjects.
# Modes: prepare (write source manifest), check (config preflight), train.
# One locomotion rank plus NUM_GPUS - 1 HOI ranks (default three GPUs). Select the
# allocated GPUs explicitly, e.g. CUDA_VISIBLE_DEVICES=0,5,6.
# The eight-rank Euler run cannot be resumed here (the rank assignment is part of
# the checkpoint contract); WARM_START=<joint PVQ ckpt> loads its student instead.
set -euo pipefail
MODE="${1:-check}"
case "$MODE" in prepare|check|train) ;; *) echo "Usage: bash $0 {prepare|check|train}" >&2; exit 2 ;; esac
PROTOMOTIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROTOMOTIONS_DIR"
LOCAL_ROOT="${LOCAL_ROOT:-/local/home/lijingy}"
PYTHON="${PYTHON:-$PROTOMOTIONS_DIR/IsaacLab/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-$LOCAL_ROOT/data}"
AMASS_MOTION_FILE="${AMASS_MOTION_FILE:-$DATA_DIR/amassx/amass_smplx_train.pt}"
HOI_TEACHER_DIR="${HOI_TEACHER_DIR:-$LOCAL_ROOT/checkpoints/HOI_smplx}"
LOCO_TEACHER_DIR="${LOCO_TEACHER_DIR:-$LOCAL_ROOT/checkpoints/smplx_results}"
NUM_GPUS="${NUM_GPUS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_ROOT/output/joint_pvq_hyperion_${NUM_GPUS}gpu}"
# Beside, not inside, the run directory: a fresh run requires an empty output.
SOURCES="${SOURCES:-${OUTPUT_DIR%/}_sources.yaml}"
export PYTHONPATH="$PROTOMOTIONS_DIR:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
# As in the Euler sbatch: a background run has no terminal for Kit's EULA prompt.
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PROTOMOTIONS_COLLISION_CACHE_DIR="${PROTOMOTIONS_COLLISION_CACHE_DIR:-$LOCAL_ROOT/.cache/protomotions/collision}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCAL_ROOT/.cache/matplotlib}"

if [[ "$MODE" == prepare ]]; then
    exec "$PYTHON" -m data.scripts.prepare_joint_prior_hyperion \
        --data-dir "$DATA_DIR" --amass-motion-file "$AMASS_MOTION_FILE" \
        --hoi-teacher-dir "$HOI_TEACHER_DIR" \
        --loco-teacher-dir "$LOCO_TEACHER_DIR" --output "$SOURCES"
fi
if [[ ! -f "$SOURCES" ]]; then
    echo "Run prepare first: $SOURCES" >&2
    exit 2
fi
# An HOI env steps about 2.5x slower than a locomotion env (Hyperion stage-two
# benchmarks), so locomotion 4096 / batch 8192 and HOI 2048 / 4096 keep the ranks
# close and give each one 16 minibatches, which DDP requires to be equal. The
# loss is reweighted to the manifest's source weights regardless of the layout.
# The layout is in the training contract; keep it when resuming.
args=(
    --distillation-sources "$SOURCES" --output-dir "$OUTPUT_DIR"
    --num-envs "${NUM_ENVS:-4096}" --batch-size "${BATCH_SIZE:-8192}"
    --hoi-num-envs "${HOI_NUM_ENVS:-2048}" --hoi-batch-size "${HOI_BATCH_SIZE:-4096}"
    --rollout-steps "${ROLLOUT_STEPS:-32}" --mini-epochs "${MINI_EPOCHS:-6}"
    --learning-rate "${LEARNING_RATE:-2e-5}"
    --iterations "${ITERATIONS:-100000000}"
    --save-every "${SAVE_EVERY:-50}" --eval-every "${EVAL_EVERY:-1000}"
    --student-psi "${STUDENT_PSI:-teacher}"
    --kit-log-dir "$OUTPUT_DIR/kit_logs"
    --ujitso-cache-dir "$LOCAL_ROOT/.cache/protomotions/ujitso"
    --timeout-minutes 180
)
if [[ -n "${HOI_TEACHER_FILTER:-}" ]]; then args+=(--hoi-teacher-filter "$HOI_TEACHER_FILTER"); fi
if [[ "${EVALUATE_TEACHERS:-0}" == 1 ]]; then args+=(--evaluate-teachers); fi
if [[ "${USE_WANDB:-0}" == 1 ]]; then args+=(--use-wandb --wandb-project "${WANDB_PROJECT:-physical_animation}"); fi
if [[ -n "${RESUME:-}" && -n "${WARM_START:-}" ]]; then
    echo "Set RESUME or WARM_START, not both" >&2
    exit 2
fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ -n "${WARM_START:-}" ]]; then args+=(--warm-start "$WARM_START"); fi
"$PYTHON" -m protomotions.train_multi_source_distill "${args[@]}" --check-config-only --world-size "$NUM_GPUS"
if [[ "$MODE" == check ]]; then exit 0; fi
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to $NUM_GPUS allocated GPU IDs}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
unique_gpus=$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l)
if [[ ${#gpu_ids[@]} != "$NUM_GPUS" || "$unique_gpus" != "$NUM_GPUS" ]]; then
    echo "CUDA_VISIBLE_DEVICES must list exactly $NUM_GPUS different GPUs" >&2
    exit 2
fi
if [[ -f "$OUTPUT_DIR/last.ckpt" && -z "${RESUME:-}" ]]; then
    echo "Existing run: set RESUME=$OUTPUT_DIR/last.ckpt or choose a new OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$PROTOMOTIONS_COLLISION_CACHE_DIR" "$MPLCONFIGDIR"
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    -m protomotions.train_multi_source_distill "${args[@]}"
