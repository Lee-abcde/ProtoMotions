#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Native IsaacLab environment; full AMASS + all 15 training OMOMO subjects.
# Modes: prepare (write source manifest), check (checkpoint/config preflight), train.
# One locomotion rank plus NUM_GPUS - 1 HOI ranks (default three GPUs). Select the
# allocated GPUs explicitly, e.g. CUDA_VISIBLE_DEVICES=0,5,6.
# Start a new stage-two run: eight-rank stage-two resumes are incompatible.
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
POSTERIOR_CHECKPOINT="${POSTERIOR_CHECKPOINT:-$LOCAL_ROOT/checkpoints/joint_pvq_v1/score_based.ckpt}"
NUM_GPUS="${NUM_GPUS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_ROOT/output/joint_pvq_prior_hyperion_${NUM_GPUS}gpu}"
# Beside, not inside, the run directory: a fresh run requires an empty output.
SOURCES="${SOURCES:-${OUTPUT_DIR%/}_sources.yaml}"
export PYTHONPATH="$PROTOMOTIONS_DIR:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
# As in the Euler sbatch: a background run has no terminal for Kit's EULA prompt.
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export PXR_WORK_THREAD_LIMIT="${PXR_WORK_THREAD_LIMIT:-8}"
export PROTOMOTIONS_COLLISION_CACHE_DIR="${PROTOMOTIONS_COLLISION_CACHE_DIR:-$LOCAL_ROOT/.cache/protomotions/collision}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOCAL_ROOT/.cache/matplotlib}"

if [[ "$MODE" == prepare ]]; then
    exec "$PYTHON" -m data.scripts.prepare_joint_prior_hyperion \
        --data-dir "$DATA_DIR" --amass-motion-file "$AMASS_MOTION_FILE" \
        --hoi-teacher-dir "$HOI_TEACHER_DIR" \
        --loco-teacher-dir "$LOCO_TEACHER_DIR" --output "$SOURCES"
fi
if [[ ! -f protomotions/train_joint_prior.py ]]; then
    echo "This checkout lacks joint prior training code; synchronize the training branch first." >&2
    exit 2
fi
if [[ ! -f "$SOURCES" || ! -f "$POSTERIOR_CHECKPOINT" ]]; then
    echo "Run prepare first and upload the stage-one checkpoint: $POSTERIOR_CHECKPOINT" >&2
    exit 2
fi
# Locomotion 4096 envs / batch 8192 and two HOI ranks of 2048 / 4096 match the
# eight-rank Euler run: 8 * 1024 * 32 samples per iteration, half per task, a
# global batch of 8 * 2048, and 16 minibatches on every rank. With two GPUs use
# HOI_NUM_ENVS=4096 HOI_BATCH_SIZE=8192. The layout, rank count and TF32 are in
# the training contract; keep them when resuming.
args=(
    --checkpoint "$POSTERIOR_CHECKPOINT"
    --experiment-path examples/experiments/gcc/s2_joint_pvq_prior.py
    --distillation-sources "$SOURCES" --output-dir "$OUTPUT_DIR"
    --num-envs "${NUM_ENVS:-4096}" --batch-size "${BATCH_SIZE:-8192}"
    --hoi-num-envs "${HOI_NUM_ENVS:-2048}" --hoi-batch-size "${HOI_BATCH_SIZE:-4096}"
    --rollout-steps "${ROLLOUT_STEPS:-32}" --mini-epochs "${MINI_EPOCHS:-6}"
    --iterations "${ITERATIONS:-100000000}"
    --save-every "${SAVE_EVERY:-50}" --snapshot-every "${SNAPSHOT_EVERY:-1000}"
    --eval-every "${EVAL_EVERY:-0}" --student-psi "${STUDENT_PSI:-teacher}"
    --prior-rollout-start-iteration "${PRIOR_ROLLOUT_START:-500}"
    --prior-rollout-ramp-iterations "${PRIOR_ROLLOUT_RAMP:-9500}"
    --prior-rollout-max-prob "${PRIOR_ROLLOUT_MAX_PROB:-0.95}"
    --kit-log-dir "$OUTPUT_DIR/kit_logs"
    --ujitso-cache-dir "$LOCAL_ROOT/.cache/protomotions/ujitso"
    --timeout-minutes 180
)
# TF32 for prior updates only (~2x faster updates on H200); TRAIN_TF32=0 disables.
if [[ "${TRAIN_TF32:-1}" == 1 ]]; then args+=(--train-tf32); fi
if [[ -n "${HOI_TEACHER_FILTER:-}" ]]; then args+=(--hoi-teacher-filter "$HOI_TEACHER_FILTER"); fi
if [[ "${USE_WANDB:-0}" == 1 ]]; then args+=(--use-wandb --wandb-project "${WANDB_PROJECT:-physical_animation}"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
"$PYTHON" -m protomotions.train_joint_prior "${args[@]}" --check-config-only --world-size "$NUM_GPUS"
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
    -m protomotions.train_joint_prior "${args[@]}"
