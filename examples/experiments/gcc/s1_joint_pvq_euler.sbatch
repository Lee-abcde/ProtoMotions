#!/bin/bash
#SBATCH --job-name=joint_pvq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=rtx_4090:8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=80G
#SBATCH --time=24:00:00
# Absolute so the log lands next to the run outputs; the directory must exist.
#SBATCH --output=/cluster/scratch/lijingy/joint_pvq_results/joint_pvq_%j.log
#SBATCH --error=/cluster/scratch/lijingy/joint_pvq_results/joint_pvq_%j.log

set -euo pipefail

: "${CONTAINER:?Submit with --export=ALL,CONTAINER=/cluster/scratch/lijingy/<image>.sif}"
PROTOMOTIONS_DIR="${PROTOMOTIONS_DIR:-/cluster/home/lijingy/ProtoMotions}"
DATA_DIR="${DATA_DIR:-/cluster/scratch/lijingy/protomotion_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/lijingy/joint_pvq_results}"
# Teacher run directories (each holds score_based.ckpt + resolved_configs.pt).
HOI_TEACHER_DIR="${HOI_TEACHER_DIR:-/cluster/scratch/lijingy/HOI_smplx}"
LOCO_TEACHER_DIR="${LOCO_TEACHER_DIR:-/cluster/scratch/lijingy/smplx_results}"
RUN_NAME="${RUN_NAME:-joint_pvq_v1}"
CONTAINER_ENV="${CONTAINER_ENV:-/opt/IsaacLab_new/.venv/bin/activate}"
NUM_ENVS="${NUM_ENVS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-32}"
ITERATIONS="${ITERATIONS:-10000}"
# Full student evaluation (all AMASS-X shards and every OMOMO subject); 0 disables it.
EVAL_EVERY="${EVAL_EVERY:-1000}"
# Weights & Biases: USE_WANDB=1 enables it. Online like train_slurm.py, with
# credentials from ~/.netrc on the cluster; set WANDB_MODE=offline to defer upload.
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-physical_animation}"
WANDB_MODE="${WANDB_MODE:-online}"
export CONTAINER_ENV RUN_NAME NUM_ENVS BATCH_SIZE ROLLOUT_STEPS ITERATIONS EVAL_EVERY
export USE_WANDB WANDB_PROJECT WANDB_MODE

for path in "$CONTAINER" "$PROTOMOTIONS_DIR" "$DATA_DIR" "$HOI_TEACHER_DIR" "$LOCO_TEACHER_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "Required Euler path does not exist: $path" >&2
    exit 2
  fi
done

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
NODE_TMP_ROOT="${TMPDIR:-/tmp}/protomotions_joint_pvq_$SLURM_JOB_ID"
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/collision_cache" "$NODE_TMP_ROOT"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 20000))
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PXR_WORK_THREAD_LIMIT="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1

container_args=(
  --nv
  --writable-tmpfs
  --bind "$PROTOMOTIONS_DIR:/workspace/ProtoMotions:rw"
  --bind "$DATA_DIR:/workspace/protomotion_data:ro"
  --bind "$HOI_TEACHER_DIR:/workspace/hoi_teachers:ro"
  --bind "$LOCO_TEACHER_DIR:/workspace/loco_teachers:ro"
  --bind "$OUTPUT_ROOT:/workspace/output:rw"
  --bind "$NODE_TMP_ROOT:/tmp:rw"
  --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro
)

inner_script=$(cat <<'INNER'
set -euo pipefail
source "$CONTAINER_ENV"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH=/workspace/ProtoMotions:${PYTHONPATH:-}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TMPDIR=/tmp
export MPLCONFIGDIR="/tmp/matplotlib_rank_${SLURM_PROCID}"
export PROTOMOTIONS_COLLISION_CACHE_DIR=/workspace/output/collision_cache
mkdir -p "$MPLCONFIGDIR" "$PROTOMOTIONS_COLLISION_CACHE_DIR"
cd /workspace/ProtoMotions

resume_args=()
if [[ -f "/workspace/output/$RUN_NAME/last.ckpt" ]]; then
  resume_args=(--resume "/workspace/output/$RUN_NAME/last.ckpt")
fi

wandb_args=()
if [[ "$USE_WANDB" == "1" ]]; then
  export WANDB_DIR="/workspace/output/$RUN_NAME"
  export WANDB_CACHE_DIR="/workspace/output/wandb_cache"
  export WANDB_CONFIG_DIR="/workspace/output/wandb_config"
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
  wandb_args=(--use-wandb --wandb-project "$WANDB_PROJECT")
fi

python -u -m protomotions.train_multi_source_distill \
  --distillation-sources examples/experiments/gcc/s1_joint_pvq_sources_euler.yaml \
  --output-dir "/workspace/output/$RUN_NAME" \
  --num-envs "$NUM_ENVS" \
  --batch-size "$BATCH_SIZE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --iterations "$ITERATIONS" \
  --save-every 50 \
  --eval-every "$EVAL_EVERY" \
  --ujitso-cache-dir /tmp/protomotions_ujitso \
  --ujitso-cache-budget-mb 1024 \
  --timeout-minutes 180 \
  "${wandb_args[@]}" \
  "${resume_args[@]}"
INNER
)

echo "Euler preflight: code, data, checkpoints, action contract, and rank assignment"
apptainer exec "${container_args[@]}" "$CONTAINER" bash -lc \
  'source "$CONTAINER_ENV"; export PYTHONPATH=/workspace/ProtoMotions:${PYTHONPATH:-}; cd /workspace/ProtoMotions; python -m protomotions.train_multi_source_distill --distillation-sources examples/experiments/gcc/s1_joint_pvq_sources_euler.yaml --check-config-only --world-size 8'

echo "Starting eight-rank training in $RUN_DIR"
srun --kill-on-bad-exit=1 --wait=60 \
  apptainer exec "${container_args[@]}" "$CONTAINER" bash -lc "$inner_script"
