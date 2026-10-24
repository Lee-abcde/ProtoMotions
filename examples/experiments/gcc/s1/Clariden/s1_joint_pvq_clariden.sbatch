#!/bin/bash
#SBATCH --job-name=joint_pvq
#SBATCH --account=abs33
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# Every rank sees all four GPUs and picks its own by local rank, as on Euler, so
# NCCL can use NVLink between them.
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
# Absolute so the log lands next to the run outputs; the directory must exist.
#SBATCH --output=/capstor/scratch/cscs/jli/joint_pvq_results/%x_%j.log

# Stage one (joint Product-VQ posterior) on one Clariden node (4x GH200): one
# locomotion rank with the full AMASS-X dataset plus three HOI ranks with the 16
# OMOMO v9_3 training subjects (sub14 is the held-out test set). Every step runs in the Isaac Lab container from EDF with
# the pinned ProtoMotions venv. A run continues from last.ckpt on resubmission.
set -euo pipefail

PROTOMOTIONS_DIR="${PROTOMOTIONS_DIR:-/users/jli/ProtoMotions}"
EDF="${EDF:-/users/jli/.edf/isaaclab.toml}"
VENV="${VENV:-/users/jli/venvs/protomotions-isaaclab-pin}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/capstor/scratch/cscs/jli/joint_pvq_results}"
RUN_NAME="${RUN_NAME:-joint_pvq_v2}"
# Relative to PROTOMOTIONS_DIR, or absolute.
SOURCES="${SOURCES:-examples/experiments/gcc/s1/Clariden/s1_joint_pvq_sources_clariden_v9_3.yaml}"
# Layout from the Clariden sweeps (debug jobs 3731255-3737924): HOI collects in
# ~33 s and locomotion in ~28 s, ~39 s per iteration with the 5.5 s update, about
# 20k samples/s. HOI and locomotion each give 393216 samples, so the task loss split
# matches the data. 8192 HOI envs ran a GPU out of memory (PhysX scene corrupted);
# more locomotion envs do not fit next to the GPU-resident AMASS-X frames.
# Every rank must take as many minibatches per epoch:
# NUM_ENVS/BATCH_SIZE == HOI_NUM_ENVS/HOI_BATCH_SIZE (preflight checks). Batches
# equal to the env counts give 32 minibatches and Euler-sized updates (192 per
# iteration). The layout is in the training contract; keep it when resuming.
NUM_ENVS="${NUM_ENVS:-12288}"
BATCH_SIZE="${BATCH_SIZE:-12288}"
HOI_NUM_ENVS="${HOI_NUM_ENVS:-4096}"
HOI_BATCH_SIZE="${HOI_BATCH_SIZE:-4096}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-32}"
# A changed value cannot --resume an existing run, so give it a new RUN_NAME.
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NORMALIZER_MIN_STD="${NORMALIZER_MIN_STD:-0}"
NORMALIZE_LATENT="${NORMALIZE_LATENT:-1}"
# Effectively unlimited. A run ends on the wall clock and continues from
# last.ckpt on the next submission, so the iteration count must not cap it.
ITERATIONS="${ITERATIONS:-100000000}"
# Full student evaluation (AMASS-X and every OMOMO subject); 0 disables it.
EVAL_EVERY="${EVAL_EVERY:-1000}"
# teacher seeds HOI PSI from each teacher's env_<motion>.ckpt. sub3, 6, 7, 9, 10,
# 11, 13 and 16 use their v9_3 teacher's buffer; sub1, 2, 4, 5, 8, 12, 15 and 17
# use the v8_29 teacher's buffer (see PSI_SOURCE.txt in those teacher directories).
# empty (student PSI only) or off (raw reference) also work.
STUDENT_PSI="${STUDENT_PSI:-teacher}"
EVALUATE_TEACHERS="${EVALUATE_TEACHERS:-0}"
# Built once by the v9_3 teachers for this manifest's 16 HOI sources (~36 min);
# a manifest with other HOI sources needs HOI_TEACHER_FILTER= to rebuild it.
HOI_TEACHER_FILTER="${HOI_TEACHER_FILTER-/capstor/store/cscs/swissai/abs33/jli/checkpoints/hoi_teacher_filter/v9_3_train16/hoi_teacher_filter.json}"
# A GH200 holds the full AMASS-X frames (~18.5 GiB) on the GPU; Euler's 24 GB
# RTX 4090s need 1 to keep them in host RAM.
LOCO_FRAMES_ON_CPU="${LOCO_FRAMES_ON_CPU:-0}"
TRAIN_TF32="${TRAIN_TF32:-0}"
# Online with the key that "wandb login" on a login node writes to ~/.netrc;
# without it the run logs offline (upload later with wandb sync). 0 disables it.
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-physical_animation}"
WANDB_MODE="${WANDB_MODE:-online}"
export PROTOMOTIONS_DIR VENV OUTPUT_ROOT RUN_NAME SOURCES
export NUM_ENVS BATCH_SIZE HOI_NUM_ENVS HOI_BATCH_SIZE ROLLOUT_STEPS LEARNING_RATE
export NORMALIZER_MIN_STD NORMALIZE_LATENT ITERATIONS EVAL_EVERY STUDENT_PSI
export EVALUATE_TEACHERS HOI_TEACHER_FILTER LOCO_FRAMES_ON_CPU TRAIN_TF32
export USE_WANDB WANDB_PROJECT WANDB_MODE
# Read here, where it is the job's task count: inside the one-task preflight
# step it would be 1. Not WORLD_SIZE, which would make the ranks expect torchrun.
PREFLIGHT_RANKS="${SLURM_NTASKS:-4}"
export PREFLIGHT_RANKS

sources_path="$SOURCES"
[[ "$SOURCES" == /* ]] || sources_path="$PROTOMOTIONS_DIR/$SOURCES"
for path in "$PROTOMOTIONS_DIR" "$EDF" "$VENV/bin/activate" "$sources_path"; do
  if [[ ! -e "$path" ]]; then
    echo "Required Clariden path does not exist: $path" >&2
    exit 2
  fi
done

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT/collision_cache"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PXR_WORK_THREAD_LIMIT=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1

# Runs inside the container, once per task. MODE=check is the preflight.
inner_script=$(cat <<'INNER'
set -euo pipefail
# activate reads variables that may be unset.
set +u
source "$VENV/bin/activate"
set -u
# Isaac Sim on aarch64 needs the system OpenMP runtime and libcarb preloaded.
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:$VIRTUAL_ENV/lib/python3.12/site-packages/omni/client/libcarb.so"
export PYTHONPATH="$PROTOMOTIONS_DIR:${PYTHONPATH:-}"
export TMPDIR=/tmp
export MPLCONFIGDIR="/tmp/matplotlib_rank_${SLURM_PROCID}"
export PROTOMOTIONS_COLLISION_CACHE_DIR="$OUTPUT_ROOT/collision_cache"
export PROTOMOTIONS_DEBUG_STAGES="${PROTOMOTIONS_DEBUG_STAGES:-1}"
export PROTOMOTIONS_KIT_LOG_LEVEL="${PROTOMOTIONS_KIT_LOG_LEVEL:-warning}"
mkdir -p "$MPLCONFIGDIR"
cd "$PROTOMOTIONS_DIR"

common_args=(
  --distillation-sources "$SOURCES"
  --num-envs "$NUM_ENVS"
  --batch-size "$BATCH_SIZE"
  --hoi-num-envs "$HOI_NUM_ENVS"
  --hoi-batch-size "$HOI_BATCH_SIZE"
  --rollout-steps "$ROLLOUT_STEPS"
  --student-psi "$STUDENT_PSI"
)
if [[ -n "$HOI_TEACHER_FILTER" ]]; then
  common_args+=(--hoi-teacher-filter "$HOI_TEACHER_FILTER")
fi
if [[ -f "$OUTPUT_ROOT/$RUN_NAME/last.ckpt" ]]; then
  common_args+=(--resume "$OUTPUT_ROOT/$RUN_NAME/last.ckpt")
fi

if [[ "$MODE" == check ]]; then
  exec python -m protomotions.train_multi_source_distill "${common_args[@]}" \
    --check-config-only --world-size "$PREFLIGHT_RANKS"
fi

train_args=(
  --output-dir "$OUTPUT_ROOT/$RUN_NAME"
  --learning-rate "$LEARNING_RATE"
  --normalizer-min-std "$NORMALIZER_MIN_STD"
  --iterations "$ITERATIONS"
  --save-every 50
  --eval-every "$EVAL_EVERY"
  --kit-log-dir "$OUTPUT_ROOT/$RUN_NAME/kit_logs"
  --ujitso-cache-dir /tmp/protomotions_ujitso
  --ujitso-cache-budget-mb 1024
  --timeout-minutes 180
)
if [[ "$NORMALIZE_LATENT" == 0 ]]; then
  train_args+=(--no-normalize-latent)
else
  train_args+=(--normalize-latent)
fi
if [[ "$EVALUATE_TEACHERS" == 1 ]]; then
  train_args+=(--evaluate-teachers)
fi
if [[ "$LOCO_FRAMES_ON_CPU" == 1 ]]; then
  train_args+=(--locomotion-frames-on-cpu)
fi
if [[ "$TRAIN_TF32" == 1 ]]; then
  train_args+=(--train-tf32)
fi
if [[ "$USE_WANDB" == 1 ]]; then
  export WANDB_DIR="$OUTPUT_ROOT/$RUN_NAME"
  export WANDB_CACHE_DIR="$OUTPUT_ROOT/wandb_cache"
  export WANDB_CONFIG_DIR="$OUTPUT_ROOT/wandb_config"
  # wandb spawns its service with this interpreter; sys.executable is not usable
  # once Kit is running.
  export WANDB__EXECUTABLE="$VIRTUAL_ENV/bin/python"
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
  train_args+=(--use-wandb --wandb-project "$WANDB_PROJECT")
fi

# Report the real exit status: a signal (128+n) otherwise looks like a clean exit.
set +e
python -u -m protomotions.train_multi_source_distill "${common_args[@]}" "${train_args[@]}"
status=$?
set -e
echo "[rank ${SLURM_PROCID:-?}] training process exited with status $status" >&2
exit $status
INNER
)

echo "Clariden preflight: code, data, checkpoints, action contract, and rank assignment"
MODE=check srun --ntasks=1 --environment="$EDF" bash -c "$inner_script"

echo "Starting $PREFLIGHT_RANKS-rank training in $RUN_DIR"
MODE=train srun --kill-on-bad-exit=1 --wait=60 --environment="$EDF" bash -c "$inner_script"
