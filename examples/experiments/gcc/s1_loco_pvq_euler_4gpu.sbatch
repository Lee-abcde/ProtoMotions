#!/bin/bash
#SBATCH --job-name=loco_pvq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=80G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/lijingy/joint_pvq_results/loco_pvq_%j.log
#SBATCH --error=/cluster/scratch/lijingy/joint_pvq_results/loco_pvq_%j.log

set -euo pipefail
export PROTOMOTIONS_DIR="${PROTOMOTIONS_DIR:-/cluster/home/lijingy/ProtoMotions}"
export INCLUDE_HOI=0
export SOURCES="${SOURCES:-examples/experiments/gcc/s1_loco_pvq_sources_euler_4gpu.yaml}"
export RUN_NAME="${RUN_NAME:-joint_pvq_loco_4gpu_v1}"
export USE_WANDB="${USE_WANDB:-1}"
exec bash "$PROTOMOTIONS_DIR/examples/experiments/gcc/s1_joint_pvq_euler.sbatch"
