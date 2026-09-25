#!/bin/bash
#SBATCH --job-name=hoi_pvq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=rtx_4090:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=80G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/lijingy/joint_pvq_results/hoi_pvq_%j.log
#SBATCH --error=/cluster/scratch/lijingy/joint_pvq_results/hoi_pvq_%j.log

export RUN_NAME="${RUN_NAME:-hoi_pvq_v1}"
export EVALUATE_TEACHERS="${EVALUATE_TEACHERS:-1}"
export SOURCES="${SOURCES:-examples/experiments/gcc/s1_hoi_pvq_sources_euler_4gpu.yaml}"
source "${PROTOMOTIONS_DIR:-/cluster/home/lijingy/ProtoMotions}/examples/experiments/gcc/s1_joint_pvq_euler.sbatch"
