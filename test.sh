#!/bin/bash

#SBATCH --job-name=test_vflow
#SBATCH --array=0-2
#SBATCH --output=/n/tambe_lab_tier1/Everyone/CompositeFlow/slurm_logs/%x_%A_%a.out
#SBATCH --error=/n/tambe_lab_tier1/Everyone/CompositeFlow/slurm_logs/%x_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append

# --- Base Directory ---
BASE_DIR="/n/tambe_lab_tier1/Everyone/CompositeFlow"
SCRIPT_PATH="$(realpath $0)"
SUBMIT_DIR="$(pwd)"

mkdir -p ${BASE_DIR}/slurm_logs
mkdir -p ${BASE_DIR}/training_output

# --- Grid Values ---
ALGORITHMS=("vflow")
SRCTYPES=("medium-replay" "medium" "medium-expert")
ENVS=("walker2d-friction" "hopper-friction")
SEEDS=(0 1 2)

DGS=(0.01 0.1)                            # dynamics_gap_reward_scale values
FPS=(0.7 0.8)                            # filter_percent values

# --- Compute Indices ---
INDEX=$SLURM_ARRAY_TASK_ID

N_ALGOS=${#ALGORITHMS[@]}
N_SRCTYPES=${#SRCTYPES[@]}
N_ENVS=${#ENVS[@]}
N_SEEDS=${#SEEDS[@]}
N_DGS=${#DGS[@]}
N_FPS=${#FPS[@]}

SEED_IDX=$(( INDEX % N_SEEDS ))
ENV_IDX=$(( (INDEX / N_SEEDS) % N_ENVS ))
SRCTYPE_IDX=$(( (INDEX / (N_SEEDS*N_ENVS)) % N_SRCTYPES ))
ALGO_IDX=$(( (INDEX / (N_SEEDS*N_ENVS*N_SRCTYPES)) % N_ALGOS ))
DGS_IDX=$(( (INDEX / (N_SEEDS*N_ENVS*N_SRCTYPES*N_ALGOS)) % N_DGS ))
FPS_IDX=$(( (INDEX / (N_SEEDS*N_ENVS*N_SRCTYPES*N_ALGOS*N_DGS)) % N_FPS ))

# --- Final Values ---
SEED=${SEEDS[$SEED_IDX]}
ENV=${ENVS[$ENV_IDX]}
SRCTYPE=${SRCTYPES[$SRCTYPE_IDX]}
ALGO=${ALGORITHMS[$ALGO_IDX]}
DYNAMICS_GAP=${DGS[$DGS_IDX]}
FILTER=${FPS[$FPS_IDX]}


OUTPUT_DIR="${BASE_DIR}/training_output/seed_${SEED}"
mkdir -p ${OUTPUT_DIR}

echo "Running: algo=${ALGO}, src=${SRCTYPE}, env=${ENV}, seed=${SEED}, dgr=$DYNAMICS_GAP, fp=$FILTER"
echo "Output Dir: ${OUTPUT_DIR}"

# --- Activate Conda ---
source ~/.bashrc
conda activate final_offdynamics

# --- Run Training ---
python train.py \
    --policy ${ALGO} \
    --env ${ENV} \
    --mode 1 \
    --srctype ${SRCTYPE} \
    --shift_level 5.0 \
    --seed ${SEED} \
    --n_samples 30 \
    --checkpoint_freq 5000 \
    --dynamics_gap_reward_scale ${DYNAMICS_GAP} \
    --filter_percent ${FILTER} \
    --dir ${OUTPUT_DIR} \

echo "Job finished for task ID ${SLURM_ARRAY_TASK_ID}"