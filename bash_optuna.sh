#!/usr/bin/env bash
#SBATCH --job-name=physae-optuna
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log
## Optionnel : lancer N workers via un job array
## #SBATCH --array=0-7

set -euo pipefail

mkdir -p logs

# Valeurs optionnelles configurables directement depuis le script / l'environnement.
# Par exemple :
#   N_TRAIN=50000 STAGEA_EPOCHS=10 MAX_TRIALS=20 sbatch bash_optuna.sh
N_TRAIN=${N_TRAIN:-}
N_VAL=${N_VAL:-}
N_POINTS=${N_POINTS:-}
STAGEA_EPOCHS=${STAGEA_EPOCHS:-}
STAGEB1_EPOCHS=${STAGEB1_EPOCHS:-}
STAGEB2_EPOCHS=${STAGEB2_EPOCHS:-}

# --- Préparation de l'environnement logiciel (adapter à votre cluster) ---
# module load cuda/12.1
# module load python/3.10
# source /path/to/venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- Stockage partagé pour le journal Optuna ---
JOURNAL_DIR=${JOURNAL_DIR:-${SCRATCH:-$SLURM_SUBMIT_DIR}/optuna_physae}
mkdir -p "${JOURNAL_DIR}"
STUDY_NAME=${STUDY_NAME:-physae_hpo}
JOURNAL_PATH=${JOURNAL_PATH:-${JOURNAL_DIR}/${STUDY_NAME}.journal}

# --- Dossiers pour les logs et sorties par essai ---
LOG_DIR=${LOG_DIR:-$SLURM_SUBMIT_DIR/optuna_logs}
mkdir -p "${LOG_DIR}"
TRIAL_OUTPUT=${TRIAL_OUTPUT:-${SCRATCH:-$SLURM_SUBMIT_DIR}/optuna_runs}
mkdir -p "${TRIAL_OUTPUT}"

ARGS=(
  --storage "${JOURNAL_PATH}"
  --study-name "${STUDY_NAME}"
  --log-dir "${LOG_DIR}"
  --output-dir "${TRIAL_OUTPUT}"
  --sampler "${OPTUNA_SAMPLER:-tpe}"
  --pruner "${OPTUNA_PRUNER:-median}"
)

if [[ -n "${OPTUNA_SEED:-}" ]]; then
  ARGS+=(--seed "${OPTUNA_SEED}")
fi
if [[ "${TRIALS_PER_WORKER:-0}" -gt 0 ]]; then
  ARGS+=(--trials-per-worker "${TRIALS_PER_WORKER}")
fi
if [[ "${MAX_TRIALS:-0}" -gt 0 ]]; then
  ARGS+=(--max-trials "${MAX_TRIALS}")
fi
if [[ "${TIMEOUT_SECONDS:-0}" -gt 0 ]]; then
  ARGS+=(--timeout "${TIMEOUT_SECONDS}")
fi
if [[ -n "${OPTUNA_LOG_LEVEL:-}" ]]; then
  ARGS+=(--log-level "${OPTUNA_LOG_LEVEL}")
fi

if [[ -n "${N_TRAIN}" ]]; then
  ARGS+=(--n-train "${N_TRAIN}")
fi
if [[ -n "${N_VAL}" ]]; then
  ARGS+=(--n-val "${N_VAL}")
fi
if [[ -n "${N_POINTS}" ]]; then
  ARGS+=(--n-points "${N_POINTS}")
fi

if [[ -n "${STAGEA_EPOCHS}" ]]; then
  ARGS+=(--stageA-epochs "${STAGEA_EPOCHS}")
fi
if [[ -n "${STAGEB1_EPOCHS}" ]]; then
  ARGS+=(--stageB1-epochs "${STAGEB1_EPOCHS}")
fi
if [[ -n "${STAGEB2_EPOCHS}" ]]; then
  ARGS+=(--stageB2-epochs "${STAGEB2_EPOCHS}")
fi

srun --cpu-bind=cores python -u optuna_phisae.py "${ARGS[@]}"
