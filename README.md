# PhysAE Optuna orchestration

Ce dépôt contient les scripts nécessaires pour lancer les recherches d'hyperparamètres Optuna de PhysAE sur un cluster SLURM.

## Lancer un job Optuna

```bash
# Exemple complet avec ajustement du dataset et des epochs de chaque stage
N_TRAIN=50000 \
N_VAL=4000 \
N_POINTS=256 \
STAGEA_EPOCHS=24 \
STAGEB1_EPOCHS=12 \
STAGEB2_EPOCHS=18 \
sbatch bash_optuna.sh
```

Variables d'environnement utiles :

- `MAX_TRIALS`, `TRIALS_PER_WORKER` et `TIMEOUT_SECONDS` pour limiter la durée ou le nombre d'essais.
- `OPTUNA_SEED`, `OPTUNA_SAMPLER`, `OPTUNA_PRUNER` pour contrôler le comportement de la recherche.
- `JOURNAL_DIR`, `STUDY_NAME`, `LOG_DIR`, `TRIAL_OUTPUT` pour personnaliser les chemins des sorties.

## Paramètres des stages (A, B1, B2)

Optuna échantillonne les hyperparamètres suivants pour chaque stage. Les valeurs ci-dessous donnent un exemple réaliste ainsi que les bornes explorées :

### Stage A

| Paramètre      | Exemple  | Intervalle recherché |
|----------------|----------|----------------------|
| `epochs`       | 22       | 15 → 30              |
| `base_lr`      | 1.5e-4   | 5e-5 → 3e-4          |
| `refiner_lr`   | 2e-5     | 1e-6 → 5e-5          |

### Stage B1

| Paramètre        | Exemple | Intervalle recherché |
|------------------|---------|----------------------|
| `epochs`         | 12      | 8 → 20               |
| `refiner_lr`     | 1.2e-4  | 5e-5 → 5e-4          |
| `delta_scale`    | 0.12    | 0.05 → 0.20          |
| `refine_steps`   | 2       | 1 → 3                |

### Stage B2

| Paramètre        | Exemple | Intervalle recherché |
|------------------|---------|----------------------|
| `epochs`         | 18      | 10 → 25              |
| `base_lr`        | 6e-5    | 1e-5 → 1e-4          |
| `refiner_lr`     | 1.5e-5  | 5e-6 → 5e-5          |
| `delta_scale`    | 0.10    | 0.05 → 0.15          |
| `refine_steps`   | 2       | 1 → 3                |

Ces exemples correspondent aux paramètres maximisés par Optuna et permettent de rapidement interpréter les résultats dans les logs (`optuna_logs/worker_*.log`).

## Logs et sorties

- `logs/%x_%j.log` : sortie combinée `stdout`/`stderr` du job SLURM.
- `optuna_logs/worker_*.log` : journal détaillé de chaque worker Optuna.
- `optuna_runs/` : répertoires de sortie générés par chaque essai Optuna.
