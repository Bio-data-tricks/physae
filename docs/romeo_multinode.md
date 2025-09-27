# Déploiement multi-nœuds sur Romeo

Ce guide résume la marche à suivre pour orchestrer l'optimisation Optuna, l'entraînement
multi-étapes et l'inférence sur le supercalculateur **Romeo**.

## Pré-requis

- Conda/Anaconda disponible sur les nœuds de calcul.
- Modules `cuda` et bibliothèques PyTorch Lightning / Optuna installés dans l'environnement
  chargé par le script de soumission.
- Accès à un espace de travail partagé pour écrire les checkpoints et sorties Optuna.

## Script Slurm fourni

Le script `scripts/romeo_multinode.sh` est prêt à l'emploi. Il enchaîne automatiquement :

1. L'optimisation du stage A.
2. L'optimisation séquentielle des stages B1/B2 (`--stages B`).
3. L'entraînement complet (A → B1 → B2) avec Lightning en mode DDP multinœud.
4. Une phase d'inférence sur le jeu de validation pour valider le modèle final.

### Personnalisation minimale

```bash
sbatch --nodes=4 --gres=gpu:4 --time=12:00:00 scripts/romeo_multinode.sh
```

- Adapter `--nodes`, `--gres` et `--time` selon le quota.
- Ajuster le nombre d'essais Optuna via `--n-trials` dans le script si nécessaire.
- Les résultats sont stockés dans `results/optuna`, les checkpoints dans `results/checkpoints`
  et les prédictions d'inférence dans `results/inference`.

## Utilisation manuelle du CLI

Le module `physae.cli` expose trois sous-commandes :

### Optimisation

```bash
python -m physae.cli optimise --stages A B --n-trials 40 --output-dir results/optuna
```

- `--stages A` optimise uniquement le stage A.
- `--stages B` déroule automatiquement B1 puis B2.
- Les figures Optuna sont sauvegardées dans `results/optuna/stage_<X>/`.

### Entraînement

```bash
python -m physae.cli train \
  --stages A B \
  --studies-dir results/optuna \
  --ckpt-dir results/checkpoints \
  --accelerator gpu \
  --strategy ddp \
  --devices 4 \
  --num-nodes 2
```

- Relit automatiquement `best_stage_params.yaml` et `best_data_overrides.yaml`.
- Les métriques par stage peuvent être exportées avec `--metrics-out chemin.json`.

### Inférence

```bash
python -m physae.cli infer \
  --checkpoint results/checkpoints/stage_B2.ckpt \
  --output results/inference/val_predictions.csv \
  --device cuda
```

- Calcule les erreurs relatives (%) et les sauvegarde en CSV.
- `--metrics-out` permet d'écrire un résumé (moyenne/médiane) en JSON.

## Conseils pour Romeo

- Définir `MASTER_ADDR` et `MASTER_PORT` (déjà fait dans le script) pour la communication DDP.
- Définir `NCCL_DEBUG=INFO` lors de la mise au point.
- Utiliser `--precision 16-mixed` pour réduire l'empreinte mémoire GPU si besoin.
- Vérifier que `results/` pointe vers un répertoire visible depuis tous les nœuds.

Bon calcul !

