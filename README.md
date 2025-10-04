# physae

## Lancement des entraînements Optuna

Le script SLURM `soumission.txt` permet désormais de paramétrer depuis la ligne de
commande le nombre de trials, le nombre d'epochs et les tailles des jeux de
données synthétiques pour chaque étape.

Exemple d'appel avec `sbatch` :

```bash
sbatch soumission.txt \
  --trials-a 150 \
  --trials-b 120 \
  --epochs-a 40 \
  --epochs-b 30 \
  --train-samples 200000 \
  --val-samples 1000 \
  --retrain-samples 1500000
```

Utilisez `sbatch soumission.txt --help` pour afficher la liste complète des
options disponibles et leurs valeurs par défaut.
