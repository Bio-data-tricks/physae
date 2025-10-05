# physae

## Entraînement distribué

L'entraînement Lightning détecte désormais automatiquement le nombre de GPU et
de nœuds disponibles à partir des variables d'environnement (`WORLD_SIZE`,
`LOCAL_WORLD_SIZE`, `SLURM_GPUS_ON_NODE`, etc.). Si plusieurs processus sont
présents, la stratégie `DDP` est sélectionnée avec `find_unused_parameters=False`
et l'adresse maître est définie via `MASTER_ADDR`/`MASTER_PORT`.

Vous pouvez forcer la configuration via :

```bash
export PHYS_AE_DEVICES=2        # nombre de GPU par nœud
export PHYS_AE_NUM_NODES=1      # nombre de nœuds
```

Les essais Optuna conservent un entraînement mono-GPU grâce à la réinitialisation
des paramètres Lightning (`devices=1`, `strategy="auto"`).
