# PhysAE

Ce dépôt expose une version modulaire du réseau PhysAE avec une configuration basée sur des fichiers YAML et des utilitaires d'optimisation. Vous trouverez les fichiers de configuration dans `physae/configs` ainsi qu'un carnet Jupyter d'exemple dans `notebooks/optimisation_physae.ipynb`.

Les modules peuvent maintenant être importés de manière paresseuse (`lazy`) ce qui évite de déclencher immédiatement des dépendances lourdes (PyTorch, Matplotlib, Optuna, …) lorsqu'on ne manipule que la configuration.

## Configurations YAML

* `physae/configs/data/default.yaml` définit les hyperparamètres de génération des données (tailles d'échantillons, intervalles physiques, bruit, etc.).
* `physae/configs/stages/stage_*.yaml` regroupe les paramètres d'entraînement pour les différentes phases (A, B1, B2) ainsi que les espaces de recherche Optuna associés.
* Le module `physae.validation` fournit des fonctions `validate_data_config` et `validate_stage_config` pour vérifier rapidement la cohérence des fichiers YAML (taille des intervalles, clés manquantes, etc.) sans initialiser PyTorch.

Les fonctions `build_data_and_model` et `train_stage_*` acceptent toujours un argument `config_overrides` permettant d'ajuster ponctuellement les réglages définis dans les fichiers YAML tout en conservant la structure existante. Pour les usages avancés, `physae.factory.prepare_training_environment` permet de réutiliser un même jeu de DataLoader entre plusieurs entraînements ou essais, et `physae.factory.instantiate_model` crée un nouveau modèle à partir des paramètres sauvegardés dans cet environnement.

## Optimisation avec Optuna

Le module `physae.optimization` introduit la fonction `optimise_stage` qui s'appuie sur Optuna pour explorer l'espace de recherche défini dans les fichiers de configuration. Un appel minimal ressemble à ceci :

```python
from physae.optimization import optimise_stage

study = optimise_stage(
    "A",
    n_trials=10,
    metric="val_loss",
    data_overrides={"n_train": 2048, "n_val": 256},
    stage_overrides={"epochs": 8},
    reuse_dataloaders=True,
)
print(study.best_params)
```

Le carnet `notebooks/optimisation_physae.ipynb` présente un flux complet : validation des fichiers YAML, construction d'un environnement de données partagé, entraînement rapide des phases A et B, puis optimisation automatique des hyperparamètres en réutilisant les DataLoader.
