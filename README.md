# PhysAE

Ce dépôt expose une version modulaire du réseau PhysAE avec une configuration basée sur des fichiers YAML et des utilitaires d'optimisation. Vous trouverez les fichiers de configuration dans `physae/configs` ainsi qu'un carnet Jupyter d'exemple dans `notebooks/optimisation_physae.ipynb`.

## Documentation Read the Docs

Une documentation complète (installation, tutoriel, guide des hyperparamètres) est disponible dans le dossier `docs/` au format Sphinx. Pour la construire localement :

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

Les pages HTML seront générées dans `docs/_build/html`. Cette structure est compatible avec un déploiement direct sur https://readthedocs.io/.

## Configurations YAML

* `physae/configs/data/default.yaml` définit les hyperparamètres de génération des données (tailles d'échantillons, intervalles physiques, bruit, etc.).
* `physae/configs/stages/stage_*.yaml` regroupe les paramètres d'entraînement pour les différentes phases (A, B1, B2) ainsi que les espaces de recherche Optuna associés.

Les fonctions `build_data_and_model` et `train_stage_*` acceptent désormais un argument `config_overrides` permettant d'ajuster ponctuellement les réglages définis dans les fichiers YAML tout en conservant la structure existante.

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
)
print(study.best_params)
```

Le carnet `notebooks/optimisation_physae.ipynb` présente un flux complet : chargement des configurations YAML, entraînement rapide des phases A et B, puis optimisation automatique des hyperparamètres.
