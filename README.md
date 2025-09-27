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

## Installation du package

Le dépôt suit désormais une structure standard de distribution Python avec un fichier `pyproject.toml`. Pour installer PhysAE et ses dépendances de base dans un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\Scripts\activate
pip install .
```

Une fois le package installé, la commande CLI `physae` devient disponible (identique au module `physae.cli`). Pour installer les dépendances nécessaires à la construction de la documentation :

```bash
pip install .[docs]
```

## Étendre les composants du modèle

Les éléments clés du modèle (encodeur principal et module de raffinement) sont enregistrés dans des registres afin de faciliter l'ajout de nouvelles variantes. Plusieurs constructeurs d'encodeur sont fournis par défaut :

* `efficientnet`, fidèle au backbone historique.
* `efficientnet_large`, une déclinaison plus profonde et plus large reposant sur les mêmes blocs MBConv pour des expériences à plus forte capacité.
* `efficientnet_v2`, qui combine blocs MBConv classiques et blocs *fused* inspirés d'EfficientNet V2 pour une meilleure efficacité.
* `convnext`, une alternative de type ConvNeXt avec normalisation de type LayerNorm et convolutions profondes pour augmenter la capacité représentative.

Ces implémentations sont déclarées dans `physae/models/backbone.py` tandis que les raffinements associés sont définis dans `physae/models/refiner.py` via les décorateurs `@register_encoder` et `@register_refiner`.

Pour ajouter un encodeur personnalisé, il suffit de définir une fonction de construction qui retourne un module `nn.Module` et d'utiliser le décorateur approprié :

```python
from physae.models import register_encoder

@register_encoder("mon_encodeur")
def build_mon_encodeur(**kwargs):
    return MonEncodeur1D(**kwargs)
```

La configuration YAML peut ensuite sélectionner ce nouvel encodeur :

```yaml
model:
  encoder:
    name: mon_encodeur
    params:
      largeur: 64
```

La même approche s'applique au raffineur via `register_refiner`. Les paramètres définis dans la section `model.encoder.params` (ou `model.refiner.params`) sont transmis au constructeur correspondant, ce qui permet d'expérimenter facilement avec de nouveaux blocs.

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
