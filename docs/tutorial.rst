Tutoriel complet
================

Ce tutoriel montre comment :

1. charger les configurations YAML ;
2. générer des données synthétiques ;
3. entraîner successivement les étapes ``A``, ``B1`` et ``B2`` ;
4. explorer l'espace d'hyperparamètres avec Optuna.

Préparation
-----------

.. code-block:: python

   from physae.factory import build_data_and_model
   from physae.training import train_stage_A, train_stage_B1, train_stage_B2

   model, train_loader, val_loader = build_data_and_model(
       config_path="physae/configs/data",
       config_name="default",
   )

La fonction :func:`physae.factory.build_data_and_model` lit le fichier YAML,
construit un :class:`~physae.dataset.SpectraDataset` pour l'entraînement et la
validation, et instancie :class:`physae.model.PhysicallyInformedAE` avec les
paramètres de base (taux d'apprentissage, refinements, film, etc.).

Entraînement par étapes
-----------------------

Étape A (pré-entraînement des têtes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = train_stage_A(
       model,
       train_loader,
       val_loader,
       epochs=20,          # surcharge optionnelle
       base_lr=2e-4,
       enable_progress_bar=True,
   )

Cette phase active ``train_base`` et ``train_heads`` (voir
:func:`physae.training.train_stage_custom`). Aucune étape de raffinement n'est
lancée (``refine_steps=0``).

Étape B1 (raffinement ciblé)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = train_stage_B1(
       model,
       train_loader,
       val_loader,
       refiner_lr=1e-5,
       delta_scale=0.12,
       film_subset=["T"],
   )

Seule la tête de raffinage est entraînée et les paramètres ``T`` sont utilisés
pour la modulation FiLM. Le raffinement applique un décalage de ``delta_scale``
sur les paramètres prédits.

Étape B2 (affinage conjoint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = train_stage_B2(
       model,
       train_loader,
       val_loader,
       base_lr=3e-5,
       refiner_lr=3e-6,
       film_subset=["P", "T"],
   )

Tous les sous-modèles sont dégelés. Utilisez ``heads_subset`` si vous souhaitez
ne raffiner qu'une partie des sorties.

Contrôle du matériel (CPU/GPU)
------------------------------

Les fichiers ``stage_*.yaml`` acceptent des arguments Lightning supplémentaires
pour sélectionner l'accélérateur et le nombre de périphériques. Ils sont
directement transmis à :func:`physae.training.train_stage_custom`.

.. code-block:: yaml

   accelerator: gpu   # ou "cpu" pour désactiver CUDA
   devices: 2         # nombre de GPU (ou liste [0, 1])
   precision: 16      # mix-precision si supporté

Lorsque ``devices`` est supérieur à 1 et que ``accelerator`` cible les GPU, la
stratégie ``ddp`` est activée automatiquement pour orchestrer le multi-processus
de Lightning. Spécifiez ``strategy`` dans le YAML si vous devez choisir une
variante différente (``ddp_spawn``, ``fsdp``, ...).

Sur un supercalculateur sans accès Internet, exécutez vos expériences via un
fichier de soumission. Exemple minimal SLURM :

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=physae-optuna
   #SBATCH --gpus=4
   #SBATCH --cpus-per-task=8
   #SBATCH --time=04:00:00

   module load anaconda
   source activate physae

   python - <<'PYTHON'
   from physae.optimization import optimise_stage

   study = optimise_stage(
       "A",
       n_trials=10,
       stage_overrides={
           "accelerator": "gpu",
           "devices": 4,
       },
   )
   print("Meilleurs hyperparamètres:", study.best_params)
   PYTHON

Recherche d'hyperparamètres avec Optuna
---------------------------------------

.. code-block:: python

   from physae.optimization import optimise_stage

   study = optimise_stage(
       "B2",
       n_trials=15,
       metric="val_loss",
       data_overrides={"n_train": 8192, "noise": {"train": {"p_drift": 0.2}}},
       stage_overrides={"epochs": 10},
       show_progress_bar=True,
   )
   print("Meilleurs paramètres:", study.best_params)
   print("Score:", study.best_value)

La clé ``data_overrides`` accepte toute structure conforme au YAML. Les
paramètres préfixés par ``data.`` dans la section ``optuna`` des fichiers YAML
sont automatiquement redirigés vers ``data_overrides`` (voir
:func:`physae.optimization.optimise_stage`).

Sauvegarde & reprise
--------------------

Les fonctions ``train_stage_*`` acceptent ``ckpt_in`` et ``ckpt_out`` pour
recharger un point de contrôle Lightning ou sauver les poids finaux. Combinez
les callbacks Lightning (ex. ``ModelCheckpoint``) avec ``callbacks=[...]`` pour
personnaliser vos expériences.
