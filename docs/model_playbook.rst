Guide complet des modèles et des flux PhysAE
===========================================

.. contents::
   :local:
   :depth: 2

Aperçu général
--------------

Ce guide synthétise tous les éléments nécessaires pour entraîner, adapter et
exploiter les autoencodeurs physiques fournis par **PhysAE**. Il complète les
pages :doc:`hyperparameters` et :doc:`tutorial` en détaillant :

* l'architecture des modèles (encodeur, têtes, raffineur) et les variantes
  disponibles ;
* la liste hiérarchisée des hyperparamètres et leurs liens avec les fichiers
  YAML ;
* plusieurs recettes pratiques pour ajuster les flux d'entraînement en fonction
  de contraintes réelles (bruit accru, gaz supplémentaires, transfert) ;
* des exemples Jupyter autoportants couvrant entraînement, optimisation Optuna
  et génération de fichiers de soumission.

Architecture de référence
-------------------------

Les modules sont définis dans :mod:`physae.model` et
:mod:`physae.models.backbone`. Un modèle complet associe trois blocs
principaux :

1. **Encodeur spectral** – extrait une représentation 1D compacte à partir des
   spectres bruités. Les constructeurs enregistrés via
   :func:`physae.models.register_encoder` incluent ``efficientnet`` (base),
   ``efficientnet_large`` (capacité accrue), ``efficientnet_v2`` et
   ``convnext``. Chaque encodeur doit exposer un attribut ``feat_dim``
   renseignant la dimension de la représentation.
2. **Tête partagée + sorties** – le `shared head` projette les caractéristiques
   vers un espace latent (``shared_head_hidden_scale`` contrôle sa taille)
   avant d'alimenter soit une unique couche linéaire multi-sorties
   (``head_mode: single``) soit un dictionnaire de têtes indépendantes
   (``head_mode: multi``). Les paramètres à prédire sont spécifiés via
   ``predict_list`` dans les fichiers YAML.
3. **Raffineur itératif** – optionnel, il opère dans l'espace des paramètres
   physiques pour réduire les écarts résiduels entre la reconstruction et les
   spectres cibles. Les variantes suivent le même mécanisme de registre que
   l'encodeur (:func:`physae.models.register_refiner`). Le nombre d'itérations
   et l'échelle des mises à jour sont contrôlés par ``refine_steps`` et
   ``refine_delta_scale``.

Le module :class:`physae.model.PhysicallyInformedAE` orchestre ces composants,
assure les passages au modèle physique différentiable
(:func:`physae.physics.batch_physics_forward_multimol_vgrid`) et gère la
perte combinée reconstruction/paramètres (:class:`physae.losses.ReLoBRaLoLoss`).


Correspondance des hyperparamètres
----------------------------------

Le tableau suivant résume les hyperparamètres les plus utilisés, la section YAML
associée et l'attribut Python concerné.

+----------------------------+------------------------------------------+----------------------------------+
| Catégorie                  | Fichier YAML                             | Attribut / argument              |
+============================+==========================================+==================================+
| Taille du maillage         | ``configs/data/default.yaml`` → ``n_points`` | ``PhysicallyInformedAE.n_points`` |
+----------------------------+------------------------------------------+----------------------------------+
| Paramètres prédits         | ``configs/data/default.yaml`` → ``predict_list`` | ``PhysicallyInformedAE.predict_params`` |
+----------------------------+------------------------------------------+----------------------------------+
| Conditionnement FiLM       | ``configs/data/default.yaml`` → ``film_list`` | ``PhysicallyInformedAE.film_params`` |
+----------------------------+------------------------------------------+----------------------------------+
| Architecture encodeur      | ``configs/data/default.yaml`` → ``model.encoder`` | ``build_encoder`` + arguments    |
+----------------------------+------------------------------------------+----------------------------------+
| Architecture raffineur     | ``configs/data/default.yaml`` → ``model.refiner`` | ``build_refiner`` + arguments    |
+----------------------------+------------------------------------------+----------------------------------+
| Stratégie bruit            | ``configs/data/default.yaml`` → ``noise`` | :func:`physae.noise.add_noise_variety` |
+----------------------------+------------------------------------------+----------------------------------+
| Plan de stages             | ``configs/stages/stage_*.yaml``          | :func:`physae.training.train_stage_custom` |
+----------------------------+------------------------------------------+----------------------------------+
| Apprentissage différencié  | ``base_lr`` / ``refiner_lr`` dans ``stage_*.yaml`` | ``configure_optimizers``         |
+----------------------------+------------------------------------------+----------------------------------+
| Optimiseur & scheduler     | ``stage_*.yaml`` → ``optimizer`` / ``scheduler_*`` | :mod:`physae.optimizers`, ``CosineAnnealingLR`` |
+----------------------------+------------------------------------------+----------------------------------+

Les hyperparamètres détaillés (bornes, conseils d'exploration) sont décrits dans
:doc:`hyperparameters`. Ce guide fournit en plus des **recettes thématiques**
ci-dessous.

Recettes pratiques
------------------

Amplification du bruit instrument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Activez ``noise.train.p_drift = 0.3`` et augmentez ``fringe_amp_range`` pour
  simuler des dérives fortes.
* Réduisez ``refine_delta_scale`` à ``0.05`` dans les phases B pour éviter des
  corrections instables.
* Fixez ``baseline_fix_enable: true`` dans ``stage_B1.yaml`` ou ``stage_B2.yaml``
  afin de laisser le modèle corriger automatiquement la ligne de base.

Adaptation à de nouveaux gaz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Ajoutez les paramètres ciblés (ex. ``mf_CO2``) dans ``train_ranges_base`` et
  ``predict_list``.
* Mettez à jour ``poly_freq_<gaz>`` dans les métadonnées passées au modèle via
  :func:`physae.factory.build_data_and_model`.
* Ajustez ``shared_head_hidden_scale`` à ``0.65``–``0.8`` pour supporter la
  nouvelle tête.

Transfert sur spectromètre réel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Entraînez d'abord un stage A court (``epochs: 10``) sur un mélange synthétique
  proche, puis geler l'encodeur (``train_base: false``) et n'affiner que le
  raffineur avec ``refiner_lr`` plus élevé (``3e-5`` à ``1e-4``).
* Injectez un petit jeu de spectres réels en validation en utilisant
  ``Dataset.from_arrays`` et l'argument ``external_val_loader`` des fonctions de
  formation personnalisées.


Flux Jupyter prêts à l'emploi
-----------------------------

Trois carnets accompagnent ce guide dans le dossier :mod:`notebooks/`.

* ``01_quickstart_training.ipynb`` – construction des jeux de données, lancement
  d'un entraînement multi-stages condensé (epochs réduits) et sauvegarde des
  checkpoints.
* ``02_hyperparameter_sweeps.ipynb`` – configuration d'une étude Optuna,
  analyse des essais et export des meilleurs hyperparamètres dans un fichier
  ``yaml``.
* ``03_inference_and_submission.ipynb`` – chargement d'un checkpoint, exécution
  de l'inférence en lot et génération d'un fichier ``CSV`` conforme aux formats
  de soumission présentés ci-dessous.

Chaque carnet démarre par une cellule d'installation minimale et peut être
lancé depuis ``jupyter lab`` ou ``jupyter notebook`` après installation de
PhysAE en mode développement (`pip install -e .[docs]`).


Gabarits de soumission
----------------------

Deux formats de soumission types sont proposés dans ``docs/examples/submissions`` :

* ``submission_nominal.csv`` – pour des prédictions complètes (tous les
  paramètres présents dans ``predict_list``) ;
* ``submission_partial.json`` – pour des campagnes ne prédisant que certaines
  grandeurs et laissant le reste vide ou nul.

Ces fichiers sont générés automatiquement dans le carnet
``03_inference_and_submission.ipynb`` mais peuvent également être téléchargés
et remplis manuellement.

.. list-table:: Champs attendus dans les soumissions CSV
   :header-rows: 1

   * - Colonne
     - Description
   * - ``sample_id``
     - Identifiant unique du spectre (correspondant à la clé ``"id"`` du
       ``Dataset``).
   * - ``sig0`` / ``dsig`` / ``mf_CH4`` / ``P`` / ``T`` / ``baseline1`` / ``baseline2``
     - Valeurs physiques dénormalisées produites par
       :func:`physae.normalization.unnorm_param_torch`.

Pour les soumissions ``JSON``, chaque entrée suit la structure :

.. code-block:: json

   {
     "sample_id": "val_0001",
     "params": {
       "sig0": 3085.44,
       "dsig": 0.00153,
       "mf_CH4": 8.2e-06
     },
     "metadata": {
      "model": "PhysAE-efficientnet",
       "refine_steps": 2
     }
   }

Les scripts CLI :mod:`physae.cli` et les fonctions d'évaluation prennent en
charge ces deux formats pour automatiser la publication de résultats.


Aller plus loin
---------------

* Chaînez les appels à :func:`physae.training.train_stage_A`,
  :func:`physae.training.train_stage_B1` puis
  :func:`physae.training.train_stage_B2` pour rejouer la stratégie multi-stage
  standard (A → B1 → B2).
* Adaptez le scheduler cosinus en définissant ``scheduler_T_max`` à la somme
  des étapes pour aligner le cycle complet d'apprentissage.
* Surveillez les métriques retournées par :func:`physae.evaluation.evaluate_and_plot`
  pour détecter les biais systématiques et ajuster la pondération
  ``alpha_param``/``alpha_phys``.

