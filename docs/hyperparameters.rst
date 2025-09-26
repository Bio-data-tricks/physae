Hyperparamètres et réglages
===========================

Cette section décrit les paramètres disponibles dans les fichiers YAML et leurs
contreparties dans le code Python.

Configuration des données
-------------------------

Le fichier :mod:`physae/configs/data/default.yaml <physae/configs/data/default.yaml>`
définit la génération de spectres.

``n_points``
    Taille du maillage spectral passé à :class:`physae.dataset.SpectraDataset`.
    Impose la dimension d'entrée du réseau (``n_points`` est également transmis
    à :class:`physae.model.PhysicallyInformedAE`).
``n_train`` / ``n_val``
    Nombre d'échantillons synthétiques générés à la volée pour les ensembles
    d'entraînement et de validation.
``batch_size``
    Taille du lot PyTorch dans les :class:`~torch.utils.data.DataLoader`.
``train_ranges_base`` et ``train_ranges_expand``
    Définissent les intervalles de sampling. Le module :mod:`physae.config`
    applique ``expand_interval`` afin d'élargir les bornes pour chaque
    paramètre (méthode logarithmique pour ``mf_CH4``).
``val_ranges``
    Intervalles strictement contenus dans ``train_ranges`` pour l'évaluation.
``noise``
    Paramètres stochastiques consommés par :func:`physae.noise.add_noise_variety`.
    Les clés ``std_add_range``, ``std_mult_range`` contrôlent les composantes
    gaussiennes tandis que ``p_drift``, ``p_fringes`` et ``p_spikes`` activent
    des perturbations structurées.
``predict_list`` / ``film_list``
    Sélection des paramètres prédit par le modèle et de ceux utilisés comme
    conditionnement FiLM. Les valeurs sont transmises à
    :class:`physae.model.PhysicallyInformedAE` via :func:`physae.factory.build_data_and_model`.
``lrs``
    Paire ``(base_lr, refiner_lr)`` utilisée pour initialiser les taux
    d'apprentissage des optimisateurs principaux et du raffineur.
``model``
    Sous-clefs ``encoder`` et ``refiner`` qui contrôlent la largeur, la
    profondeur et le facteur ``expand_ratio`` des blocs EfficientNet (cf.
    :class:`physae.models.EfficientNetEncoder`).

Configurations d'étape
----------------------

Les fichiers ``physae/configs/stages/stage_*.yaml`` pilotent
:func:`physae.training.train_stage_custom`.

``epochs``
    Nombre d'époques Lightning.
``base_lr`` / ``refiner_lr``
    Taux utilisés respectivement par l'encodeur (``model.base_lr``) et la tête de
    raffinement.
``train_base`` / ``train_heads`` / ``train_film`` / ``train_refiner``
    Indiquent quelles sous-parties du modèle sont dégelées. Ils déclenchent les
    appels à ``requires_grad_(True)`` dans :func:`physae.training._apply_stage_freeze`.
``refine_steps`` / ``delta_scale``
    Contrôlent le raffinement itératif : ``refine_steps`` définit le nombre de
    mises à jour successives, ``delta_scale`` la magnitude appliquée aux
    décalages prédits.
``use_film`` / ``film_subset`` / ``heads_subset``
    Permettent d'activer sélectivement la modulation FiLM et de limiter l'entraînement
    à certaines têtes de sortie.
``baseline_fix_enable``
    Active les corrections de baseline gérées par :class:`physae.model.PhysicallyInformedAE`
    via ``baseline_fix_*``.
``optimizer``
    Choix entre ``adamw`` et ``lion`` (voir :mod:`physae.optimizers`). Les
    paramètres associés (``optimizer_weight_decay``, ``optimizer_beta1``,
    ``optimizer_beta2``) surchargent ``model.optimizer_*``.
``scheduler_eta_min`` / ``scheduler_T_max``
    Paramètres de :class:`torch.optim.lr_scheduler.CosineAnnealingLR` configuré
    dans :meth:`physae.model.PhysicallyInformedAE.configure_optimizers`.

Bonnes pratiques de sélection
-----------------------------

* **Étape A** : utilisez un ``base_lr`` plus élevé (1e-4 à 3e-4) pour apprendre
  rapidement les représentations, sans activer le raffinement.
* **Étape B1** : abaissez ``refiner_lr`` (1e-6 à 1e-4) et maintenez
  ``delta_scale`` dans ``[0.05, 0.15]`` pour stabiliser le raffinement.
* **Étape B2** : réactivez ``train_base`` pour affiner la base avec un
  ``base_lr`` réduit (3e-5 par défaut) et couplez-le à un ``refiner_lr`` environ
  dix fois plus faible.
* **Bruit** : augmentez ``p_drift`` ou ``fringe_amp_range`` pour tester la
  robustesse aux dérives instrumentales ; fixez ``freeze_noise=True`` dans
  :class:`physae.dataset.SpectraDataset` pour des benchmarks reproductibles.
* **Recherche Optuna** : pour limiter l'espace de recherche, réduisez les bornes
  des hyperparamètres en éditant la section ``optuna`` des fichiers YAML.
