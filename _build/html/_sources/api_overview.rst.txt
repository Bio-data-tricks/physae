Panorama des modules
====================

La bibliothèque est organisée autour de quelques modules clés :

:mod:`physae.factory`
    Fournit :func:`physae.factory.build_data_and_model` qui instancie le modèle,
    prépare les jeux de données et applique les paramètres YAML.

:mod:`physae.dataset`
    Contient :class:`physae.dataset.SpectraDataset` qui génère des paires
    ``(spectre, paramètres)`` en appelant le moteur physique et en injectant
    du bruit réaliste via :func:`physae.noise.add_noise_variety`.

:mod:`physae.model`
    Implémente :class:`physae.model.PhysicallyInformedAE`, un module Lightning
    avec deux têtes :

    * une tête de reconstruction spectrale (perte ``ReLoBRaLoLoss`` issue de
      :mod:`physae.losses`) ;
    * une tête de prédiction de paramètres normalisés avec refinements
      itératifs (``set_stage_mode`` et ``refine_predictions``).

:mod:`physae.training`
    Regroupe les fonctions ``train_stage_A/B1/B2`` qui appliquent des schémas de
    gel/dégel adaptés. Le coeur ``train_stage_custom`` accepte des callbacks
    Lightning standards et gère la sauvegarde/reprise.

:mod:`physae.optimization`
    Enveloppe Optuna pour la recherche d'hyperparamètres. La fonction
    :func:`physae.optimization.optimise_stage` lit la section ``optuna`` des
    YAML, construit automatiquement l'espace de recherche et retourne un objet
    :class:`optuna.study.Study`.

:mod:`physae.physics`
    Offre les primitives de simulation (fonction de Faddeeva, profil de Pine,
    line mixing). La fonction ``batch_physics_forward_multimol_vgrid`` génère les
    transmissions spectrales sur la grille ``v_grid``.

:mod:`physae.config`
    Centralise les noms des paramètres physiques, les bornes de normalisation et
    des utilitaires pour manipuler les intervalles (``map_ranges`` et
    ``expand_interval``).

:mod:`physae.config_loader`
    Lecture et fusion des fichiers YAML. Les fonctions ``merge_dicts`` et
    ``coerce_sequence`` garantissent une transformation cohérente entre YAML et
    objets Python.

:mod:`physae.optimizers`
    Fournit l'implémentation de ``Lion`` (optimiseur "Lookahead AdamW") ainsi
    que les wrappers nécessaires pour :mod:`torch.optim`.

Pour obtenir une documentation API exhaustive, activez ``sphinx.ext.autodoc``
et ajoutez ``.. automodule::`` dans vos pages personnalisées. Les options sont
préconfigurées dans ``docs/conf.py``.
