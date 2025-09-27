Gestion des gaz et transitions
==============================

Le module :mod:`physae.physics` modélise la génération spectrale en utilisant la
fonction :func:`physae.physics.batch_physics_forward_multimol_vgrid`. Par
défaut, la fabrique (:mod:`physae.factory`) configure un dictionnaire de
transitions contenant uniquement le CH\ :sub:`4`. Voici comment personnaliser la
physique.

Ajouter un nouveau gaz
----------------------

1. Récupérez les transitions spectrales (format HITRAN ou personnalisé).
2. Convertissez-les au format CSV semi-colon séparé attendu par
   :func:`physae.physics.parse_csv_transitions` (14 colonnes).
3. Instanciez manuellement :class:`physae.dataset.SpectraDataset` et
   :class:`physae.model.PhysicallyInformedAE` en fournissant un dictionnaire de
   transitions enrichi.

.. code-block:: python

   from physae.dataset import SpectraDataset
   from physae.model import PhysicallyInformedAE
   from physae.physics import parse_csv_transitions
   from physae import config

   poly_freq_coeffs = [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]
   train_ranges = {
       "sig0": (3085.43, 3085.46),
       "dsig": (0.001521, 0.00154),
       "mf_CH4": (2e-6, 2e-5),
       "mf_CO2": (5e-6, 5e-5),
       "baseline0": (0.99, 1.01),
       "baseline1": (-4e-4, -3e-4),
       "baseline2": (-4.1e-8, -3.0e-8),
       "P": (400, 600),
       "T": (303.15, 313.15),
   }
   transitions = {
       "CH4": parse_csv_transitions(ch4_csv_string),
       "CO2": parse_csv_transitions(open("co2.csv").read()),
   }
   train_dataset = SpectraDataset(
       n_samples=50000,
       num_points=800,
       poly_freq_CH4=poly_freq_coeffs,
       transitions_dict=transitions,
       sample_ranges=train_ranges,
   )
   model = PhysicallyInformedAE(
       n_points=800,
       param_names=config.PARAMS,
       poly_freq_CH4=poly_freq_coeffs,
       transitions_dict=transitions,
       predict_params=["sig0", "dsig", "mf_CH4", "mf_CO2", "P", "T"],
       film_params=["P", "T"],
   )

À l'heure actuelle, la fonction :func:`physae.factory.build_data_and_model` se
concentre sur le cas CH\ :sub:`4`. Pour des scénarios multi-gaz avancés,
réutilisez les utilitaires ``load_data_config`` et ``merge_dicts`` définis dans
:mod:`physae.config_loader` pour initialiser vos propres chargeurs de données.

Modifier le maillage spectral
-----------------------------

``build_data_and_model`` accepte l'argument ``n_points``. Celui-ci influence :

* la longueur des spectres générés par :class:`physae.dataset.SpectraDataset` ;
* la taille des tenseurs traités par :class:`physae.model.PhysicallyInformedAE` ;
* le polynôme ``poly_freq_CH4`` utilisé pour générer la grille ``v_grid``
  (fichier ``factory.py``).

Veillez à ajuster ``poly_freq_CH4`` si le domaine spectral change drastiquement.

Choisir les fractions molaires
------------------------------

Les fractions sont normalisées logarithmiquement par :func:`physae.config.map_ranges`.
Pour explorer des concentrations extrêmes, augmentez ``train_ranges_expand``
(*ex.* ``mf_CH4: 4.0`` double encore les bornes logarithmiques) et gardez
``val_ranges`` dans l'enveloppe pour éviter les erreurs ``strict_check``.

Contrôler la pression et la température
---------------------------------------

Les paramètres ``P`` et ``T`` influencent directement la largeur des raies via
:func:`physae.physics.batch_physics_forward_multimol_vgrid`. Les recommandations :

* définissez ``train_ranges`` suffisamment larges pour couvrir vos conditions
  expérimentales ;
* utilisez ``film_subset`` pour conditionner les raffinements FiLM sur les
  paramètres les plus corrélés aux variations du spectre (souvent ``P`` et ``T``).
