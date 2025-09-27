Premiers pas
===========

Installation
------------

Le projet est basé sur PyTorch, PyTorch Lightning et Optuna. Pour reproduire
l'environnement, créez un nouvel environnement virtuel et installez les
modules requis :

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # sous Windows: .venv\\Scripts\\activate
   pip install -U pip
   pip install -e .[dev]

Le paquet ``physae`` est installable en mode editable : toutes les
modifications locales sont immédiatement disponibles sans réinstallation.

Structure du dépôt
------------------

``physae/``
    Contient les modules Python (génération de données, modèle Lightning,
    fonctions de physique, etc.).
``physae/configs``
    Fichiers YAML décrivant les hyperparamètres des données et des différentes
    étapes d'entraînement.
``notebooks/``
    Carnets Jupyter d'exemple, dont ``optimisation_physae.ipynb`` qui illustre
    un flux complet de formation + recherche d'hyperparamètres.
``docs/``
    La documentation Read the Docs présentée ici.

Prérequis physiques
-------------------

PhysAE cible la reconstruction de spectres CH\ :sub:`4` (méthane) sur un
maillage spectral défini par ``n_points``. Les paramètres physiques étudiés
correspondent aux entrées de :data:`physae.config.PARAMS`, notamment la
position de raie ``sig0``, la largeur ``dsig`` ou encore la fraction molaire
``mf_CH4``. La normalisation des paramètres est contrôlée par
:data:`physae.config.NORM_PARAMS`.

Ressources de données
---------------------

Aucun fichier externe n'est requis : le module :mod:`physae.factory`
construit automatiquement :

* les dictionnaires de transitions spectraux via
  :func:`physae.physics.parse_csv_transitions` ;
* les jeux de données synthétiques via :class:`physae.dataset.SpectraDataset` ;
* le module Lightning :class:`physae.model.PhysicallyInformedAE`.

Pour modifier la physique (ajout d'un gaz, nouvelles transitions, autre
maillage), consultez :doc:`gases`.
