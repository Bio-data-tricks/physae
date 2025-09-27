Variations expérimentales
=========================

Cette section propose des scénarios pour explorer les capacités de PhysAE.

Comparer différentes intensités de bruit
---------------------------------------

1. Dupliquez ``physae/configs/data/default.yaml`` sous un nouveau nom (ex.
   ``noisy.yaml``).
2. Multipliez ``std_add_range`` et ``std_mult_range`` par 5.
3. Lancez ``build_data_and_model(config_name="noisy")`` puis l'entraînement.
4. Évaluez l'impact sur ``val_loss`` et les métriques physiques (pression,
   température).

Tester plusieurs tailles d'échantillon
--------------------------------------

Utilisez ``data_overrides`` pour modifier ``n_train`` et ``n_val`` lors d'un
appel à :func:`physae.training.train_stage_A`. Par exemple :

.. code-block:: python

   model, (train_loader, val_loader), _ = build_data_and_model(
       config_overrides={"n_train": 2048, "n_val": 512}
   )

Puis relancez les trois étapes. Comparez la stabilité des pertes et la variance
des prédictions.

Balayer ``delta_scale``
----------------------

Pour évaluer la sensibilité du raffinement, utilisez Optuna ou une boucle
manuelle :

.. code-block:: python

   for delta in [0.05, 0.1, 0.15, 0.2]:
       metrics = train_stage_custom(
           model,
           train_loader,
           val_loader,
           stage_name="B1",
           epochs=12,
           base_lr=1e-6,
           refiner_lr=1e-5,
           train_base=False,
           train_heads=False,
           train_film=False,
           train_refiner=True,
           refine_steps=2,
           delta_scale=delta,
           return_metrics=True,
       )[1]
       print(delta, metrics["val_loss"])

Changer de tête FiLM
--------------------

Pour comprendre l'influence des conditionnements, modifiez ``film_subset`` :

* ``["T"]`` (par défaut B1) : FiLM basé sur la température.
* ``["P", "T"]`` (B2) : combinaison pression + température.
* ``None`` : FiLM désactivé (``use_film=False``).

Observez les écarts sur les paramètres les plus corrélés (moyenne absolue de
l'erreur sur ``mf_CH4`` et ``sig0``).

Sauvegarde systématique des métriques
------------------------------------

Couplez :class:`pytorch_lightning.callbacks.ModelCheckpoint` et
:class:`pytorch_lightning.callbacks.EarlyStopping` pour enregistrer chaque
variation. Les callbacks se passent via ``callbacks=[...]`` dans
``train_stage_custom``.
