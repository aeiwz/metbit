metbit
======

Metbit is an python package to analyse metabolomics

.. _metbit-1:

metbit
======

Metbit is a Python package designed for the analysis of metabolomics
data. It provides a range of tools and functions to process, visualize,
and interpret metabolomics datasets. With Metbit, you can perform
various statistical analyses, identify biomarkers, and generate
informative visualizations to gain insights into your metabolomics
experiments. Whether you are a researcher, scientist, or data analyst
working in the field of metabolomics, Metbit can help streamline your
data analysis workflow and facilitate the interpretation of complex
metabolomics data.

How to install
==============

.. code:: bash

   pip install metbit

Example:
========

Import package into python
--------------------------

.. code:: python


   from metbit import opls_da, pca
   import pandas as pd
   import numpy as np

Load dataset
------------

For example dataset are generated by random

.. code:: python

   data = pd.DataFrame(np.random.rand(500, 50000))
   class_ = pd.Series(np.random.choice(['A', 'B', 'C'], 500), name='Group')
   time = pd.Series(np.random.choice(['1-wk', '2-wk', '3-wk', '4-wk'], 500), name='Time point')

   dataset = pd.concat
   # Assign X and target
   X = datasets.iloc[:, 2:]
   y = datasets['Group']
   time = datasets['Time point']
   features_name = list(X.columns.astype(float))

Perform PCA model
-----------------

.. code:: python


   pca_mod = pca(X = X, label = y, features_name=features_name, n_components=2, scale='pareto', random_state=42, test_size=0.3)
   pca_mod.fit()

Visualisation of PCA model
==========================

.. code:: python


   pca_mod.plot_observe_variance()

   pca_mod.plot_cumulative_observed()

   shape_ = {'1-wk': 'circle', '2-wk': 'square', '3-wk': 'diamond', '4-wk': 'cross'}

   pca_mod.plot_pca_scores(symbol=time, symbol_dict=shape_)

   pca_mod.plot_loading_()

   pca_mod.plot_pca_trajectory(time_=time, time_in_number={'1-wk': 0, '2-wk': 1, '3-wk': 2, '4-wk': 3}, color_dict={'A': '#636EFA', 'B': '#EF553B', 'C': '#00CC96'}, symbol_dict=shape_)

OPLS-DA model
=============

.. _import-package-into-python-1:

Import package into python
--------------------------

.. code:: python


   from metbit import opls_da, pca
   import pandas as pd
   import numpy as np

.. _load-dataset-1:

Load dataset
------------

For example dataset are generated by random

.. code:: python

   data = pd.DataFrame(np.random.rand(500, 50000))
   class_ = pd.Series(np.random.choice(['A', 'B'], 500), name='Group')

   datasets = pd.concat([class_, data], axis=1)


   # Assign X and target
   X = datasets.iloc[:, 2:]
   y = datasets['Group']
   time = datasets['Time point']
   features_name = list(X.columns.astype(float))

Perform OPLS-DA model
---------------------

.. code:: python


   opls_da_mod = opls_da(X=X, y=y,features_name=features_name, n_components=2, scale='pareto', kfold=3, estimator='opls', random_state=42):
           
   opls_da.fit()

   opls.permutation_test(n_permutataion=1000,cv=3, n_jobs=-1, verbose=10)

   opls_da.vip_scores()

Isualiseation of OPLs-DA model
------------------------------

.. code:: python


   opls_da_model.plot_oplsda_scores()

   opls_da_model.vip_plot()

   opls_da_model.plot_hist()

   opls_da_model.plot_s_scores()

   opls_da_model.plot_loading()