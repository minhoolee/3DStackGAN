3DStackGAN
==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile                  <- Makefile with commands like `make data` or `make train`
    ├── README.md                 <- The top-level README for developers using this project.
    ├── data
    │   ├── external              <- Data from third party sources.
    │   ├── interim               <- Intermediate data that has been transformed.
    │   ├── processed             <- The final, canonical data sets for modeling.
    │   └── raw                   <- The original, immutable data dump.
    │
    ├── docs                      <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                    <- Trained and serialized models, model predictions, or model summaries
    │   ├── csv                   <- CSV logs of epoch and batch runs
    │   ├── json                  <- JSON representation of the models
    │   ├── predictions           <- Predictions generated the train models and their best weights
    │   ├── weights               <- Best weights for the models
    │   └── yaml                  <- YAML representation of the models
    │
    ├── notebooks                 <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                the creator's initials, and a short `-` delimited description, e.g.
    │                                `1.0-jqp-initial-data-exploration`.
    │
    ├── references                <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures               <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt          <- The requirements file for reproducing the analysis environment, e.g.
    │                                generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                  <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                       <- Source code for use in this project.
    │   ├── __init__.py           <- Makes src a Python module
    │   │
    │   ├── data                  <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features              <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── logging               <- Scripts to improve python logging
    │   │   └── log_utils.py
    │   │
    │   ├── models                <- Scripts to train and test models and then use trained models to make
    │   │   │                         predictions
    │   │   ├── create_model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization         <- Scripts to create exploratory and results oriented visualizations
    │       ├── stats.py
    │       └── visualize.py
    │
    └── tox.ini                   <- tox file with settings for running tox; see tox.testrun.org

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
