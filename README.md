3D-MRI-Domain-Adoption
==============================

Model to learn the affine transformation from AX CMR towards SAX CMR.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

Contributions:
--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


Example:
--------
<p><small>Example project <a target="_blank" href="https://github.com/minority4u/keras_flask_deployment">Example datascience project build on tensorflow and flask</a>. #tensorflow # flask</small></p>


Setup native with OSX and Ubuntu
------------

- Precondition Python 3.6 locally installed
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)


- clone repository
```
git clone https://github.com/minority4u/keras_flask_deployment
cd keras_flask_deployment
```

- Install all Dependencies, and start the app (all in one), works with OSX and Linux
```
make run
```

Setup native with Windows
------------

- Precondition Python 3.6 locally installed
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)

- Clone Repository
```
git clone https://github.com/minority4u/keras_flask_deployment
cd keras_flask_deployment
```

```
pip install virtualenv
virtualenv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\app\app.py
```

Setup Docker
------------

- Precondition: Installed Docker Deamon (e.g.:  <a target="_blank" href="https://docs.docker.com/install/">Docker CE</a>)

- Make sure you have docker-compose installed (e.g.:  <a target="_blank" href="https://docs.docker.com/compose/install/">Docker-Compose</a>)

- Clone Repository
```
git clone https://github.com/minority4u/keras_flask_deployment
cd keras_flask_deployment
```
- Create and run Docker-Container
```
docker-compose -up docker-compose.yml
```


Train new Model
------------

- OSX/Linux
```
make train
```

- Windows
```
Python src\models\train_model.py
```
