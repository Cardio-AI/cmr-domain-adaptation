3D-MRI-Domain-Adoption
==============================

This repository includes the code and notebooks for an unsupervised AX--> SAX transformation model.
The repository environment is saved as conda environment file. The DL graph is build on tensorflow.

![Unsupervised Domain adaptation learning](https://github.com/Cardio-AI/3d-mri-domain-adaption/blob/master/reports/ax2sax_learning_example_.mp4 "learning progress") 

Overview:
--------
Figure of the final pipeline
Use the Notebooks to interact with the python functions within the src directories.

Project Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like 'make environment' or 'make requirement'
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metadata       <- Excel and csv files with additional metadata
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predicted      <- Model predictions, will be used for the evaluations
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── Dataset        <- call the dataset helper functions, analyze the datasets
    │   ├── Evaluate       <- Evaluate the model performance, create plots
    │   ├── Predict        <- Use the models and predict the segmentation
    │   ├── Train          <- Train a new model
    │   └── Test_Models    <- Tensorflow functional or subclassing tests
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── configs        <- Experiment config files as json
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── history        <- Tensorboard trainings history files
    │   └── tensorboard_logs  <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Helper functions that will be used by the notebooks.
        ├── data           <- create, preprocess and extract the nrrd files
        ├── models         <- Modelzoo, Modelutils and Tensorflow layers
        ├── utils          <- Metrics, callbacks, io-utils, notebook imports
        └── visualization  <- Plots for the data, generator or evaluations

Paper:
--------
- link to paper if accepted

- info for cite

Setup native with OSX or Ubuntu
------------
### Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

### Local setup
- Clone repository
```
git clone %repo-name%
cd %repo-name%
```
- Create a conda environment from environment.yaml (environment name will be ax2sax)
```
conda env create --file environment.yaml
```

- Activate environment
```
conda activate ax2sax
```
- Install a helper to automatically change the working directory to the project root directory
```
pip install --extra-index-url https://test.pypi.org/simple/ ProjectRoot
```
- Create a jupyter kernel from the activated environment, this kernel will be visible in the jupyter lab
```
python -m ipykernel install --user --name ax2sax --display-name "ax2sax kernel"
```






### The repository is split into /notebooks and python /src modules

All interaction (load data, train load models, predict on held out splits) are summarized into one notebook /Notebooks/Train/3D_AX_to_SAX_Rotation_cycle.ipynb.
The model and layer definitions are within /src/models. The transformation module is copyed and modified from the neuron project, which is also part of the current Voxelmorph approach (https://github.com/voxelmorph/voxelmorph).
