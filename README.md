3D-MRI-Domain-Adaptation
==============================

This repository includes the code and notebooks for an unsupervised AX--> SAX transformation model.
The repository environment is saved as conda environment file. The DL graph is build on tensorflow.

The following gif shows exemplary the learning progress of this model. Each frame shows the predicted AX2SAX prediction of the model after it is trained for one additional epoch.
![Unsupervised Domain adaptation learning](https://github.com/Cardio-AI/3d-mri-domain-adaption/blob/master/reports/ax_sax_learning_example.gif "learning progress") 

Overview:
--------
This repository splits the source code into interactive notebooks (/notebooks), python source modules (/src) and the experiment related files such as the experiment configs  (/reports).

Each experiment includes the following artefacts:
- One text logfile per experiment for all runtime specific information. 
- Three tensorboard logfiles per training to keep track of the trainings, evaluation and visual output progress. 
- One dataframe as abstraction layer, which orchestrated the dicom files for each experiment. 
- One config file, which collected the experiment setup such as the experiment name, the experiment-specific paths and the preprocessing-, model- and trainings-hyperparameters. 
- One model graph description as json file and one h5 checkpoint file with the best performing model weights.

The tensorflow model and layer definitions are within /src/models. 
The transformation layer is built on the neuron project, which is also part of the current Voxelmorph approach (https://github.com/voxelmorph/voxelmorph).

Use the Notebooks to interact (train, predict or evaluate) with the python functions.




Project Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like 'make environment'
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metadata       <- Excel and csv files with additional metadata are stored here
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predicted      <- Model predictions, will be used for the evaluation scripts
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized model definitions and weights
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── Dataset        <- Create, map, split or pre-process the dataset
    │   ├── Evaluate       <- Evaluate the model predictions, create dataframes and plots
    │   ├── Predict        <- Load an experiment config and a pre-trained model, 
    │         │                     transform AX CMR into the SAX domain, apply the task network, 
    │         │                     transform the predicted mask back into the AX domain, 
    │         │                     undo the generator steps and save the prediction to disk   
    │   └── Train          <- Inspect the generators, define an experiment config,
    │                              load and inject a task network, build the graph and train a new AX2SAX model
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── configs        <- Experiment config files as json with all hyperparameters and paths
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── history        <- Tensorboard trainings history files
    │   └── tensorboard_logs  <- Trainings-scalars and images of the intermediate predictions
    │
    ├── environment.yaml   <- Conda requirements file. Generated with `conda env export > environment.yml`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Python modules with classes and functions that will be orchestrated in the notebooks.
        ├── data           <- Python modules - Generators, preprocess and postprocess functions
        ├── models         <- Python modules - TF.keras 2.X Model and Layer definition
        ├── utils          <- Python modules - Metrics, losses, prediction, evaluation code, TF-callbacks and io-utils
        └── visualization  <- Python modules - Plot functions for the evaluation and data description

Paper:
--------
- <a target="_blank" href="https://www.embs.org/wp-content/uploads/2020/04/Special_Issue_CFP_DL4MI.pdf">In review by TMI</a>

- Cite will follow as soon as the paper is accepted and published

Setup instructions (tested with OSX and Ubuntu)
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


### Enable interactive widgets in Jupyterlab

- Pre-condition: nodejs installed globally or into the conda environment. e.g.:
```
conda install -c conda-forge nodejs
```
- Full documentation on hw to enable the jupyterlab-extensions:
<a target="_blank" href="https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension">JupyterLab</a>
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


