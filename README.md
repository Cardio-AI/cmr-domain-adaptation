3D-MRI-Domain-Adaptation
==============================

This repository includes the code and notebooks to train a deep learning model for Unsupervised Domain Adaptation of 3D CMR images.

The trained model is able to transform an AX CMR into the patient specific SAX direction.
The following gif shows exemplary the learning progress of this model. Each frame shows the predicted AX2SAX prediction of the model after it is trained for one additional epoch.
![Unsupervised Domain adaptation learning](https://github.com/Cardio-AI/3d-mri-domain-adaption/blob/master/reports/ax_sax_learning_example.gif "learning progress") 

- The repository dependencies are saved as conda environment (environment.yaml) file. 
- The Deep Learning models/layers are build with TF 2.X.
- Unfortunately we are not allowed to make the data public accessible. 
- The corresponding paper is currently under review for the special issue call @ TMI (cf. <a target="_blank" href="https://www.embs.org/wp-content/uploads/2020/04/Special_Issue_CFP_DL4MI.pdf">TMI Special Issue Call</a>)
- An setup instruction is given here: [Setup hints](##Setup instructions (tested with OSX and Ubuntu))




## Paper:
This work was created for a TMI special Issue Call (<a target="_blank" href="https://www.embs.org/wp-content/uploads/2020/04/Special_Issue_CFP_DL4MI.pdf">Special Issue Call</a>): 
```
"Call for Papers IEEE Transactions on Medical ImagingSpecial Issue on Annotation-Efficient Deep Learning for Medical Imaging"
```
- The paper is currently within the review process (minor revision).
- The Bibtex info and a link to the paper will be added as soon as it got accepted and published.

- Title:
```
Unsupervised Domain Adaptation from Axial to Short-Axis Multi-Slice Cardiac MR Images by Incorporating Pretrained Task Networks
```

- Authors:
```
Sven Koehler, Tarique Hussain, Zach Blair, Tyler Huffaker, Florian Ritzmann, Animesh Tandon,
Thomas Pickardt, Samir Sarikouch, Heiner Latus, Gerald Greil, Ivo Wolf, Sandy Engelhardt
```


## Repository overview:

This repository splits the source code into 
- interactive notebooks (/notebooks), 
- python source modules (/src) and 
- the experiment related files such as the experiment configs (/reports) or trained models (/models).

Each experiment includes the following artefacts:
- One config file, which represents all experimental hyper-parameters which are neccessary to reproduce the experiment or to load it for later predictions
- Three tensorboard logfiles per training to keep track of the trainings, evaluation and visual output progress. 
- One dataframe which orchestrates the nrrd files with the metadata and the experiment splitting. 
- Model graph description either as json or tensorflow protobuf file and the corresponding model weights as h5 file.

The tensorflow model and layer definitions are within /src/models. 
The transformation layer is built on the neuron project, which is also part of the current Voxelmorph approach (https://github.com/voxelmorph/voxelmorph).

Use the Notebooks to interact (train, predict or evaluate) with the python functions.


## Project Structure

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
    │   │                     transform AX CMR into the SAX domain, apply the task network, 
    │   │                     transform the predicted mask back into the AX domain, 
    │   │                     undo the generator steps and save the prediction to disk   
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


##Setup instructions (tested with OSX and Ubuntu)

###Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

###Local setup
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


