# 3D CMR-Domain-Adaptation

This repo contains code to train a deep learning model for **Unsupervised Domain Adaptation (UDA)** of 3D cardiac magnetic resonance (CMR) cine images to **transform from axial to short-axis orientation**. The task associated to the domain adaptation is to perform a **segmentation task via a pre-trained fixed network**, and the results are leveraged to guide the transformation process (rigid transform via spatial transformer networks).

The trained model is able to transform an axial (AX) CMR into the patient specific short-axis (SAX) direction. The model learns from paired AX/SAX CMR image pairs and a pre-trained SAX segmentation model.

The following gif exemplary visualizes the learning progress of this model. Slices along z-direction are shown horizontally. 

![Unsupervised Domain adaptation learning](https://github.com/Cardio-AI/3d-mri-domain-adaption/blob/master/reports/ax_sax_learning_example.gif "learning progress") 

Each temporal frame shows the AX2SAX prediction of the model after successive epochs. At the end of the learning process, the model is able to transform the data set such that it corresponds to a short-axis view, which can be segmented more reliably by the pre-trained short-axis segmentation module. Finally the inverse transformation could be applied to the segmentation, which results in a automatically segmented AX CMR. For the training of this transformation model, neither AX nor SAX ground truth segmentations are required. 

A special **Euler2Affine Layer** was implemented to restrict the transform of the spatial transformer network to be rigid and invertible.

# Overview

- The repository dependencies are saved as conda environment (environment.yaml) file. 
- The Deep Learning models/layers are build with TF 2.X.
- Setup instruction for the repository are given here: [Install requirements](https://github.com/Cardio-AI/3d-mri-domain-adaptation#setup-instructions-tested-with-osx-and-ubuntu)
- An overview of all files and there usage is given here: [Repository Structure](https://github.com/Cardio-AI/3d-mri-domain-adaptation#repository-structure)

# Paper:

Please cite the following paper if you use/modify or adapt part of the code from this repository:

S. Koehler, T. Hussain, Z. Blair, T. Huffaker, F. Ritzmann, A. Tandon,
T. Pickardt, S. Sarikouch, H. Latus, G. Greil, I. Wolf, S. Engelhardt, "Unsupervised Domain Adaptation from Axial to Short-Axis Multi-Slice Cardiac MR Images by Incorporating Pretrained Task Networks," in IEEE Transactions on Medical Imaging (TMI), early access, doi: 10.1109/TMI.2021.3052972.

- Link to original paper: [IEEE TMI paper link](https://ieeexplore.ieee.org/document/9328840) (accepted 13.1.2021)
- A pre-print version is available here: [arxiv-Preprint](http://arxiv.org/abs/2101.07653)

Bibtex:

>@article{Koehler_2021, <br>
>   title={Unsupervised Domain Adaptation from Axial to Short-Axis Multi-Slice Cardiac MR Images by Incorporating Pretrained Task Networks},<br>
>   ISSN={1558-254X},<br>
>   url={http://dx.doi.org/10.1109/TMI.2021.3052972},<br>
>   DOI={10.1109/tmi.2021.3052972},<br>
>   journal={IEEE Transactions on Medical Imaging},<br>
>   publisher={Institute of Electrical and Electronics Engineers (IEEE)},<br>
>   author={Koehler, Sven and Hussain, Tarique and Blair, Zach and Huffaker, Tyler and Ritzmann, Florian and Tandon, Animesh and Pickardt, Thomas and Sarikouch, Samir and Sarikouch, Samir and Latus, Heiner and Greil, Gerald and Wolf, Ivo and Engelhardt, Sandy},
>   year={2021},<br>
>   pages={1–1}<br>
>}


# Institutions:

>- [Heidelberg University Hospital, Artificial Intelligence in Cardiovascular Medicine (AICM) Group](https://www.klinikum.uni-heidelberg.de/chirurgische-klinik-zentrum/herzchirurgie/forschung/ag-artificial-intelligence-in-cardiovascular-medicine)
>- [German Competence network for Congenital heart defects](https://www.kompetenznetz-ahf.de/en/about-us/competence-network/)
>- [UT Southwestern Medical Center, Pediatric Cardiology](https://www.utsouthwestern.edu/education/medical-school/departments/pediatrics/divisions/cardiology/)
>- [Department of Computer Science, University Of Applied Science Mannheim](https://www.informatik.hs-mannheim.de/wir/menschen/professoren/prof-dr-ivo-wolf.html)


# How to use:

This repository splits the source code into: 
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


## Repository Structure

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

## Dataset
For this work a multi-centric heterogeneous cine-SSFPs CMR TOF data set from the [German Competence Network for Congenital Heart Defects](https://www.kompetenznetz-ahf.de/en/about-us/competence-network/)) was used (study identifier: NCT00266188, title: Non-invasive Imaging and Exercise Tolerance Tests in Post-repair Tetralogy of Fallot -Intervention and Course in Patients Over 8 Years Old). 
This TOF dataset constitutes one of the largest compiled data set of this pathology to date. 
The data was acquired at 14 different German sites between 2005-2008 on 1.5T and 3T machines; 
further descriptions can be found in [original study](https://www.ahajournals.org/doi/epub/10.1161/CIRCIMAGING.111.963637), [eprint previous work](https://arxiv.org/abs/2002.04392) [1],[2].

[1] Sarikouch S, Koerperich H, Dubowy KO, Boethig D, Boettler P, Mir TS, Peters B, Kuehne T, Beerbaum P; German Competence Network for Congenital Heart Defects Investigators. Impact of gender and age on cardiovascular function late after repair of tetralogy of Fallot: percentiles based on cardiac magnetic resonance. Circ Cardiovasc Imaging. 2011 Nov;4(6):703-11. doi: 10.1161/CIRCIMAGING.111.963637. Epub 2011 Sep 9. PMID: 21908707.

[2] Köhler, Sven, Animesh Tandon, Tarique Hussain, H. Latus, T. Pickardt, S. Sarikouch, P. Beerbaum, G. Greil, S. Engelhardt and Ivo Wolf. “How well do U-Net-based segmentation trained on adult cardiac magnetic resonance imaging data generalise to rare congenital heart diseases for surgical planning?” Medical Imaging: Image-Guided Procedures (2020).

## Setup instructions (Ubuntu)

### Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

### Local setup
Clone repository
```
git clone %repo-name%
cd %repo-name%
```
Create a conda environment from environment.yaml (environment name will be ax2sax)
```
conda env create --file environment.yaml
```

Activate environment
```
conda activate ax2sax
```
Install a helper to automatically change the working directory to the project root directory
```
pip install --extra-index-url https://test.pypi.org/simple/ ProjectRoot
```
Create a jupyter kernel from the activated environment, this kernel will be visible in the jupyter lab
```
python -m ipykernel install --user --name ax2sax --display-name "ax2sax kernel"
```


### Enable interactive widgets in Jupyterlab

Pre-condition: nodejs installed globally or into the conda environment. e.g.:
```
conda install -c conda-forge nodejs
```
Install the jupyterlab-manager which enables the use of interactive widgets
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

Further infos on how to enable the jupyterlab-extensions:
[JupyterLab](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension)



