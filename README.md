# Explaining COVID-19 and Thoracic Pathology Model Predictions by Identifying Informative Input Features
This is the source code for our [paper](). The repository is divided into 3 parts, _IBA_ contains code for attribution methods (Information Bottleneck), _model_ contains training script; dataset ([NIH ChestXray](https://nihcc.app.box.com/v/ChestXray-NIHCC), [BrixIA](https://brixia.github.io/)) and learned model; _evaluation_ contains codes for quantitative evaluations.

# Setup
1. Make sure you have conda installed, then create an environment using
`conda env create -f environment.yml`.
2. Follow the installation guide from [IBA](https://github.com/BioroboticsLab/IBA-paper-code) to install Information Bottleneck method.
3. Download BrixIA and NIH ChestXray images, the labels is included in this repository in _model/labels_

# Usage
We provide several Jupyter notebooks in each sub-folder (IBA/notebooks, model/notebooks)
## Model Training
In  _model/results_ we provide trained model for NIH ChestXray, BrixIA regression, and BrixIA classification scheme
In _model/notebooks_ their are notebooks to train, fine tune, and evaluate models
## Model Attribution 
We have included various notebooks to run Information Bottleneck on ImageNet, NIH ChestXray, and BrixIA dataset. The notebooks can be found in _IBA/notebooks_
## Evaluate Attribution Maps
To evaluate the correctness of attribution maps, we provide two quantitative evaluations, which are sensitivity-N and insertion/deletion. To batch run evaluations, use `source run_insertion_deletion.h` and `source run_sensitivity_n.h` respectively. Before run the evaluations, make sure the variable defined inside bash scripts have correct path assigned

