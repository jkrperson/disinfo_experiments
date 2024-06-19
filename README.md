# Fake News Detection Repository

This repository contains code for training and evaluating models for fake news detection. It includes various models, data modules, and scripts for running the experiments.

## Repository Structure

- `classification_trainer.py`: Script for training the classification model.
- `con_to_class_trainer.py`: Script for training the contrastive to classification model.
- `sup_con_trainer.py`: Script for training the supervised contrastive model.
- `test_model.py`: Script for testing the models.
- `data/`: Contains the data modules for different datasets.
- `models/`: Contains the model definitions.
- `notebooks/`: Contains Jupyter notebooks for data processing and model evaluation.
- `running_scripts/`: Contains bash scripts for running the experiments.
- `fakenews_detection`: Where the `Tensorboard` directory should be ran.

## Models

- `ClassifierModel`: The main classification model.
- `ContrastivePretrainedModel`: A contrastive model that uses pretrained embeddings.
- `SupConModel`: A supervised contrastive model.

## Data Modules

- `ContrastiveFakeNewsDataModule`: Data module for contrastive fake news dataset.
- `LiarContrastiveDataModule`: Data module for the Liar-Liar dataset.

## Running the Experiments

To replicate the experiments, run the bash scripts in the `running_scripts/` directory. For example:

```sh
bash running_scripts/continuous_script.sh
bash running_scripts/another_continuous_script.sh
```

Please ensure that you have the necessary dependencies installed and the appropriate data downloaded.

### Dependencies

This project requires Python 3.11 or later and the packages listed in `requirements.txt`. You can install them using:

```sh
pip install -r requirements.txt
```

### Data 

All the data is included in this repository.
