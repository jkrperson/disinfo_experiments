import argparse
import lightning as L

from data.xfact import FakeNewsDataModule
from data.verafiles import VeraFilesNewsDataModule
from data.liarliar import LiarDataModule

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from model import NLPModel

import random
import numpy as np
import torch


def setup_logger(experiment_name, logger_directory="fakenews_detection"):
    """
    Sets up the TensorBoard logger.

    Returns:
        TensorBoardLogger: The configured logger.
    """
    return TensorBoardLogger(logger_directory, experiment_name)


def create_trainer(max_epochs, log_every_n_steps, gpus, logger):
    """
    Creates a PyTorch Lightning trainer.

    Args:
        max_epochs (int): The maximum number of epochs for training.
        log_every_n_steps (int): How often to log within steps.
        gpus (int): The number of GPUs to use.
        logger (TensorBoardLogger): The logger to use.

    Returns:
        Trainer: The configured trainer.
    """
    # log model only if `val_accuracy` increases
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy", mode="max", filename='best-checkpoint',  # Name of the checkpoint files
        save_top_k=2,  # Only keep the top 1 model
        verbose=True  # Print when a new checkpoint is saved
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    return L.Trainer(logger=logger, max_epochs=max_epochs, log_every_n_steps=log_every_n_steps, devices=gpus, callbacks=[lr_monitor, checkpoint_callback])


def train_model(trainer, model, datamodule):
    """
    Trains the model.

    Args:
        trainer (Trainer): The trainer to use.
        learning_rate (float): The learning rate for the optimizer.
        num_workers (int): The number of workers for the data loader.

    Returns:
        None
    """

    trainer.fit(model=model, datamodule=datamodule)

    best_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    trainer.test(model=best_model, datamodule=datamodule)

    print("Best model path:", trainer.checkpoint_callback.best_model_path)


def train_classifier_model(
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        experiment_name: str,
        max_epochs: int, 
    ):
    """
    Trains the supervised contrastive model.

    Args:
        max_epochs (int): The maximum number of epochs for training.
        log_every_n_steps (int): How often to log within steps.
        num_workers (int): The number of workers for the data loader.
        gpus (int): The number of GPUs to use.
        learning_rate (float): The learning rate for the optimizer.
        seed (int): The seed for random number generation.

    Returns:
        None
    """
    L.seed_everything(42)

    logger = setup_logger(experiment_name)

    trainer = create_trainer(max_epochs, 10, 1, logger)

    train_model(trainer, model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classifier Model")
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset")
    parser.add_argument("--experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
                        
    args = parser.parse_args()



    # model = SupConModel(
    #     model_name=args.model_name, 
    #     learning_rate=args.learning_rate,
    #     loss=SupConLoss(temperature=args.temperature)
    # )

    if args.dataset_name == "verafiles":
        datamodule = VeraFilesNewsDataModule("datasets/verafiles_dataset", num_worker=1, model=args.model_name)
    elif args.dataset_name == "xlm_fakenews":
        datamodule = FakeNewsDataModule("datasets/xlm_fakenews", num_worker=1, model=args.model_name)
    elif args.dataset_name == "liar":
        datamodule = LiarDataModule("datasets/liar_dataset", num_worker=1, model_name=args.model_name)



    train_classifier_model(args.max_epochs, args.log_every_n_steps, args.num_workers, args.gpus, args.learning_rate, args.seed)
