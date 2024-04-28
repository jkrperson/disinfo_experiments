import argparse
import lightning as L

from data.xfact import FakeNewsDataModule
from data.verafiles import VeraFilesNewsDataModule
from data.liarliar import LiarDataModule
from data.fakenews_filipino import FakeTLDataModule

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from models.classification_model import ClassifierModel
from models.loss import SupConLoss
from models.sup_contrastive_model import SupConModel

import random
import numpy as np
import torch



def train_classifier_model(
        model: L.LightningModule,
        datamodule: L.LightningDataModule,
        experiment_name: str,
        max_epochs: int, 
    ):
    """
    Trains the classifier model.
    """
    L.seed_everything(42)

    logger = TensorBoardLogger("fakenews_detection", name=experiment_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy", mode="max", filename='best-checkpoint',  # Name of the checkpoint files
        save_top_k=2,  # Only keep the top 1 model
        verbose=True,  # Print when a new checkpoint is saved
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer =  L.Trainer(logger=logger, max_epochs=max_epochs, log_every_n_steps=10, devices=1, callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model=model, datamodule=datamodule)

    best_model = ClassifierModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    trainer.test(model=best_model, datamodule=datamodule)

    print("Best model path:", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classifier Model")
    parser.add_argument("--model_name", type=str, default=None, help="The name of the model")
    parser.add_argument("--model_path", type=str, default=None, help="The path to the model")
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset")
    parser.add_argument("--experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
                        
    args = parser.parse_args()

    if args.dataset_name == "verafiles":
        datamodule = VeraFilesNewsDataModule("datasets/verafiles_dataset", num_worker=1, model=args.model_name)
        num_labels = 3
    elif args.dataset_name == "xlm_fakenews":
        datamodule = FakeNewsDataModule("datasets/xlm_fakenews", num_worker=1, model=args.model_name)
        num_labels = 7
    elif args.dataset_name == "liar":
        datamodule = LiarDataModule("datasets/liar_dataset", num_worker=1, model_name=args.model_name)
        num_labels = 6
    elif args.dataset_name == "fakenewsTL":
        datamodule = FakeTLDataModule("datasets/fakenewsTL_dataset", num_worker=1, model=args.model_name)
        num_labels = 2


    if args.model_path is not None:
        supconmodel = SupConModel.load_from_checkpoint(args.model_path)
        model = ClassifierModel(model=supconmodel, num_labels=num_labels, learning_rate=args.learning_rate)
    elif args.model_name is not None:
        model = ClassifierModel(model_name=args.model_name, num_labels=num_labels, learning_rate=args.learning_rate)
    else:
        raise ValueError("Either model_path or model_name must be provided")


    train_classifier_model(model, datamodule, args.experiment_name, args.max_epochs)
