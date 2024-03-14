import argparse
import lightning as L

from data.contrastive_fakenews import ContrastiveFakeNewsDataModule
from data.liarliar import LiarContrastiveDataModule
from data.verafiles import VeraFilesNewsDataModule

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from models.sup_contrastive_model import SupConModel
from models.loss import SupConLoss

import random
import numpy as np
import torch
    

def train_sup_con_model(
        model: L.LightningModule,
        datamodule,
        experiment_name, 
        max_epochs, 
    ):

    L.seed_everything(42)

    logger = TensorBoardLogger("fakenews_detection", name=experiment_name)

    # log model only if `val_accuracy` increases
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", filename='best-checkpoint',  # Name of the checkpoint files
        save_top_k=2,  # Only keep the top 1 model
        verbose=True, # Print when a new checkpoint is saved
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(logger=logger, max_epochs=max_epochs, log_every_n_steps=10, devices=1, callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model=model, datamodule=datamodule)

    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SupConModel")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for the loss function")
    parser.add_argument("--embedding_size", type=int, default=2048, help="Size of the projection layer")
                        
    args = parser.parse_args()

    model = SupConModel(
        model_name=args.model_name, 
        learning_rate=args.learning_rate,
        loss=SupConLoss(temperature=args.temperature),
        embedding_size=2048
    )

    if args.dataset_name == "verafiles":
        datamodule = VeraFilesNewsDataModule("datasets/verafiles_dataset", num_worker=1, model=args.model_name)
    elif args.dataset_name == "xlm_fakenews":
        datamodule = ContrastiveFakeNewsDataModule("datasets/xlm_fakenews", num_worker=1, model=args.model_name)
    elif args.dataset_name == "liar":
        datamodule = LiarContrastiveDataModule("datasets/liar_dataset", num_worker=1, model_name=args.model_name)


    train_sup_con_model(
        model,
        datamodule,
        args.experiment_name,
        args.max_epochs, 
    )
