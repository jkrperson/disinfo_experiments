import argparse
import lightning as L
from data import ContrastiveFakeNewsDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from model import SupConModel
from loss import SupConLoss

import random
import numpy as np
import torch
    

def train_sup_con_model(max_epochs, log_every_n_steps, num_workers, gpus, learning_rate, seed):
    L.seed_everything(seed)

    logger = TensorBoardLogger("fakenews_detection", name="RoBERTa Normal")

    # log model only if `val_accuracy` increases
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy", mode="max", filename='best-checkpoint',  # Name of the checkpoint files
        save_top_k=1,  # Only keep the top 1 model
        verbose=True  # Print when a new checkpoint is saved
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    loss = SupConLoss(temperature=0.5)

    trainer = L.Trainer(logger=logger, max_epochs=max_epochs, log_every_n_steps=log_every_n_steps, devices=gpus, callbacks=[lr_monitor])

    fakenews_datamodule = ContrastiveFakeNewsDataModule("xlm_fakenews", num_worker=num_workers)

    xlm_roberta = SupConModel(loss=loss, embedding_size=1024, learning_rate=learning_rate)

    trainer.fit(model=xlm_roberta, datamodule=fakenews_datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SupConModel")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
                        
    args = parser.parse_args()

    train_sup_con_model(args.max_epochs, args.log_every_n_steps, args.num_workers, args.gpus, args.learning_rate, args.seed)
