import lightning as L

from data import VeraFilesNewsDataModule
from model import NLPModel
from lightning.pytorch.loggers import TensorBoardLogger

def test_nlp_model_from_checkpoint(checkpoint_path):
    fakenews_datamodule = VeraFilesNewsDataModule("verafiles_dataset", num_worker=1)
    nlp_model = NLPModel.load_from_checkpoint(checkpoint_path, num_labels=3)

    logger = TensorBoardLogger("fakenews_detection", name="Verafiles_RoBERTa Test")

    trainer = L.Trainer(logger=logger, devices=1)
    trainer.test(model=nlp_model, datamodule=fakenews_datamodule)


if __name__ == "__main__":
    test_nlp_model_from_checkpoint("fakenews_detection/Verafiles_RoBERTa/version_3/checkpoints/best-checkpoint.ckpt")
