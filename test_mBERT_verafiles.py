import lightning as L

from data import VeraFilesNewsDataModule
from model import NLPModel
from lightning.pytorch.loggers import TensorBoardLogger

def test_nlp_model_from_checkpoint(checkpoint_path):
    fakenews_datamodule = VeraFilesNewsDataModule("verafiles_dataset", num_worker=1, model="google-bert/bert-base-multilingual-cased")
    nlp_model = NLPModel.load_from_checkpoint(checkpoint_path, num_labels=3, model_name="google-bert/bert-base-multilingual-cased")

    logger = TensorBoardLogger("fakenews_detection", name="Verafiles_mBERT Test")

    trainer = L.Trainer(logger=logger, devices=1)
    trainer.test(model=nlp_model, datamodule=fakenews_datamodule)


if __name__ == "__main__":
    test_nlp_model_from_checkpoint("fakenews_detection/Verafiles_mBERT/version_0/checkpoints/best-checkpoint.ckpt")
