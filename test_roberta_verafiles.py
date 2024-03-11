import lightning as L

from data import VeraFilesNewsDataModule
from model import NLPModel


def test_nlp_model_from_checkpoint(checkpoint_path):
    fakenews_datamodule = VeraFilesNewsDataModule("verafiles_dataset", num_worker=1)
    nlp_model = NLPModel.load_from_checkpoint(checkpoint_path)

    trainer = L.Trainer(gpus=1)
    trainer.test(model=nlp_model, datamodule=fakenews_datamodule)


if __name__ == "__main__":
    test_nlp_model_from_checkpoint()