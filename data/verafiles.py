from torch.utils.data import Dataset
import torch
import pandas as pd
# from textattack.augmentation import EasyDataAugmenter

import lightning as L
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

# Note - you must have torchvision installed for this example
from transformers import DataCollatorWithPadding


class VeraFilesDataset(Dataset):
    def __init__(self, path_to_dataset, tokenizer):

        self.df = pd.read_csv(path_to_dataset)
        self.tokenizer = tokenizer

        self.labels = self.df["RATING"].unique()
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def __getitem__(self, idx):

        entry = self.df.iloc[idx]

        text = entry["CONCAT QUOTES"]
        label = entry["RATING"]

        return text, self.label2id[label]

    def __len__(self):
        return len(self.df)


class VeraFilesNewsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=10, num_worker=1, model="xlm-roberta-base"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def collate_fn(self, batch):

        tokenized = self.tokenizer([x for x,y in batch], padding=True, )

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor([y for x, y in batch]),
        }
    
    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.fakenews_train = VeraFilesDataset(self.data_dir + "/" + "train.csv", self.tokenizer)
            self.fakenews_valid = VeraFilesDataset(self.data_dir + "/" + "val.csv", self.tokenizer)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.fakenews_test = VeraFilesDataset(self.data_dir + "/" + "test.csv", self.tokenizer)

        if stage == "predict":
            self.fakenews_test = VeraFilesDataset(self.data_dir + "/" + "test.csv", self.tokenizer)


    def train_dataloader(self):
        return DataLoader(self.fakenews_train, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, shuffle=True, multiprocessing_context='fork')

    def val_dataloader(self):
        return DataLoader(self.fakenews_valid, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, multiprocessing_context='fork')

    def test_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, multiprocessing_context='fork')

    def predict_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, multiprocessing_context="fork")
    