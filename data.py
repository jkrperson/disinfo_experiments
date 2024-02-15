from torch.utils.data import Dataset
import torch
import pandas as pd
# from textattack.augmentation import EasyDataAugmenter

class FakeNewsDataset(Dataset):
    def __init__(self, path_to_dataset, tokenizer):
        train_df = pd.read_csv(path_to_dataset, sep="\t")

        self.texts = train_df["claim"].to_list()
        self.labels = train_df["label"].to_list()
        self.tokenizer = tokenizer

        self.id2label = {0:'true', 
                         1:'false', 
                         2:'partly true/misleading', 
                         3:'mostly false' , 
                         4:'mostly true', 
                         5:'complicated/hard to categorise', 
                         6:'other'}
        
        self.label2id = {'true':0, 'false':1 , 'partly true/misleading':2, 'mostly false':3, 'mostly true':4, 'complicated/hard to categorise':5, 'other':6}
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label2id[self.labels[idx]]

        return text, label
    

import lightning as L
from torch.utils.data import random_split, DataLoader

import pandas

from transformers import AutoTokenizer

# Note - you must have torchvision installed for this example
from transformers import DataCollatorWithPadding

from textattack.augmentation.recipes import SwapAugmenter, DeletionAugmenter
import random  


class FakeNewsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=10, num_worker=14):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def augmentor(self, batch):
        pass

    def collate_fn(self, batch):

        tokenized = self.tokenizer([x for x,y in batch], padding=True)

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor([y for x, y in batch]),
        }
    
    def train_batch_transform(self, batch):
        pass

    def test_batch_transform(self, batch):
        pass

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.fakenews_train = FakeNewsDataset(self.data_dir + "/" + "train.tsv", self.tokenizer)
            self.fakenews_valid = FakeNewsDataset(self.data_dir + "/" + "valid.tsv", self.tokenizer)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.fakenews_test = FakeNewsDataset(self.data_dir + "/" + "zeroshot_test.tsv", self.tokenizer)

        if stage == "predict":
            self.fakenews_test = FakeNewsDataset(self.data_dir + "/" + "zeroshot_test.tsv", self.tokenizer)


    def train_dataloader(self):
        return DataLoader(self.fakenews_train, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, shuffle=True, multiprocessing_context='fork')

    def val_dataloader(self):
        return DataLoader(self.fakenews_valid, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, multiprocessing_context='fork')

    def test_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, multiprocessing_context='fork')

    def predict_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.num_worker, multiprocessing_context="fork")
    

class ContrastiveFakeNewsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=10, num_worker=14):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.swapper = SwapAugmenter()
        self.deleter = DeletionAugmenter()

    def augmentor(self, batch):
        augmented_data = []

        for x, y in batch:
            augmented_text = []
            augmented_text += self.swapper.augment(x)
            augmented_text += self.deleter.augment(x)

            random.shuffle(augmented_text)

            augmented_data.append((augmented_text[0], y))

        return augmented_data

    def train_collate_fn(self, batch):

        augmented_data = self.augmentor(batch)

        batch = batch + augmented_data
        tokenized = self.tokenizer([x for x,y in batch], padding=True)

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor([y for x, y in batch]),
        }
    
    def valid_collate_fn(self, batch):
        tokenized = self.tokenizer([x for x,y in batch], padding=True)

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor([y for x, y in batch]),
        }

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.fakenews_train = FakeNewsDataset(self.data_dir + "/" + "train.tsv", self.tokenizer)
            self.fakenews_valid = FakeNewsDataset(self.data_dir + "/" + "valid.tsv", self.tokenizer)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.fakenews_test = FakeNewsDataset(self.data_dir + "/" + "zeroshot_test.tsv", self.tokenizer)

        if stage == "predict":
            self.fakenews_test = FakeNewsDataset(self.data_dir + "/" + "zeroshot_test.tsv", self.tokenizer)


    def train_dataloader(self):
        return DataLoader(self.fakenews_train, batch_size=self.batch_size, collate_fn=self.train_collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.fakenews_valid, batch_size=self.batch_size, collate_fn=self.valid_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.valid_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.valid_collate_fn)