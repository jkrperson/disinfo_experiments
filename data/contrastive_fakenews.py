from torch.utils.data import Dataset
import torch
import pandas as pd
# from textattack.augmentation import EasyDataAugmenter

import lightning as L
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

# Note - you must have torchvision installed for this example
from transformers import DataCollatorWithPadding

from data.xfact import FakeNewsDataset

from textattack.augmentation.recipes import SwapAugmenter, DeletionAugmenter
import random  

import random

class ContrastiveFakeNewsDataset(Dataset):
    def __init__(self, path_to_dataset, tokenizer):
        self.train_df = pd.read_csv(path_to_dataset, sep="\t")

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
        return len(self.train_df)

    def __getitem__(self, idx):

        entry = self.train_df.iloc[idx]

        text = entry["claim"]
        label = entry["label"]

        augmentations = [entry["augmented_en"], 
                         entry["augmented_tl"], 
                         entry["augmented_vi"], 
                         entry["augmented_th"], 
                         entry["augmented_zh"]]
        
        augmentation = random.choice(augmentations)

        return [(text, self.label2id[label]), (augmentation, self.label2id[label])]


class ContrastiveFakeNewsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=5, num_worker=14):
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

        batch = sum(batch, [])
        tokenized = self.tokenizer([x for x,y in batch], padding=True, max_length=512, truncation=True)

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
            self.fakenews_train = ContrastiveFakeNewsDataset(self.data_dir + "/" + "train_augmented.tsv", self.tokenizer)
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
