from torch.utils.data import Dataset
import torch
import pandas as pd
# from textattack.augmentation import EasyDataAugmenter

import lightning as L
from torch.utils.data import random_split, DataLoader

import pandas

from transformers import AutoTokenizer

# Note - you must have torchvision installed for this example
from transformers import DataCollatorWithPadding

from textattack.augmentation.recipes import SwapAugmenter, DeletionAugmenter
import random  



class LiarDatasetContrastive(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Args:
        path_to_dataset (str): Path to the dataset .csv file.
        tokenizer (Tokenizer): Tokenizer to be used to tokenize the text.

    Attributes:
        texts (List[str]): List of claim texts from the dataset.
        labels (List[str]): List of labels corresponding to the texts.
        tokenizer (Tokenizer): Tokenizer to be used to tokenize the text.
        label2id (Dict[str, int]): A dictionary to map label names to integers.
        id2label (Dict[int, str]): A dictionary to map integers to label names.
    """

    def __init__(self, path_to_dataset, tokenizer):
        self.train_df = pd.read_csv(path_to_dataset, delimiter="\t")

        self.tokenizer = tokenizer

        self.labels = self.train_df["label"].unique()
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def __len__(self):
        """
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.train_df)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            Tuple[str, int]: The text and its corresponding label as an integer.
        """

        entry = self.train_df.iloc[idx]

        text = entry["text"]
        label = entry["label"]

        augmentations = [
                        entry["augmented_tl"], 
                        entry["augmented_vi"], 
                        entry["augmented_th"], 
                        entry["augmented_zh"]]
        
        augmentation = random.choice(augmentations)

        return [(text, self.label2id[label]), (augmentation, self.label2id[label])]
            

class LiarDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Args:
        path_to_dataset (str): Path to the dataset .csv file.
        tokenizer (Tokenizer): Tokenizer to be used to tokenize the text.

    Attributes:
        texts (List[str]): List of claim texts from the dataset.
        labels (List[str]): List of labels corresponding to the texts.
        tokenizer (Tokenizer): Tokenizer to be used to tokenize the text.
        label2id (Dict[str, int]): A dictionary to map label names to integers.
        id2label (Dict[int, str]): A dictionary to map integers to label names.
    """

    def __init__(self, path_to_dataset, tokenizer):
        self.train_df = pd.read_csv(path_to_dataset, delimiter="\t", header=None)

        self.train_df = self.train_df[[1, 2]]
        self.train_df.columns = ["label", "text"]

        self.tokenizer = tokenizer

        self.labels = self.train_df["label"].unique()
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def __len__(self):
        """
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.train_df)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            Tuple[str, int]: The text and its corresponding label as an integer.
        """

        entry = self.train_df.iloc[idx]

        text = entry["text"]
        label = entry["label"]

        return text, self.label2id[label]
    
    
class LiarContrastiveDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=5, num_worker=1, model_name="xlm-roberta-base"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)


    def train_collate_fn(self, batch):

        batch = sum(batch, [])
        tokenized = self.tokenizer([x for x,y in batch], padding=True, max_length=512, truncation=True)

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor([y for x, y in batch]),
        }
    
    def valid_collate_fn(self, batch):
        tokenized = self.tokenizer([x for x,y in batch], padding=True, max_length=512, truncation=True)

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor([y for x, y in batch]),
        }

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.fakenews_train = LiarDatasetContrastive(self.data_dir + "/" + "train_augmented.tsv", self.tokenizer)
            self.fakenews_valid = LiarDataset(self.data_dir + "/" + "valid.tsv", self.tokenizer)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.fakenews_test = LiarDataset(self.data_dir + "/" + "test.tsv", self.tokenizer)

        if stage == "predict":
            self.fakenews_test = LiarDataset(self.data_dir + "/" + "test.tsv", self.tokenizer)


    def train_dataloader(self):
        return DataLoader(self.fakenews_train, batch_size=self.batch_size, collate_fn=self.train_collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.fakenews_valid, batch_size=self.batch_size, collate_fn=self.valid_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.valid_collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.valid_collate_fn)


class LiarDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size=8, num_worker=1, model_name="xlm-roberta-base"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)


    def collate_fn(self, batch):
        tokenized = self.tokenizer([x for x,y in batch], padding=True, max_length=512, truncation=True)

        return {
            "input_ids": torch.tensor(tokenized["input_ids"]),
            "attention_mask": torch.tensor(tokenized["attention_mask"]),
            "labels": torch.tensor([y for x, y in batch]),
        }

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.fakenews_train = LiarDataset(self.data_dir + "/" + "train.tsv", self.tokenizer)
            self.fakenews_valid = LiarDataset(self.data_dir + "/" + "valid.tsv", self.tokenizer)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.fakenews_test = LiarDataset(self.data_dir + "/" + "test.tsv", self.tokenizer)

        if stage == "predict":
            self.fakenews_test = LiarDataset(self.data_dir + "/" + "test.tsv", self.tokenizer)


    def train_dataloader(self):
        return DataLoader(self.fakenews_train, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.fakenews_valid, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.fakenews_test, batch_size=self.batch_size, collate_fn=self.collate_fn)
