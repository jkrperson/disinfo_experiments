import torch
import pandas as pd

class VerafilesDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        
        self.root = root
        self.transform = transform
        self.df = pd.read_csv(root)

        self.idx2label = {k:v for k,v in enumerate(self.df["label"].unique())}
        self.label2idx = {v:k for k,v in self.idx2label.items()}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row["text"]
        label = self.label2idx[row["label"]]
        return {
            "text": text,
            "label": label
        }