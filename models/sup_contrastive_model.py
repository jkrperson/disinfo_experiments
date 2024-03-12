import torch
import lightning as L
from transformers import  AutoModelForSequenceClassification, AdamW, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from torchmetrics.classification import MulticlassConfusionMatrix

import torchmetrics

import PIL

from models.loss import SupConLoss


class SupConModel(L.LightningModule):
    def __init__(self, model_name='xlm-roberta-base', embedding_size=128, learning_rate=2e-5, loss=SupConLoss()):
        super().__init__()

        # Load the pretrained transformer model
        self.model = AutoModel.from_pretrained(
          model_name
        )

        self.projection_layer = torch.nn.Linear(self.model.config.hidden_size, embedding_size)

        # Set up the loss criterion (CrossEntropyLoss is used for multi-class classification)
        self.loss = loss

        # Set up hyperparameters
        self.learning_rate = learning_rate


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        pooled_output = last_hidden_state[:, 0]
        loss = self.loss(pooled_output, labels)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss = self(input_ids, attention_mask, labels)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss = self(input_ids, attention_mask, labels)

        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.trainer.estimated_stepping_batches//10, num_training_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]