import torch
import lightning as L
from transformers import  AutoModelForSequenceClassification, AdamW, AutoModel
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from torchmetrics.classification import MulticlassConfusionMatrix

import torchmetrics

import PIL

from models.loss import SupConLoss

import torch.nn.functional as F


class SupConModel(L.LightningModule):
    def __init__(self, model_name='xlm-roberta-base', embedding_size=1024, learning_rate=2e-5, loss=SupConLoss()):
        super().__init__()

        self.save_hyperparameters("model_name", "embedding_size", "learning_rate")

        self.model_name = model_name

        # Load the pretrained transformer model
        self.model = AutoModel.from_pretrained(
          model_name
        )

        self.projection_layer = torch.nn.Linear( self.model.config.hidden_size, embedding_size)

        # Set up the loss criterion (CrossEntropyLoss is used for multi-class classification)
        self.loss = loss

        # Set up hyperparameters
        self.learning_rate = learning_rate

        self.test_projections = []
        self.test_labels = []


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = output[0]
        pooled_output = last_hidden_state[:, 0]

        projection = self.projection_layer(pooled_output)

        return F.normalize(projection, dim=1)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        projection = self(input_ids, attention_mask)
        loss = self.loss(projection, labels)

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        projection = self(input_ids, attention_mask)
        loss = self.loss(projection, labels)

        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        projection = self(input_ids, attention_mask)
        loss = self.loss(projection, labels)

        self.test_projections.append(projection)
        self.test_labels.extend(labels.cpu().tolist())

        self.log('test_loss', loss, prog_bar=True)

    def on_test_epoch_end(self):

        self.test_projections = torch.cat(self.test_projections, dim=0)

        self.logger.experiment.add_embedding(
            self.test_projections,
            metadata=self.test_labels,
            global_step=0
        )
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.trainer.estimated_stepping_batches//10, num_training_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]