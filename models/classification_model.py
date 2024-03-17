import torch
import lightning as L
from transformers import  AutoModelForSequenceClassification

from torchmetrics.classification import MulticlassConfusionMatrix

import torchmetrics

import PIL

from torch import nn
from transformers import get_cosine_schedule_with_warmup

import numpy as np

class ContrastivePretrainedModel(L.LightningModule):
    def __init__(self, model:L.LightningModule, num_labels, dropout_rate=0.3):
        super().__init__()

        # Load the pretrained transformer model
        model.freeze()
        
        self.encoder = model
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.LazyLinear(num_labels)

    def forward(self, input_ids, attention_mask):
        projection = self.encoder(input_ids, attention_mask)
        dropped = self.dropout(projection)
        preds = self.fc(dropped)
        return preds
    

class ClassifierModel(L.LightningModule):
    def __init__(self, model:L.LightningModule=None, model_name:str=None, num_labels=7, learning_rate=0.05):
        super().__init__()

        self.save_hyperparameters("model", "model_name", "num_labels", "learning_rate")

        # Load the pretrained transformer model
        if model_name is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.25
            )
        elif model is not None:
            # Model should not have a FC layer at the end
            self.model = ContrastivePretrainedModel(model, num_labels)

        else: 
            raise ValueError("Either model or model_name should be provided")

        # Set up the loss criterion (CrossEntropyLoss is used for multi-class classification)
        self.loss = torch.nn.CrossEntropyLoss()

        # Set up hyperparameters
        self.learning_rate = learning_rate

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_labels)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_labels)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_labels)
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_labels)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=num_labels)

        self.test_preds = []
        self.test_labels = []

    def forward(self, input_ids, attention_mask):
        
        if type(self.model) == ContrastivePretrainedModel:
            logits = self.model(input_ids, attention_mask=attention_mask)
        else:
            output = self.model(input_ids, attention_mask=attention_mask)
            logits = output.logits

        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.valid_acc(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        logits = self(input_ids, attention_mask)

        loss = self.loss(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.test_preds.append(preds)
        self.test_labels.append(labels)
    
        # return {"loss": loss, "outputs": preds, "labels": labels}


    def on_test_epoch_end(self):

        preds = torch.cat(self.test_preds)
        labels = torch.cat(self.test_labels)
        
        self.conf_matrix.update(preds, labels)

        fig, ax = self.conf_matrix.plot()

        self.logger.experiment.add_figure("Confusion Matrix", fig, 0)

        self.test_preds = []
        self.test_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.trainer.estimated_stepping_batches//10, num_training_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
