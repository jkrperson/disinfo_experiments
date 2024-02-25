import torch
import lightning as L
from transformers import  AutoModelForSequenceClassification, AdamW, AutoModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from torchmetrics.classification import MulticlassConfusionMatrix

import torchmetrics

import PIL

from loss import SupConLoss


class NLPModel(L.LightningModule):
    def __init__(self, model_name='xlm-roberta-base', num_labels=7, learning_rate=0.05):
        super().__init__()

        # Load the pretrained transformer model
        self.model = AutoModelForSequenceClassification.from_pretrained(
          model_name, num_labels=num_labels
        )

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

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss, outputs = self(input_ids, attention_mask, labels)
        preds = torch.argmax(outputs, dim=1)

        self.train_acc(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss, outputs = self(input_ids, attention_mask, labels)
        preds = torch.argmax(outputs, dim=1)

        self.valid_acc(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss, outputs = self(input_ids, attention_mask, labels)
        preds = torch.argmax(outputs, dim=1)

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

        canvas = fig.canvas

        canvas.draw()
        
        conf_img = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        
#         self.logger.experiment.log({"image": [wandb.Image(conf_img)]})

        self.test_preds = []
        self.test_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

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


#     def on_test_epoch_end(self):

#         preds = torch.cat(self.test_preds)
#         labels = torch.cat(self.test_labels)
        
#         self.conf_matrix.update(preds, labels)

#         fig, ax = self.conf_matrix.plot()

#         canvas = fig.canvas

#         canvas.draw()
        
#         conf_img = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        
# #         self.logger.experiment.log({"image": [wandb.Image(conf_img)]})

#         self.test_preds = []
#         self.test_labels = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler_warmup = OneCycleLR(optimizer, max_lr=self.learning_rate, epochs=10, steps_per_epoch=1811*2)


        return [optimizer], [scheduler_warmup]