import torch
import lightning as L
from transformers import  AutoModelForSequenceClassification, AdamW

from torchmetrics.classification import MulticlassConfusionMatrix

import torchmetrics

import PIL


class NLPModel(L.LightningModule):
    def __init__(self, model_name='xlm-roberta-base', num_labels=7, learning_rate=0.05):
        super().__init__()

        # Load the pretrained transformer model
        self.model = AutoModelForSequenceClassification.from_pretrained(
          model_name, num_labels=num_labels, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.25
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
        
        self.logger.add_image("Confusion Matrix", conf_img, 0)

        self.test_preds = []
        self.test_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)