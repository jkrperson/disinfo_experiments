import torch
from torch import nn
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)  # Move mask to correct device
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float() - mask.float()
        mask_neg = torch.ne(labels, labels.T).float()

        sim_matrix = torch.matmul(features, features.T)
        sim_matrix = sim_matrix / self.temperature

        exp_sim_matrix = torch.exp(sim_matrix)
        pos_sum = torch.sum(exp_sim_matrix * mask_pos, dim=-1)
        neg_sum = torch.sum(exp_sim_matrix * mask_neg, dim=-1)

        loss = -torch.log(pos_sum / neg_sum)
        return loss.mean()