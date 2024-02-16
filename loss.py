import torch
from torch import nn
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder

import torch.nn.functional as F
import torch.distributed.nn

import torch.distributed as dist

def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

class SupConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, features, labels):
        feats = features  # feats shape: [B, D]
        labels = labels    # labels shape: [B]

        device = features.device

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)

        all_feats = feats
        all_labels = labels

        
        mask = torch.eq(labels.view(-1, 1),
                        all_labels.contiguous().view(1, -1)).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss