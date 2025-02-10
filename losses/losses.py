#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loss functions.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def entropy(x):
    ent = -torch.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    ent = ent.sum(-1)
    return ent


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality loss.
    """

    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.loss = nn.CosineSimilarity(dim=self.dim, eps=self.eps)

    def forward(self, inputs_1, inputs_2, reduction="mean"):
        """
        Args:
            inputs_1: torch.Tensor
                [batch, emb]
            inputs_1: torch.Tensor
                [batch, emb]
            reduction: str
                if "mean" - take mean of losses
                if "sum" - sum the losses
                otherwise return all losses
        """

        loss = self.loss(inputs_1, inputs_2)
        loss = torch.abs(loss)
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return loss

    def __str__(self):
        return f"OrthogonalityLoss(dim={self.dim}, eps={self.eps})"


class MaxEntropyLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, reduction="mean"):
        batch_size, n_classes = logits.shape
        loss = -(F.log_softmax(logits, dim=1) + math.log(n_classes)) * (1 / n_classes)
        loss = loss.sum(-1)
        if reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class MaxEntropyFixedLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, reduction="mean"):
        batch_size, n_classes = logits.shape
        loss = -F.log_softmax(logits, dim=1) * F.softmax(logits, dim=1)
        loss = loss.sum(-1)
        if reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class ContrastiveLossGeneral(nn.Module):
    def __init__(self, eps=1e-6):
        self.eps = eps
        super().__init__()

    def forward(self, x, pos_set, neg_set, pos_index, neg_index):
        """
        x: B, E
        pos_set: Npos, E
        neg_set: Nneg, E
        """
        x = F.normalize(x, p=2)
        pos_set = F.normalize(x, p=2)
        neg_set = F.normalize(neg_set, p=2)
        # positive
        prod = x.unsqueeze(1) * pos_set.unsqueeze(0)  # B, Npos, E
        prod = prod.sum(-1)  # B, Npos
        prod = prod * pos_index
        pos = -1 / (pos_index.sum(-1)) * prod
        pos = pos.sum(-1)  # B
        # negative
        prod = x.unsqueeze(1) * neg_set.unsqueeze(0)  # B, Nneg, E
        prod = prod.sum(-1)  # B, Nneg
        prod = torch.exp(prod)  # B, Nneg
        prod = prod * neg_index  #  B, Nneg
        neg = torch.log(prod[neg_index])
        neg = neg.sum(-1)  # B

        loss = pos + neg
        loss = loss.mean()

        return loss


class EntropyRegularizationLoss(nn.Module):
    def __init__(self, eps=1e-6):
        self.eps = eps
        super().__init__()

    def forward(self, prototypes):
        C, E = prototypes.shape
        prototypes = F.normalize(prototypes, p=2, dim=1)
        proto_logits = prototypes.unsqueeze(0) * prototypes.unsqueeze(1)  # C, C, E
        proto_logits = proto_logits.sum(-1)  # C, C
        proto_logits_sum = proto_logits.sum(0, keepdim=True)
        loss = entropy(proto_logits).sum(-1) / C + entropy(proto_logits_sum)
        loss = loss.sum()
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x_label, x_domain, y_label, y_domain):
        """
        x_label: B, E
        x_domain: B, E

        """
        B, E = x_label.shape
        x_label = F.normalize(x_label, p=2, dim=1)
        x_domain = F.normalize(x_domain, p=2, dim=1)
        # label label product
        dot_prod_ll = x_label.unsqueeze(0) * x_domain.unsqueeze(1)
        dot_prod_ll = dot_prod_ll.sum(-1)
        # label domain product
        dot_prod_ld = x_label.unsqueeze(1) * x_domain.unsqueeze(0)
        dot_prod_ld = dot_prod_ld.sum(-1)
        dot_prod_ld = torch.exp(dot_prod_ld)
        eye_mask = 1 - torch.eye(B).to(x_label.device)  # set diagional to 0
        # set building
        same_label = y_label.unsqueeze(0) == y_label.unsqueeze(1)  # B, B
        same_domain = y_domain.unsqueeze(0) == y_domain.unsqueeze(1)  # B, B
        # positive set
        positive_set = same_label * same_domain  # same domain and same label
        positive_set = positive_set * eye_mask  # remove self similarity
        n_pos = positive_set.sum(-1)  # number of positives for one example
        positive_sum = positive_set * dot_prod_ll  # B, B
        # negative set
        negative_set = ~same_label
        negative_sum = torch.cat(
            [negative_set * dot_prod_ll, dot_prod_ld * eye_mask], -1
        )
        negative_sum = negative_sum.sum(-1)  # B
        # loss
        take_index = n_pos > 0  # take only those with at least 1 positive
        n_pos = n_pos[take_index]
        if n_pos.sum(-1) == 0:
            return torch.FloatTensor([0.0]).sum(-1)
        positive_sum = positive_sum[take_index].sum(-1)
        negative_sum = torch.log(negative_sum[take_index].sum(-1))

        loss = -1 / n_pos * positive_sum + negative_sum

        # reduce
        loss = loss.mean(-1)

        return loss
