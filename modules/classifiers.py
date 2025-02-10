#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
from torch.nn import functional as F


class LinearClassifier(nn.Module):
    """
    Classification head.

    Args:
        dim_in: int
            input size
        dim_out: int
            output size, number of classes
    """

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.net = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, embeddings):
        """
        Args:
            embeddings: torch.Tensor
                [batch, dim_in]
        """
        logits = self.net(embeddings)
        return logits


class CosineClassifier(nn.Module):
    """
    Cosine classification head.

    Args:
        dim_in: int
            input size
        dim_out: int
            output size, number of classes
    """

    def __init__(self, dim_in, dim_out, a=math.sqrt(5)):
        super().__init__()
        self.net = nn.Parameter(torch.FloatTensor(dim_out, dim_in))
        nn.init.kaiming_uniform_(self.net, mode="fan_out", a=a)

    def forward(self, embeddings):
        """
        Args:
            embeddings: torch.Tensor
                [batch, dim_in]

        Returns:
            torch.Tensor
                logits
        """
        # normalize
        embeddings = F.normalize(embeddings, p=2)
        prototypes = F.normalize(self.net, p=2)

        # logits = F.linear(embeddings, self.net)
        logits = embeddings @ prototypes.T
        # logits = self.net(embeddings)

        return logits
