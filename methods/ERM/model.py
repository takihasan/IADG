#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module definitions.
"""

import copy
import math
from collections import deque

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet

from modules.classifiers import LinearClassifier
from modules.featurizers import FeatureExtractor


class TrainingModel(nn.Module):
    """
    Args:
        base_model: str
            resnet18 or resnet50
        dropout: float
            dropout rate
        n_classes: int
            number of output classes
        n_domains: int
            number of domains

    """

    def __init__(
        self,
        base_model,
        dropout,
        n_classes,
        n_domains,
        encoder_n_layers=3,
        encoder_bn=True,
        encoder_dropout=None,
    ):
        super().__init__()
        self.featurizer = FeatureExtractor(base_model, dropout)
        self.dim_out = self.featurizer.dim_out
        self.classifier_label = LinearClassifier(self.dim_out, n_classes)

    def embed_label(self, inputs):
        """
        Args:
            inputs: torch.Tensor
                [batch, channel, height, width]
        """
        emb = self.featurizer(inputs)  # featurize

        return emb

    def embed(self, inputs):
        """
        Args:
            inputs: torch.Tensor
                [batch, channel, height, width]
        """
        emb = self.featurizer(inputs)  # featurize
        out_dict = {"label": emb}

        return out_dict

    def classify_label(self, emb):
        """
        Args:
            emb: torch.Tensor
                [batch, dim_out]
        """
        logits = self.classifier_label(emb)
        return logits

    def classify(self, emb):
        """
        Args:
            emb: torch.Tensor
                [batch, dim_out]

        Returns:
            dict:
                dictionary with keys label, domain, task
        """
        logits_label = self.classifier_label(emb)

        out_dict = {"label": logits_label}
        return out_dict

    def forward(self, inputs):
        """
        Args:
            inputs: torch.Tensor
                [batch, channel, height, width]
        """
        embeddings = self.embed(inputs)

        logits = {}
        for type_, emb in embeddings.items():
            logits[type_] = self.classify(emb)

        return embeddings, logits

    def get_params_label(self):
        param_list = [
            {"params": self.featurizer.parameters()},
            {"params": self.classifier_label.parameters()},
        ]
        return param_list
