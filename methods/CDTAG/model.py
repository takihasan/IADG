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
from modules.classifiers import CosineClassifier
from modules.featurizers import MLP, FeatureExtractor
from torch import nn


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
        n_classes,
        n_domains,
        base_model="resnet50",
        dropout=0.0,
    ):
        super().__init__()
        # featurizer
        self.featurizer = FeatureExtractor(base_model, dropout)
        self.dim_out = self.featurizer.dim_out
        # encoder
        self.encoder_label = MLP(
            dim_in=self.dim_out,
            dim_out=self.dim_out,
            n_hidden=2,
            use_bn=True,
            dropout=0.0,
        )
        self.encoder_domain = MLP(
            dim_in=self.dim_out,
            dim_out=self.dim_out,
            n_hidden=2,
            use_bn=True,
            dropout=0.0,
        )
        # classifier
        self.classifier_label = CosineClassifier(self.dim_out, n_classes)
        self.classifier_domain = CosineClassifier(self.dim_out, n_domains)
        # projector
        self.projector_label = MLP(
            dim_in=self.dim_out,
            dim_out=self.dim_out,
            n_hidden=2,
            use_bn=True,
            dropout=0.0,
        )
        self.projector_domain = MLP(
            dim_in=self.dim_out,
            dim_out=self.dim_out,
            n_hidden=2,
            use_bn=True,
            dropout=0.0,
        )


    def forward(self, inputs):
        """
        Args:
            inputs: torch.Tensor
                [batch, channel, height, width]
        """
        # only label part
        features = self.featurizer(inputs)
        emb_label = self.encoder_label(features)
        logits_label= self.classifier_label(emb_label)
        return logits_label

    def get_params(self):
        param_list = [
            {"params": self.featurizer.parameters()},
            {"params": self.encoder_label.parameters()},
            {"params": self.encoder_domain.parameters()},
            {"params": self.classifier_label.parameters()},
            {"params": self.classifier_domain.parameters()},
            {"params": self.projector_label.parameters()},
            {"params": self.projector_domain.parameters()},
        ]
        return param_list

    def featurize(self, x):
        """
        x: image tensor
        """
        emb = self.featurizer(x)
        return emb

    def encode_label(self, x):
        """
        x: feature tensor
        """
        emb = self.encoder_label(x)
        return emb

    def encode_domain(self, x):
        """
        x: feature tensor
        """
        emb = self.encoder_domain(x)
        return emb

    def classify_label(self, x):
        """
        f: encoded tensr
        """
        logit = self.classifier_label(x)
        return logit

    def classify_domain(self, x):
        """
        f: encoded tensor
        """
        logit = self.classifier_domain(x)
        return logit
