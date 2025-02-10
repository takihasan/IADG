#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torch import nn
from torchvision.models import resnet


def get_resnet(name):
    """
    Get pretrained resnet model by name.

    Args:
        name: str
            resnet18 or resnet50
    """
    if name == "resnet18":
        model = resnet.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V2)

    # freeze batch norm
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    return model


class FeatureExtractor(nn.Module):
    """
    Params:
        net: nn.Module
            feature extractor network
        dropout: nn.Module
            dropout layer
        dim_out: int
            size of output

    Args:
        base_model: str
            resnet18 or resnet50
        dropout: float
            dropout rate
    """

    def __init__(self, base_model, dropout=0.5):
        super().__init__()
        self.net = get_resnet(base_model)
        self.dropout = nn.Dropout(dropout)

        self.dim_out = self.net.fc.in_features

        del self.net.fc
        self.net.fc = nn.Identity()
        self._freeze_bn()

    def forward(self, inputs):
        """
        Args:
            inputs: torch.Tensor
                [batch, channel, height, width]
        """
        emb = self.net(inputs)
        emb = self.dropout(emb)
        return emb

    def train(self, mode=True):
        super().train(mode)
        self._freeze_bn()

    def _freeze_bn(self):
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


class MLP(nn.Module):
    """
    MLP Encoder.
    Consists of <n_hidden> linear layers with <dim_in> as both input and output dimension and
    1 linear layer with <dim_in> as input and <dim_out> as output.
    If <use_bn> is set to True, BatchNorm layers are added.
    If <dropout> is not None, Dropout layer is added.

    Args:
        dim_in: int
            input tensor size
        n_layers: int
            number of layers
        use_bn: bool
            use BatchNorm between layers, defaults to True
        dropout: None or float
            if not None, add dropout with <dropout> ratio
    """

    def __init__(self, dim_in, dim_out, n_hidden=3, use_bn=True, dropout=0.0):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout

        # init net
        net = []
        # hidden layers
        for _ in range(n_hidden):
            net.append(nn.Linear(self.dim_in, self.dim_in))
            net.append(nn.ReLU())
            if use_bn:
                net.append(nn.BatchNorm1d(self.dim_in))
            net.append(nn.Dropout(self.dropout))

        # last layer
        net.append(nn.Linear(self.dim_in, self.dim_out))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
