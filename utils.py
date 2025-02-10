#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    return device


def get_optimizer_name(optimizer):
    optimizer_name = str(optimizer).split(".")[-1].split("'>")[0]
    return optimizer_name


class Logger:
    def __init__(self, dir_log, name="log"):
        self.dir_log = dir_log
        self.log_dict = {}
        self.name = name

    def save(self, name=None):
        if name is None:
            name = self.name
        path = os.path.join(self.dir_log, f"{name}.csv")
        pd.DataFrame(self.log_dict).to_csv(path, index=False)

    def log(self, key, val):
        if key not in self.log_dict:
            self.log_dict[key] = []
        self.log_dict[key].append(val)

    def load(self, csv_file):
        dir_log, filename = os.path.split(csv_file)
        name, _ = os.path.splitext(filename)
        self.dir_log = dir_log
        self.name = name

        df = pd.read_csv(csv_file)
        self.log_dict = df.to_dict(orient="list")

    def set_name(self, name):
        self.name = name


class Loaders:
    def __init__(self, loaders):
        """

        """
        # loaders
        self.loaders = loaders
        self.loader_train = self.loaders["train"]
        self.loader_dev = self.loaders["dev"]
        self.loader_test = self.loaders["test"]
        self.batch_size = self.loader_train.batch_size

    def get_loaders(self):
        return self.loaders

    def get_loader_train(self):
        return self.loaders["train"]

    def get_loader_dev(self):
        return self.loaders["dev"]

    def get_loader_test(self):
        return self.loaders["test"]

    def load(self, labels_train, labels_dev):
        self.loader_train = load_loader(labels_train, self.batch_size)
        self.loader_dev = load_loader(labels_dev, self.batch_size)
