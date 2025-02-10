#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses import ContrastiveLossGeneral, EntropyRegularizationLoss
from utils import Logger, get_device, get_optimizer_name

class BatchMaker:
    def __init__(self, transforms, n_weak=16, n_strong=8, seed=None):
        self.augment_weak = transforms["weak"]
        self.augment_strong = transforms["strong"]
        self.augment_orig = transforms["orig"]
        self.n_weak = n_weak
        self.n_strong = n_strong

    def __call__(self, image, n_weak=None, n_strong=None):
        if n_weak is None:
            n_weak = self.n_weak
        if n_strong is None:
            n_strong = self.n_strong
        # original
        batch_orig = [self.augment_orig(image) for _ in range(1)]
        batch_orig = torch.stack(batch_orig)
        # weak augmentations
        batch_weak = [self.augment_weak(image) for _ in range(n_weak)]
        batch_weak = torch.stack(batch_weak)
        # strong augmentations
        batch_strong = [self.augment_strong(image) for _ in range(n_strong)]
        batch_strong = torch.stack(batch_strong)
        return {"weak": batch_weak, "strong": batch_strong, "orig": batch_orig}


class AdaptConfig:
    def __init__(
        self,
        optimizer,
        optimizer_hparams,
        transforms,
        n_weak=16,
        n_strong=8,
        device=None,
        seed=None,
        dir_log="logs",
        run_name=None,
        description="",
    ):
        """ """
        # optimizers
        self.optimizer = optimizer
        self.optimizer_hparams = optimizer_hparams

        # batch maker
        self.n_weak = n_weak
        self.n_strong = n_strong
        self.batch_maker = BatchMaker(transforms, n_weak=n_weak, n_strong=n_strong)

        # device
        if device is None:
            device = get_device()
        self.device = device

        # deterministic
        if seed is not None:
            # set seed
            pass
        self.seed = seed

        # logging
        if run_name is None:
            run_name = str(time() * 1000).split(".")
        self.dir_log = os.path.join(dir_log, run_name)
        os.makedirs(self.dir_log, exist_ok=True)
        self.description = description

    def save(self, name="config_adapt.json"):
        optimizer_name = get_optimizer_name(self.optimizer)

        to_save = {
            "optimizer": optimizer_name,
            "optimizer_hparams": self.optimizer_hparams,
            "seed": self.seed,
            "description": self.description,
            "n_weak": self.n_weak,
            "n_strong": self.n_strong,
        }

        savefile = os.path.join(self.dir_log, name)

        with open(savefile, "w", encoding="utf-8") as file:
            file.write(json.dumps(to_save))

    def load(self, config_file, labels_train, labels_dev, device=None):
        if device is None:
            device = get_device()
        self.device = device

        config = json.loads(config_file)

    def get_optimizer(self):
        return self.optimizer

    def get_optimizer_hparams(self):
        return self.optimizer_hparams

    def get_device(self):
        return self.device

    def get_dir_log(self):
        return self.dir_log

    def get_batch_maker(self):
        return self.batch_maker


class Adapter:
    def __init__(self, model, dataset, config, domain, seed=None):
        """ """
        # model
        self.model = model

        # data
        self.dataset = dataset
        self.batch_maker = config.get_batch_maker()

        # device
        self.device = config.get_device()

        # optimizer
        self.model = self.model.to(self.device)
        optimizer = config.get_optimizer()
        optimizer_hparams = config.get_optimizer_hparams()
        self.optimizer = optimizer(model.parameters(), **optimizer_hparams)

        # loss
        self.loss_fn_xent = nn.CrossEntropyLoss()
        self.loss_fn_contr = ContrastiveLossGeneral()
        self.loss_fn_regu = EntropyRegularizationLoss()

        # logging
        self.dir_log = os.path.join(config.get_dir_log(), domain)
        self.logger = Logger(self.dir_log, name="log_adapt")
        self.writer = SummaryWriter(log_dir=self.dir_log, flush_secs=120)

    def run(self, verbose=True):
        if verbose:
            print(f"[{datetime.now()}] Run start")

        if verbose:
            print(f"[{datetime.now()}] Adaptation start")

        time_adapt_start = time()
        self.adapt()
        time_adapt_end = time()

        if verbose:
            time_adapt_total = time_adapt_end - time_adapt_start
            time_adapt_total = time_adapt_total / 60  # minutes
            print(
                f"[{datetime.now()}] Adaptation over in {time_adapt_total:.1f} minutes"
            )

        results = self.postprocess()
        self.save()

        return results

    def adapt(self, shuffle=True):
        """
        Adapt
        """
        self.model.eval()
        # init index
        index = np.arange(len(self.dataset))
        np.random.shuffle(index)  # random data ordering
        self.label_true = np.zeros_like(index, dtype=int)
        self.label_predicted = np.zeros_like(index, dtype=int)
        self.paths = np.empty(len(index), dtype=str)

        for i in tqdm(index):
            data = self.dataset[i]
            image = data["image"]
            self.label_true[i] = data["label"]
            self.paths[i] = data["path"]
            batch = self.batch_maker(image)  # orig, weak, strong

            # move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # featurize
            fs = {k: self.model.featurizer(v) for k, v in batch.items()}
            # encode
            zs = {k: self.model.encoder_label(v) for k, v in fs.items()}

            # pseudo label
            ys_weak = self.model.classifier_label(zs.pop("weak"))
            ys_weak = torch.softmax(ys_weak, dim=1)
            ys_avg = ys_weak.mean(dim=0)
            y_pseudo = torch.argmax(ys_avg)
            pseudo_label = y_pseudo.repeat(ys_weak.shape[0])
            self.label_predicted[i] = y_pseudo.item()
            # classification loss
            loss_xent_weak = self.loss_fn_xent(ys_weak, pseudo_label)

            # project embeddings
            ps = {k: self.model.projector_label(v) for k, v in zs.items()}
            centers = self.model.projector_label(self.model.classifier_label.net)
            # positive set
            pos_set = ps["strong"]
            pos_index = torch.ones(size=(1, pos_set.shape[0]))
            pos_index = pos_index.to(self.device)
            # negative set, index
            neg_set = centers
            neg_index = torch.ones(size=(1, centers.shape[0],)) > 0
            neg_index[0][y_pseudo] = False
            neg_index = neg_index.to(self.device)
            # contrastive loss
            anchor = ps["orig"]
            loss_contr = self.loss_fn_contr(
                anchor, pos_set, neg_set, pos_index, neg_index
            )

            # regularization loss
            loss_regu = self.loss_fn_regu(centers)

            # loss total
            loss = loss_xent_weak * 1.0
            loss = loss + loss_contr * 1.0
            loss = loss + loss_regu * 1.0

            # backprop
            loss.backward()
            self.optimizer.step()

            # logging
            tracking = {
                "loss_xent_weak": loss_xent_weak.item(),
                "loss_contr": loss_contr.item(),
                "loss_regu": loss_regu.item(),
                "loss": loss.item(),
                "index": i,
            }

            for key, value in tracking.items():
                self.logger.log(key, value)

            for key, value in tracking.items():
                self.writer.add_scalar(key + "_adapt", value)

            self.logger.save()

    def postprocess(self):
        metrics_dict = {}
        accuracy = (self.label_true == self.label_predicted).mean()
        metrics_dict["accuracy"] = accuracy

        # save metrics
        savename = os.path.join(self.dir_log, "metrics_adapt.json")
        with open(savename, "w", encoding="utf-8") as file:
            file.write(json.dumps(metrics_dict))

        # save results
        savename = os.path.join(self.dir_log, "predictions_adapt.csv")
        predictions_dict = {
            "image": self.paths,
            "label_true": self.label_true,
            "label_predicted": self.label_predicted,
        }
        pd.DataFrame(predictions_dict).to_csv(savename, index=False)

        return metrics_dict

    def save(self, name=None):
        if name is None:
            name = "weights_adapt"

        savepath = os.path.join(self.dir_log, name + ".pth")

        torch.save(self.model.state_dict(), savepath)
        self.logger.save()
