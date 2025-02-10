#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime
from time import time

import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import Logger, get_device, get_optimizer_name


class TrainingConfig:
    def __init__(
        self,
        optimizer,
        optimizer_hparams,
        batch_size,
        device=None,
        loss_fn=None,
        seed=None,
        dir_log="logs",
        run_name=None,
        description="",
        eval_freq=20,
        n_iters=5000,
    ):
        """ """
        # training
        self.n_iters = n_iters
        self.batch_size = batch_size

        # optimizers
        self.optimizer = optimizer
        self.optimizer_hparams = optimizer_hparams

        # device
        if device is None:
            device = get_device()
        self.device = device

        # losses
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = loss_fn

        # deterministic
        self.seed = seed

        # logging
        if run_name is None:
            run_name = str(time() * 1000).split(".")
        self.dir_log = os.path.join(dir_log, run_name)
        os.makedirs(self.dir_log, exist_ok=True)
        self.description = description
        self.eval_freq = eval_freq

    def save(self, name="config.json"):
        optimizer_name = get_optimizer_name(self.optimizer)

        to_save = {
            "optimizer": optimizer_name,
            "optimizer_hparams": self.optimizer_hparams,
            "seed": self.seed,
            "description": self.description,
            "loss_fn": str(self.loss_fn),
            "batch_size": self.batch_size,
            "n_iters": self.n_iters,
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

    def get_loss_fn(self):
        return self.loss_fn

    def get_dir_log(self):
        return self.dir_log

    def get_eval_freq(self):
        return self.eval_freq

    def get_n_iters(self):
        return self.n_iters


class Trainer:
    def __init__(self, model, loaders, config, domain):
        self.model = model

        self.loader_train = loaders.get_loader_train()
        self.loader_dev = loaders.get_loader_dev()
        self.loader_test = loaders.get_loader_test()

        self.device = config.get_device()
        self.loss_fn = config.get_loss_fn()
        self.dir_log = os.path.join(config.get_dir_log(), domain)
        self.eval_freq = config.get_eval_freq()
        self.n_iters = config.get_n_iters()

        # optimizer
        self.model = self.model.to(self.device)

        optimizer_hparams = config.get_optimizer_hparams()
        optimizer = config.get_optimizer()

        self.optimizer_label = optimizer(
            self.model.get_params_label(), **optimizer_hparams
        )

        # logging
        self.logger_train = Logger(self.dir_log, name="log_train")
        self.logger_dev = Logger(self.dir_log, name="log_dev")
        self.writer = SummaryWriter(log_dir=self.dir_log, flush_secs=120)

        # training progress
        self.accuracy_best = None
        self.loss_best = None

        self.iter_count = 0

    def train(self, n_iters=None, eval_freq=None):
        """
        n_iters: int or None
            number of training iterations
            defaults to self.n_iters
        eval_freq: int or None
            how often to run check dev set performance
            defaults to self.eval_freq
        """
        if n_iters is None:
            n_iters = self.n_iters
        if eval_freq is None:
            eval_freq = self.eval_freq
        progress_bar = tqdm(range(n_iters))
        for i in progress_bar:
            self.train_iter()
            if i % eval_freq == 0:
                self.eval()
            progress_bar.set_postfix(self._training_status())

    def _dummy_train(self, n_iters=None):
        print("Starting dummy train")
        if n_iters is None:
            n_iters = self.n_iters
        bar = tqdm(range(n_iters))
        for i in bar:
            print(f"Iter {i}")
            self._dummy_train_iter()
            if i % self.eval_freq == 0:
                self._dummy_eval()
            bar.set_postfix(self._training_status())

    def run(self, n_iters=None, verbose=True):
        """ """
        # run
        if verbose:
            print(f"[{datetime.now()}] Run start")

        # train
        if verbose:
            print(f"[{datetime.now()}] Training start")

        time_training_start = time()
        self.train(n_iters)
        time_training_end = time()

        if verbose:
            time_training_total = time_training_end - time_training_start
            time_training_total = time_training_total / 60  # minutes
            print(
                f"[{datetime.now()}] Training over in {time_training_total:.1f} minutes"
            )

        # test
        if verbose:
            print(f"[{datetime.now()}] Test start")

        time_test_start = time()
        results = self.test()
        time_test_end = time()

        if verbose:
            time_test_total = time_test_end - time_test_start
            time_test_total = time_test_total / 60  # minutes
            print(f"[{datetime.now()}] Testing over in {time_test_total:.1f} minutes")

        results = self.postprocess(results)

        return results

    def _dummy_run(self, n_iters=None):
        print("Starting dummy run")
        self._dummy_train(n_iters)
        results = self._dummy_test()
        self.postprocess(results)

    def postprocess(self, results):
        """ """
        # save metrics
        accuracy = results["accuracy"]
        metrics_dict = {
            "accuracy": accuracy,
        }
        savename = os.path.join(self.dir_log, "metrics.json")
        with open(savename, "w", encoding="utf-8") as file:
            file.write(json.dumps(metrics_dict))

        # save predictions per image
        preds = results["predictions"]

        images = []
        labels = []

        labels_test = os.path.join(self.dir_log, "split_test.txt")
        with open(labels_test, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(" ")

                image = " ".join(line[:-1])
                images.append(image)

                label = line[-1]
                labels.append(label)

        images = images[: len(preds)]
        labels = labels[: len(preds)]
        results_dict = {"images": images, "labels": labels, "preds": preds}
        savename = os.path.join(self.dir_log, "results.csv")
        pd.DataFrame(results_dict).to_csv(savename, index=False)
        return metrics_dict

    def _training_status(self):
        message = {}
        message["loss_best"] = f"{self.loss_best: .2f}"
        message["acc_best"] = f"{self.accuracy_best*100 :.1f}"
        return message

    def _dummy_train_iter(self):
        """
        Train one iteration.
        """
        self.model.train()
        all_inputs, all_labels, all_domains = next(self.loader_train)

        all_inputs = all_inputs.to(self.device)
        all_labels = all_labels.to(self.device)
        all_domains = all_domains.to(self.device)

        # logging
        tracking = {
            "loss_label": 0.0,
            "loss_domain": 0.0,
            "loss_task_label": 1.0,
            "loss_task_domain": 2.0,
            "loss_task": 3.0,
        }

        for key, value in tracking.items():
            self.logger_train.log(key, value)

        for key, value in tracking.items():
            self.writer.add_scalar(key + "_train", value, self.iter_count)

        # update best model
        if self.loss_best is None or self.loss_best > 0.0:
            self.loss_best = 0.0

        self.save_loggers()

        self.iter_count += 1

    def train_iter(self):
        """
        Train one iteration.
        """
        self.model.train()
        all_inputs, all_labels, all_domains = next(self.loader_train)

        all_inputs = all_inputs.to(self.device)
        all_labels = all_labels.to(self.device)

        # embed
        emb_label = self.model.embed_label(all_inputs)
        # logit
        logit_label = self.model.classifier_label(emb_label)

        # calculate loss
        loss = 0.0
        # loss label
        loss_label = self.loss_fn(logit_label, all_labels)

        loss = loss_label

        # zero grads
        self.optimizer_label.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        self.optimizer_label.step()

        # logging
        tracking = {
            "loss_label": loss_label.item(),
        }

        for key, value in tracking.items():
            self.logger_train.log(key, value)

        for key, value in tracking.items():
            self.writer.add_scalar(key + "_train", value, self.iter_count)

        # update best model
        if self.loss_best is None or self.loss_best > loss.item():
            self.loss_best = loss.item()

        self.save_loggers()

        self.iter_count += 1

    def eval(self):
        """
        Run evaluation.
        """
        self.model.eval()

        accuracies = []
        losses = []

        with torch.no_grad():
            # run each domain loader separately
            for loader in self.loader_dev.loaders:
                predictions = []
                truths = []
                loss = 0.0

                for inputs, labels in loader:
                    truths.append(labels)
                    # move to device
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # embed
                    emb_label = self.model.embed_label(inputs)
                    # logits
                    logit_label = self.model.classifier_label(emb_label)
                    # loss
                    loss += self.loss_fn(logit_label, labels) * len(inputs)
                    # predictions
                    predictions_batch = torch.argmax(logit_label, -1)
                    predictions.append(predictions_batch.cpu())

                predictions = torch.cat(predictions, 0)
                truths = torch.cat(truths, 0)
                accuracy = (predictions == truths).float().mean()
                loss = loss / len(truths)

                accuracies.append(accuracy.item())
                losses.append(loss.item())

        # average over all domains
        accuracy = sum(accuracies) / len(accuracies)
        loss = sum(losses) / len(losses)

        tracking = {
            "accuracy": accuracy,
            "loss": loss,
        }

        for key, value in tracking.items():
            self.logger_dev.log(key, value)

        for key, value in tracking.items():
            self.writer.add_scalar(key + "_dev", value, self.iter_count)

        self.save_loggers()

        # update best model
        if self.accuracy_best is None or accuracy > self.accuracy_best:
            self.accuracy_best = accuracy
            name = f"weights_best_{self.iter_count:05d}"
            self.model_best_name = name
            self.save(name, save_optimizer=False)

    def _dummy_eval(self):
        """
        Run evaluation.
        """
        predictions = []
        truths = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels, domains in self.loader_dev:
                # move to device
                inputs = inputs.to(self.device)
                # embed
                logit_label = torch.rand(len(inputs), 10)
                # predictions
                predictions_batch = torch.argmax(logit_label, -1)
                predictions.append(predictions_batch.cpu())
                # truths
                truths.append(labels)

        predictions = torch.cat(predictions, 0)
        truths = torch.cat(truths, 0)

        accuracy = (predictions == truths).float().mean()

        tracking = {
            "accuracy": accuracy.item(),
        }

        for key, value in tracking.items():
            self.logger_dev.log(key, value)

        for key, value in tracking.items():
            self.writer.add_scalar(key + "_dev", value, self.iter_count)

        self.save_loggers()

        # update best model
        if self.accuracy_best is None or accuracy > self.accuracy_best:
            self.accuracy_best = accuracy
            name = f"weights_best_{self.iter_count:05d}"
            self.save(name, save_optimizer=False)

    def test(self):
        """
        Run testing.
        """
        self.load_best()

        predictions = []
        truths = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels, domains in self.loader_test:
                # move to device
                inputs = inputs.to(self.device)
                # embed
                emb_label = self.model.embed_label(inputs)
                # logits
                logit_label = self.model.classifier_label(emb_label)
                # predictions
                predictions_batch = torch.argmax(logit_label, -1)
                predictions.append(predictions_batch.cpu())
                # truths
                truths.append(labels)

        predictions = torch.cat(predictions, 0)
        truths = torch.cat(truths, 0)

        accuracy = (predictions == truths).float().mean()

        tracking = {
            "predictions": predictions.tolist(),
            "accuracy": accuracy.item(),
        }

        return tracking

    def _dummy_test(self):
        """
        Run testing.
        """
        predictions = []
        truths = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels, domains in self.loader_test:
                # move to device
                inputs = inputs.to(self.device)
                # logits
                logit_label = torch.rand(len(inputs), 10)
                # predictions
                predictions_batch = torch.argmax(logit_label, -1)
                predictions.append(predictions_batch.cpu())
                # truths
                truths.append(labels)

        predictions = torch.cat(predictions, 0)
        truths = torch.cat(truths, 0)

        accuracy = (predictions == truths).float().mean()

        tracking = {
            "predictions": predictions.tolist(),
            "accuracy": accuracy.item(),
        }

        return tracking

    def save(self, name, save_optimizer=True, save_loggers=True):
        """
        Save model, optimizer, logger.
        """
        savepath = os.path.join(self.dir_log, name + ".pth")
        to_save = {}
        to_save["iter_count"] = self.iter_count
        to_save["model"] = self.model.state_dict()
        if save_optimizer:
            to_save["optimizer"] = self.optimizer.state_dict()
        torch.save(to_save, savepath)

        if save_loggers:
            self.save_loggers()

    def load(self, state_dict, logger_train=None, logger_dev=None, load_device=None):
        """
        Load model, optimizer, logger.
        """
        if load_device is None:
            load_device = self.device
        state_dict = torch.load(state_dict, map_location=load_device)
        self.model.load_state_dict(state_dict["model"])
        self.model.to(self.device)

        self.iter_count = state_dict["iter_count"]

        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if logger_train is not None:
            self.logger_train.load(logger_train)

        if logger_dev is not None:
            self.logger_dev.load(logger_dev)

    def load_best(self, load_device=None):
        """
        Load best model.
        """
        savepath = os.path.join(self.dir_log, self.model_best_name + ".pth")
        if load_device is None:
            load_device = self.device
        state_dict = torch.load(savepath, map_location=load_device)
        self.model.load_state_dict(state_dict["model"])
        self.model.to(self.device)

    def get_best(self, load_device=None):
        self.load_best(load_device=load_device)
        return self.model

    def save_loggers(self):
        self.logger_train.save()
        self.logger_train.save()
