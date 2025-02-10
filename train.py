#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training module.

Example:
    python train.py --dir-data DATA --learning-rate 0.003
"""

import argparse
import os
import random
from datetime import datetime
from time import time

import torch

from data import transform
from data.dataset import DomainData, dict_to_json, save_domain_split
from methods.CDTAG.model import TrainingModel
from methods.CDTAG.trainer import Trainer, TrainingConfig
from methods.TTA import AdaptConfig, Adapter
from utils import Loaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])

    parser.add_argument("--dir-data", type=str, help="Base directory of data.")
    parser.add_argument(
        "--dir-logs", type=str, default="logs", help="Base directory of logs."
    )
    parser.add_argument(
        "--test-domain", type=str, default=None, help="Test domain. If None, use all."
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name.")
    parser.add_argument(
        "--eval-freq", type=int, default=20, help="How often to run the evaulation."
    )
    parser.add_argument("--seed", type=int, default=None, help="Global seed.")
    parser.add_argument(
        "--seed-data", type=int, default=None, help="Seed for splitting data."
    )
    parser.add_argument(
        "--seed-train", type=int, default=None, help="Seed for training."
    )
    parser.add_argument("--seed-adapt", type=int, default=None, help="Seed for TTA.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="resnet50",
        help="Base model for feature extraction.",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument(
        "--n-iters", type=int, default=5000, help="Number of training iterations."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Run description.",
    )
    # adaptation config
    parser.add_argument(
        "--adapt",
        type=bool,
        default=False,
        help="Use TTA.",
    )
    parser.add_argument(
        "--adapt-learning-rate",
        type=float,
        default=3e-4,
        help="Adaptation learning rate.",
    )
    parser.add_argument(
        "--adapt-weight-decay",
        type=float,
        default=1e-6,
        help="Weight decay.",
    )
    parser.add_argument(
        "--adapt-n-weak",
        type=int,
        default=16,
        help="Number of weak augmented images in adapt batch.",
    )
    parser.add_argument(
        "--adapt-n-strong",
        type=int,
        default=8,
        help="Number of strong augmented images in adapt batch.",
    )

    # parse arguments
    args = parser.parse_args()
    dir_data = args.dir_data
    dir_logs = args.dir_logs
    run_name = args.run_name
    seed = args.seed
    seed_data = args.seed_data
    seed_train = args.seed_train
    seed_adapt = args.seed_adapt
    batch_size = args.batch_size

    # determinstic
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # make run dir
    if run_name is None:
        run_name = str(int(time() * 1000)).split(".", maxsplit=1)[0]
    dir_log_base = os.path.join(dir_logs, run_name)
    os.makedirs(dir_log_base, exist_ok=True)

    # load data
    domain_data = DomainData(dir_data, seed=seed_data)
    domains = domain_data.get_domains()
    n_domains = domain_data.get_n_domains() - 1
    n_classes = domain_data.get_n_classes()
    # save class2index
    class_to_index = domain_data.get_class_to_index()
    dict_to_json(class_to_index, dir_log_base, "class_to_index.json")

    # # # TRAINING # # #
    # train augmentations
    transform_train = transform.get_transform_train()
    transform_dev = transform.get_transform_dev()
    transform_test = transform.get_transform_test()

    # save augmentation config
    transform_train.save(dir_log_base, "aug_train.json")
    transform_dev.save(dir_log_base, "aug_dev.json")
    transform_test.save(dir_log_base, "aug_test.json")

    transforms = {
        "train": transform_train.get_transform(),
        "dev": transform_dev.get_transform(),
        "test": transform_test.get_transform(),
    }

    # optimizer
    lr = args.learning_rate
    weight_decay = args.weight_decay
    optimizer_train = torch.optim.Adam
    optimizer_hparams_train = {"lr": lr, "weight_decay": weight_decay}

    # training config
    train_config = TrainingConfig(
        optimizer=optimizer_train,
        optimizer_hparams=optimizer_hparams_train,
        batch_size=batch_size,
        seed=seed_train,
        dir_log=dir_logs,
        run_name=run_name,
        description=args.description,
        eval_freq=args.eval_freq,
        n_iters=args.n_iters,
    )
    train_config.save()

    # # # ADAPTATION # # #
    # adapt augmentations
    transform_adapt_weak = transform.get_transform_adapt_weak()
    transform_adapt_strong = transform.get_transform_adapt_strong()
    transform_adapt_orig = transform.get_transform_to_tensor()
    transform_adapt_weak.save(dir_log_base, "aug_adapt_weak.json")
    transform_adapt_strong.save(dir_log_base, "aug_adapt_strong.json")
    transform_adapt_orig.save(dir_log_base, "aug_adapt_orig.json")
    transforms_adapt = {
        "weak": transform_adapt_weak,
        "strong": transform_adapt_strong,
        "orig": transform_adapt_orig,
    }

    # adapt optimizer
    lr_adapt = args.adapt_learning_rate
    weight_decay_adapt = args.adapt_weight_decay
    optimizer_adapt = torch.optim.Adam
    optimizer_hparams_adapt = {
        "lr": lr_adapt,
        "weight_decay": weight_decay_adapt,
    }
    # config
    adapt_config = AdaptConfig(
        optimizer=optimizer_adapt,
        optimizer_hparams=optimizer_hparams_adapt,
        transforms=transforms_adapt,
        n_weak=args.adapt_n_weak,
        n_strong=args.adapt_n_strong,
        device=None,
        seed=seed_adapt,
        dir_log=dir_logs,
        run_name=run_name,
    )
    adapt_config.save()

    # get loaders
    test_domain = args.test_domain
    if test_domain is not None:
        test_domains = [test_domain]
    else:
        test_domains = domains

    print("-" * 32)
    time_start = time()
    print(f"[{datetime.now()}] Starting experiment")
    print(args.description)
    print("-" * 32)
    results_all = {}
    results_adapt = {}
    for test_domain_str in test_domains:
        time_start_domain = time()
        test_domain = domains.index(test_domain_str)
        print(f"[{datetime.now()}] Test domain: {test_domain_str}")
        dir_log = os.path.join(dir_log_base, test_domain_str)
        os.makedirs(dir_log, exist_ok=True)
        loaders = domain_data.get_loaders(
            test_domain,
            transforms,
            batch_size=batch_size,
            get_test=True,
        )
        domain_split = loaders["split"]
        loaders = loaders["loaders"]
        loaders = Loaders(loaders)

        # save domain split
        save_domain_split(domain_split, dir_log)

        # model
        model = TrainingModel(
            n_classes,
            n_domains,
            args.base_model,
            args.dropout,
        )
        # trainer
        trainer = Trainer(model, loaders, train_config, domain=test_domain_str)
        # train
        results = trainer.run()
        print(results)
        results_all[test_domain_str] = results["accuracy"]
        print("-" * 32)

        # adapt
        if args.adapt:
            print("Starting adaptation")
            dataset_adapt = domain_data.get_adapt_dataset(test_domain_str)
            model = trainer.get_best()
            adapter = Adapter(
                model, dataset_adapt, adapt_config, domain=test_domain_str
            )
            results_adapt = adapter.run()
            print(results_adapt)
            results_adapt[test_domain_str] = results_adapt["accuracy"]
            print("-" * 32)
    time_end = time()
    time_total = (time_end - time_start) / 60
    print(f"Run over in {time_total:.1f} minutes")
    print("-" * 32)

    # save results to log file
    results_file = os.path.join(dir_log_base, "results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        # vanilla results
        for test_domain, accuracy in results_all.items():
            msg = f"{test_domain}: {accuracy * 100:.3f} %"
            print(msg)
            f.write(msg + "\n")

        vals = results_all.values()
        avg = sum(vals) / len(vals)
        msg = f"Average: {avg * 100:.3f} %"
        print("-" * 32)
        print(msg)
        f.write(msg)

        if args.adapt:
            # TTA results
            for test_domain, accuracy in results_adapt.items():
                msg = f"{test_domain} TTA: {accuracy * 100:.3f} %"
                print(msg)
                f.write(msg + "\n")

            vals = results_adapt.values()
            avg = sum(vals) / len(vals)
            msg = f"Average TTA: {avg * 100:.3f} %"
            print("-" * 32)
            print(msg)
            f.write(msg)
