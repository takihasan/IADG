#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset related modules.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .dataloader import InfiniteDataLoader, MultipleLoader


def split_array(images, labels, ratio=0.8, seed=None, random_gen=None):
    """
    Random split numpy array.

    Args:
        images: np.array
            data
        labels: np.array
            data
        ratio: float
            split ratio

    Return:
        tuple (x_1, y_1), (x_2, y_2)
    """
    if random_gen is None:
        if seed is not None:
            random_gen = np.random.RandomState(seed)
        else:
            random_gen = np.random

    total = len(images)
    split_size = int(ratio * total)

    index = np.arange(total)
    random_gen.shuffle(index)

    index_1 = index[:split_size]
    index_2 = index[split_size:]

    x_1 = images[index_1]
    x_2 = images[index_2]

    y_1 = labels[index_1]
    y_2 = labels[index_2]

    return (x_1, y_1), (x_2, y_2)


def save_domain_split(domain_split, base_dir):
    for split, data in domain_split.items():
        path = os.path.join(base_dir, f"split_{split}.txt")
        lines = []
        for _, domain_data in data.items():
            images = domain_data["images"]
            labels = domain_data["labels"]
            lines += [f"{image} {label}" for image, label in zip(images, labels)]
        with open(path, "a", encoding="utf-8") as file:
            file.write("\n".join(lines))


def dict_to_json(class_to_index, base_dir, filename):
    savename = os.path.join(base_dir, filename)
    with open(savename, "a", encoding="utf-8") as file:
        file.write(json.dumps(class_to_index))


class DomainDataset(Dataset):
    """
    Basic torch dataset for one domain.
    """

    def __init__(self, images, labels, transform=None):
        super().__init__()

        self.images = np.array(images)
        self.labels = np.array(labels)

        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # transform
        image = Image.open(image)
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


class AdaptDataset(Dataset):
    """
    Basic torch dataset for one domain.
    """

    def __init__(self, images, labels):
        super().__init__()

        self.images = np.array(images)
        self.labels = np.array(labels)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]

        # transform
        image = Image.open(image_path)

        return {"image": image, "label": label, "path": "image_path"}

    def __len__(self):
        return len(self.images)


class DomainData:
    """
    Assumed file tree:
    root_dir
    ├── domain_1
        ├── class_1
            ├── image_001.jpg
            ├── image_001.jpg
            ...
        ├── class_2
            ├── image_001.jpg
            ├── image_001.jpg
            ...
        ...
    ├── domain_2
        ├── class_1
            ├── image_001.jpg
            ├── image_001.jpg
            ...
        ├── class_2
            ├── image_001.jpg
            ├── image_001.jpg
            ...
        ...
    ...
    """

    def __init__(self, root_dir, extensions=("jpg", "png", "jpeg"), seed=None):
        """
        Args:
            root_dir: str
                path to root domain directory
            extensions: tuple
                all image extensions
        """
        domain_paths = [p for p in Path(root_dir).glob("*") if p.is_dir()]
        domain_names = [p.stem for p in domain_paths]

        classes = [{c.stem for c in p.glob("*") if c.is_dir()} for p in domain_paths]
        common_classes = set.intersection(*classes)
        # make class mapping
        class_to_index = {c: i for i, c in enumerate(sorted(list(common_classes)))}

        # get all files and classes
        all_data = [
            [[str(i), i.parent.stem] for i in p.rglob("*") if i.is_file()]
            for p in domain_paths
        ]
        # get only files belonging to common classes
        all_data = [
            [[img, cls] for img, cls in domain_data if cls in class_to_index]
            for domain_data in all_data
        ]
        # get all files with image extensions
        all_data = [
            [[img, cls] for img, cls in domain_data if img.endswith(extensions)]
            for domain_data in all_data
        ]
        # get images
        images = [
            [img for img, _ in domain_data if img.endswith(extensions)]
            for domain_data in all_data
        ]
        # map images to labels
        labels = [
            [class_to_index[cls] for _, cls in domain_data] for domain_data in all_data
        ]

        self.data = {
            d: {"images": np.array(imgs), "labels": np.array(cls)}
            for d, imgs, cls in zip(domain_names, images, labels)
        }
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.domains = sorted(domain_names)

        if seed is not None:
            self.random_gen = np.random.RandomState(seed)
        else:
            self.random_gen = np.random

    def get_domain_data(self, domain):
        return self.data[domain]

    def get_n_domains(self):
        return len(self.domains)

    def get_n_classes(self):
        return len(self.class_to_index)

    def get_class_to_index(self):
        return self.class_to_index

    def get_index_to_class(self):
        return self.index_to_class

    def get_domains(self):
        return self.domains

    def get_domain_split(self, test_domain, ratio=0.8):
        """
        Random split of for all domains except test domain.

        Args:
            test_domain: str or int
                if string, match domain by name
                if int, match domain by index in domain list
            ratio: float
                split ratio

        Return:
            dict of dicts:
                dict with keys "train", "dev", "test"
                each value is dict, with keys "images", "labels"
        """
        if isinstance(test_domain, int):
            test_domain = self.domains[test_domain]

        test_data = {test_domain: self.data.get(test_domain)}

        train_data = {}
        dev_data = {}
        for domain in self.domains:
            if domain == test_domain:
                continue
            domain_split = self.get_single_domain_split(domain, ratio)
            train_data[domain] = domain_split["train"]
            dev_data[domain] = domain_split["dev"]

        to_return = {"train": train_data, "dev": dev_data, "test": test_data}
        return to_return

    def get_single_domain_split(self, domain, ratio=0.8):
        """
        Split domain into train and dev sets.

        Args:
            data: dict
                dict with keys "images" and "labels"
            ratio: float
                data split ratio

        Return:
            dict:
                dict with keys "images", "labels"
        """
        data = self.data[domain]

        images = data["images"]
        labels = data["labels"]
        (x_train, y_train), (x_dev, y_dev) = split_array(
            images, labels, ratio, random_gen=self.random_gen
        )

        train_data = {"images": x_train, "labels": y_train}
        dev_data = {"images": x_dev, "labels": y_dev}

        return {"train": train_data, "dev": dev_data}

    def get_loaders(
        self,
        test_domain,
        transforms,
        batch_size=32,
        ratio=0.8,
        loader_types=None,
        get_test=False,
        domain_split=None,
        seed=None,
        num_workers_train=1,
        num_workers_dev=4,
        num_workers_test=8,
    ):
        if loader_types is None:
            loader_types = {
                "train": InfiniteDataLoader,
                "dev": DataLoader,
                "test": DataLoader,
            }
        if isinstance(batch_size, int):
            batch_sizes = {
                "train": batch_size,
                "dev": batch_size * 2,
                "test": batch_size * 2,
            }
        num_workers = {
            "train": num_workers_train,
            "dev": num_workers_dev,
            "test": num_workers_test,
        }

        splits = ["train", "dev"]
        if get_test:
            splits.append("test")

        if domain_split is None:
            domain_split = self.get_domain_split(test_domain, ratio)

        all_loaders = {}
        domains = {}
        for split in splits:
            data_split = domain_split[split]
            transform = transforms[split]
            loader_type = loader_types[split]
            batch_size = batch_sizes[split]

            domains[split] = list(data_split.keys())
            # datasets
            datasets = {
                domain: DomainDataset(data["images"], data["labels"], transform)
                for domain, data in data_split.items()
            }
            # dataloaders
            loaders = {
                domain: loader_type(
                    dataset=ds,
                    batch_size=batch_size,
                    num_workers=num_workers[split],
                )
                for domain, ds in datasets.items()
            }
            # multi loader
            loader = MultipleLoader(loaders)
            all_loaders[split] = loader

        return {"loaders": all_loaders, "split": domain_split}

    def get_adapt_dataset(
        self,
        test_domain,
    ):
        """
        Get dataloader for adaptation.
        """
        data = self.get_domain_data(test_domain)
        dataset = AdaptDataset(images=data["images"], labels=data["labels"])
        return dataset
