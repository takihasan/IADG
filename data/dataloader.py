#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import (
    DataLoader,
    Sampler,
    BatchSampler,
    RandomSampler,
    WeightedRandomSampler,
)


class InfiniteSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, weights=None, num_workers=1):
        self.batch_size = batch_size
        if weights:
            sampler = WeightedRandomSampler(weights, num_samples=batch_size)
        else:
            sampler = RandomSampler(dataset, replacement=True)

        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
        infinite_batch_sampler = InfiniteSampler(batch_sampler)

        loader = DataLoader(
            dataset,
            batch_sampler=infinite_batch_sampler,
            num_workers=num_workers,
        )
        self.loader_iter = iter(loader)


    def __iter__(self):
        while True:
            yield next(self.loader_iter)


class MultipleLoader:
    def __init__(self, loaders):
        self.domains = sorted(list(loaders.keys()))
        self.loaders = [loaders[domain] for domain in self.domains]
        self.batch_size = self.loaders[0].batch_size
        self.loader_iters = [iter(loader) for loader in self.loaders]

    def __iter__(self):
        self.loader_iters = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        vals = []
        dataset_index = []

        for i, loader in enumerate(self.loader_iters):
            try:
                val = next(loader)
                vals.append(val)
                dataset_index.append(torch.LongTensor([i] * len(val[0])))
            except StopIteration:
                # dataloader is done
                continue

        if len(vals) == 0:
            # all dataloaders are done
            raise StopIteration

        images, labels = list(zip(*vals))
        images = torch.cat(images)
        labels = torch.cat(labels)
        dataset_index = torch.cat(dataset_index)
        return images, labels, dataset_index
