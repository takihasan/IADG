#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import os
import json

from torchvision import transforms as T
from torchvision.transforms import functional as TF


class RotateAngles:
    def __init__(self, angles, p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, image):
        p = random.uniform(0, 1)
        if self.p > p:
            angle = random.choice(self.angles)
            image = TF.rotate(image, angle)
        return image


class Augmentation:
    """
    Augmetation class.
    """

    def __init__(
        self,
        size=(224, 224),
        random_resized_crop=False,
        random_resized_crop_scale=(0.7, 1.0),
        random_vertical_flip=None,
        random_horizontal_flip=None,
        rotate_angles=None,
        color_jitter=False,
        color_jitter_args=(0.3, 0.3, 0.3, 0.3),
        random_grayscale=None,
        to_tensor=True,
        normalize=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    ):
        augs = []
        aug_names = []

        # crop or resize
        if random_resized_crop == True:
            augs.append(T.RandomResizedCrop(size, random_resized_crop_scale))
            aug_names.append(
                {
                    "RandomResizeCrop": {
                        "size": size,
                        "scale": random_resized_crop_scale,
                    }
                }
            )
        else:
            augs.append(T.Resize(size))
            aug_names.append({"Resize": {"size": size}})

        # flip
        if random_vertical_flip is not None:
            augs.append(T.RandomVerticalFlip(random_vertical_flip))
            aug_names.append({"RandomVerticalFlip": random_vertical_flip})
        if random_horizontal_flip is not None:
            augs.append(T.RandomHorizontalFlip(random_horizontal_flip))
            aug_names.append({"RandomHorizontalFlip": random_horizontal_flip})

        # rotate
        if rotate_angles is not None:
            angles = rotate_angles.get("angles", [0, 90, 180, 270])
            rotate_p = rotate_angles.get("p", 0.5)
            augs.append(RotateAngles(angles, rotate_p))
            aug_names.append(
                {
                    "RotateAngles": {
                        "angles": angles,
                        "p": rotate_p,
                    }
                }
            )

        # color jitter
        if color_jitter:
            augs.append(T.ColorJitter(*color_jitter_args))
            aug_names.append({"ColorJitter": color_jitter_args})

        # random grayscale
        if random_grayscale is not None:
            augs.append(T.RandomGrayscale(random_grayscale))
            aug_names.append({"RandomGrayscale": random_grayscale})

        if to_tensor:
            augs.append(T.ToTensor())
            aug_names.append("ToTensor")

        if normalize:
            augs.append(
                T.Normalize(
                    mean=normalize_mean,
                    std=normalize_std,
                ),
            )
            aug_names.append(
                {"Normalize": {"mean": normalize_mean, "std": normalize_std}}
            )

        self.aug = T.Compose(augs)
        self.aug_names = aug_names

    def save(self, basedir, filename):
        savename = os.path.join(basedir, filename)

        with open(savename, "w", encoding="utf-8") as file:
            file.write(json.dumps(self.aug_names))

    def get_transform(self):
        return self.aug

    def __call__(self, x):
        return self.aug(x)


def get_transform_train():
    """
    Get training transformation.

    Returns:
        callable:
            image transformation
    """
    transform = Augmentation(
        random_resized_crop=True,
        random_vertical_flip=0.5,
        random_horizontal_flip=0.5,
        color_jitter=True,
        random_grayscale=0.1,
    )

    return transform


def get_transform_dev():
    """
    Get eval transformation.

    Returns:
        callable:
            image transformation
    """
    transform = Augmentation()

    return transform


def get_transform_test():
    """
    Get test transformation.

    Returns:
        callable:
            image transformation
    """
    transform = Augmentation()

    return transform


def get_transform_to_tensor():
    """
    Get basic image to tensor transformation.

    Returns:
        callable:
            image transformation
    """
    transform = Augmentation()

    return transform


def get_transform_adapt_weak():
    """
    Get adaptation weak transformation.

    Returns:
        callable:
            image transformation
    """
    transform = Augmentation(
        random_resized_crop=True,
        random_vertical_flip=0.5,
        random_horizontal_flip=0.5,
        rotate_angles={"angles": [0, 90, 180, 270], "p": 0.5},
    )

    return transform


def get_transform_adapt_strong():
    """
    Get adaptation strong transformation.

    Returns:
        callable:
            image transformation
    """
    transform = Augmentation(
        random_resized_crop=True,
        random_vertical_flip=0.5,
        random_horizontal_flip=0.5,
        rotate_angles={"angles": [0, 90, 180, 270], "p": 0.5},
        color_jitter=True,
    )

    return transform
