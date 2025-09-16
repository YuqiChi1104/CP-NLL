# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original CIFAR-100 dataset script from https://huggingface.co/datasets/cifar100
# modified by Tomas Gajarsky adapting the commonly used data imbalancing from e.g.: 
# https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
# https://github.com/dvlab-research/Imbalanced-Learning/blob/main/ResCom/datasets/cifar_lt.py
# https://github.com/XuZhengzhuo/LiVT/blob/main/util/datasets.py

# Lint as: python3
"""CIFAR-100-LT Dataset"""


import pickle
from typing import Dict, Iterator, List, Tuple, BinaryIO

import numpy as np

import datasets
from datasets.tasks import ImageClassification


_CITATION = """\
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
"""

_DESCRIPTION = """\
The CIFAR-100-LT dataset is comprised of under 60,000 color images, each measuring 32x32 pixels, 
distributed across 100 distinct classes. 
The number of samples within each class decreases exponentially with factors of 10, 20, 50, 100, or 200. 
The dataset includes 10,000 test images, with 100 images per class, 
and fewer than 50,000 training images. 
These 100 classes are further organized into 20 overarching superclasses. 
Each image is assigned two labels: a fine label denoting the specific class, 
and a coarse label representing the associated superclass.
"""

_DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

_FINE_LABEL_NAMES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "cra",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

_COARSE_LABEL_NAMES = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
]


class Cifar100LTConfig(datasets.BuilderConfig):
    """BuilderConfig for CIFAR-100-LT."""

    def __init__(self, imb_type: str, imb_factor: float, rand_number: int = 0, cls_num: int = 100, **kwargs):
        """BuilderConfig for CIFAR-100-LT.

        Args:
            imb_type (str): imbalance type, including 'exp', 'step'.
            imb_factor (float): imbalance factor.
            rand_number (int): random seed, default: 0.
            cls_num (int): number of classes, default: 100.
            **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.rand_number = rand_number
        self.cls_num = cls_num

        np.random.seed(self.rand_number)


class Cifar100(datasets.GeneratorBasedBuilder):
    """CIFAR-100 Dataset"""

    BUILDER_CONFIGS = [
        Cifar100LTConfig(
            name="r-10",
            description="CIFAR-100-LT-r-10 Dataset",
            imb_type='exp',
            imb_factor=1/10,
            rand_number=0,
            cls_num=100,
        ),
        Cifar100LTConfig(
            name="r-20",
            description="CIFAR-100-LT-r-20 Dataset",
            imb_type='exp',
            imb_factor=1/20,
            rand_number=0,
            cls_num=100,
        ),
        Cifar100LTConfig(
            name="r-50",
            description="CIFAR-100-LT-r-50 Dataset",
            imb_type='exp',
            imb_factor=1/50,
            rand_number=0,
            cls_num=100,
        ),
        Cifar100LTConfig(
            name="r-100",
            description="CIFAR-100-LT-r-100 Dataset",
            imb_type='exp',
            imb_factor=1/100,
            rand_number=0,
            cls_num=100,
        ),
        Cifar100LTConfig(
            name="r-200",
            description="CIFAR-100-LT-r-200 Dataset",
            imb_type='exp',
            imb_factor=1/200,
            rand_number=0,
            cls_num=100,
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "img": datasets.Image(),
                    "fine_label": datasets.features.ClassLabel(names=_FINE_LABEL_NAMES),
                    "coarse_label": datasets.features.ClassLabel(names=_COARSE_LABEL_NAMES),
                }
            ),
            supervised_keys=None,  # Probably needs to be fixed.
            homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="img", label_column="fine_label")],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        archive = dl_manager.download(_DATA_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "test"}
            ),
        ]
    

    def _generate_examples(self, files: Iterator[Tuple[str, BinaryIO]], split: str) -> Iterator[Dict]:
        """This function returns the examples in the array form."""
        for path, fo in files:
            if path == f"cifar-100-python/{split}":
                dict = pickle.load(fo, encoding="bytes")

                fine_labels = dict[b"fine_labels"]
                coarse_labels = dict[b"coarse_labels"]
                images = dict[b"data"]

                if split == "train":
                    indices = self._imbalance_indices()
                else:
                    indices = range(len(fine_labels))

                for idx in indices:

                    img_reshaped = np.transpose(np.reshape(images[idx], (3, 32, 32)), (1, 2, 0))

                    yield idx, {
                        "img": img_reshaped,
                        "fine_label": fine_labels[idx],
                        "coarse_label": coarse_labels[idx],
                    }
                break

    def _generate_indices_targets(self, files: Iterator[Tuple[str, BinaryIO]], split: str) -> Iterator[Dict]:
        """This function returns the examples in the array form."""

        for path, fo in files:
            if path == f"cifar-100-python/{split}":
                dict = pickle.load(fo, encoding="bytes")

                fine_labels = dict[b"fine_labels"]
                coarse_labels = dict[b"coarse_labels"]

                for idx, _ in enumerate(fine_labels):
                    yield idx, {
                        "fine_label": fine_labels[idx],
                        "coarse_label": coarse_labels[idx],
                    }
                break

    def _get_img_num_per_cls(self, data_length: int) -> List[int]:
        """Get the number of images per class given the imbalance ratio and total number of images."""
        img_max = data_length / self.config.cls_num
        img_num_per_cls = []
        if self.config.imb_type == 'exp':
            for cls_idx in range(self.config.cls_num):
                num = img_max * (self.config.imb_factor**(cls_idx / (self.config.cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif self.config.imb_type == 'step':
            for cls_idx in range(self.config.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.config.cls_num // 2):
                img_num_per_cls.append(int(img_max * self.config.imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * self.config.cls_num)
        return img_num_per_cls

    def _gen_imbalanced_data(self, img_num_per_cls: List[int], targets: List[int]) -> Tuple[List[int], Dict[int, int]]:
        """This function returns the indices of imbalanced CIFAR-100-LT dataset and the number of images per class."""
        new_indices = []
        targets_np = np.array(targets, dtype=np.int64)
        classes = np.unique(targets_np)
        num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_indices.extend(selec_idx.tolist())
        return new_indices, num_per_cls_dict
    
    def _imbalance_indices(self) -> List[int]:
        """This function returns the indices of imbalanced CIFAR-100-LT dataset."""
        dl_manager = datasets.DownloadManager()
        archive = dl_manager.download(_DATA_URL)
        data_iterator = self._generate_indices_targets(dl_manager.iter_archive(archive), "train")

        indices = []
        targets = []
        for i, targets_dict in data_iterator:
            indices.append(i)
            targets.append(targets_dict["fine_label"])

        data_length = len(indices)
        img_num_per_cls = self._get_img_num_per_cls(data_length)
        new_indices, _ = self._gen_imbalanced_data(img_num_per_cls, targets)
        return new_indices
