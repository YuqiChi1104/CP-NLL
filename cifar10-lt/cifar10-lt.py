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

# Original CIFAR-10 dataset script from https://huggingface.co/datasets/cifar100
# modified by Tomas Gajarsky adapting the commonly used data imbalancing from e.g.: 
# https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
# https://github.com/dvlab-research/Imbalanced-Learning/blob/main/ResCom/datasets/cifar_lt.py
# https://github.com/XuZhengzhuo/LiVT/blob/main/util/datasets.py

# Lint as: python3
"""CIFAR-10-LT Dataset"""


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
The CIFAR-10-LT imbalanced dataset is comprised of under 60,000 color images, each measuring 32x32 pixels, 
distributed across 10 distinct classes.  
The dataset includes 10,000 test images, with 1000 images per class, 
and fewer than 50,000 training images.
The number of samples within each class of the train set decreases exponentially with factors of 10, 20, 50, 100, or 200.
"""

_DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class Cifar10LTConfig(datasets.BuilderConfig):
    """BuilderConfig for CIFAR-10-LT."""

    def __init__(self, imb_type: str, imb_factor: float, rand_number: int = 0, cls_num: int = 10, **kwargs):
        """BuilderConfig for CIFAR-10-LT.
        Args:
            imb_type (str): imbalance type, including 'exp', 'step'.
            imb_factor (float): imbalance factor.
            rand_number (int): random seed, default: 0.
            cls_num (int): number of classes, default: 10.
            **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.rand_number = rand_number
        self.cls_num = cls_num

        np.random.seed(self.rand_number)


class Cifar10(datasets.GeneratorBasedBuilder):
    """CIFAR-10 Dataset"""

    BUILDER_CONFIGS = [
        Cifar10LTConfig(
            name="r-10",
            description="CIFAR-10-LT-r-10 Dataset",
            imb_type='exp',
            imb_factor=1/10,
            rand_number=0,
            cls_num=10,
        ),
        Cifar10LTConfig(
            name="r-20",
            description="CIFAR-10-LT-r-20 Dataset",
            imb_type='exp',
            imb_factor=1/20,
            rand_number=0,
            cls_num=10,
        ),
        Cifar10LTConfig(
            name="r-50",
            description="CIFAR-10-LT-r-50 Dataset",
            imb_type='exp',
            imb_factor=1/50,
            rand_number=0,
            cls_num=10,
        ),
        Cifar10LTConfig(
            name="r-100",
            description="CIFAR-10-LT-r-100 Dataset",
            imb_type='exp',
            imb_factor=1/100,
            rand_number=0,
            cls_num=10,
        ),
        Cifar10LTConfig(
            name="r-200",
            description="CIFAR-10-LT-r-200 Dataset",
            imb_type='exp',
            imb_factor=1/200,
            rand_number=0,
            cls_num=10,
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "img": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=None,  # Probably needs to be fixed.
            homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
            citation=_CITATION,
            task_templates=[ImageClassification(image_column="img", label_column="label")],
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
        if split == "train":
            batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

        if split == "test":
            batches = ["test_batch"]
        batches = [f"cifar-10-batches-py/{filename}" for filename in batches]

        for path, fo in files:

            if path in batches:
                dict = pickle.load(fo, encoding="bytes")

                labels = dict[b"labels"]
                images = dict[b"data"]

                if split == "train":
                    indices = self._imbalance_indices()
                else:
                    indices = range(len(labels))

                for idx in indices:

                    img_reshaped = np.transpose(np.reshape(images[idx], (3, 32, 32)), (1, 2, 0))

                    yield f"{path}_{idx}", {
                        "img": img_reshaped,
                        "label": labels[idx],
                    }
                break

    def _generate_indices_targets(self, files: Iterator[Tuple[str, BinaryIO]], split: str) -> Iterator[Dict]:
        """This function returns the examples in the array form."""

        if split == "train":
            batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

        if split == "test":
            batches = ["test_batch"]
        batches = [f"cifar-10-batches-py/{filename}" for filename in batches]

        for path, fo in files:

            if path in batches:
                dict = pickle.load(fo, encoding="bytes")

                labels = dict[b"labels"]

                for idx, _ in enumerate(labels):
                    yield f"{path}_{idx}", {
                        "label": labels[idx],
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
        """This function returns the indices of imbalanced CIFAR-10-LT dataset and the number of images per class."""
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
        """This function returns the indices of imbalanced CIFAR-10-LT dataset."""
        dl_manager = datasets.DownloadManager()
        archive = dl_manager.download(_DATA_URL)
        data_iterator = self._generate_indices_targets(dl_manager.iter_archive(archive), "train")

        indices = []
        targets = []
        for i, targets_dict in data_iterator:
            indices.append(i)
            targets.append(targets_dict["label"])

        data_length = len(indices)
        img_num_per_cls = self._get_img_num_per_cls(data_length)
        new_indices, _ = self._gen_imbalanced_data(img_num_per_cls, targets)
        return new_indices