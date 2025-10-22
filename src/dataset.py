from itertools import accumulate
from numpy import searchsorted
from transformers import Qwen2Tokenizer # pyright: ignore[reportMissingTypeStubs]
from owl3.processing_mplugowl3 import mPLUGOwl3BatchFeature, mPLUGOwl3Processor
from pandas import read_csv # pyright: ignore[reportUnknownVariableType]
from pathlib import PurePath
from random import choice, randint
from torch.utils.data import Dataset
from typing import Any, Iterable, TypedDict, override
from PIL import Image

class PairDatasetImage(TypedDict):
    """
    The type of the values in a `PairDatasetItem` with the keys "image_1" and
    "image_2".
    """
    quality_message: mPLUGOwl3BatchFeature
    distortion_type_message: mPLUGOwl3BatchFeature
    scene_type_message: mPLUGOwl3BatchFeature
    level_probabilities: list[float]

class PairDatasetItem(TypedDict):
    """
    The return `dict` of `PairDataset.__getitem__`.
    """
    image_1: PairDatasetImage
    image_2: PairDatasetImage

class PairDatasetArguments(TypedDict):
    """
    A `dict` that you can pass to `PairDataset` to customize its behavior.
    """

class PairDataset(Dataset[PairDatasetItem]):
    def __init__(
        self,
        dataset_paths: Iterable[PurePath],
        processor: mPLUGOwl3Processor,
        tokenizer: Qwen2Tokenizer,
        args: PairDatasetArguments | None = None):
        """
        :param processor: The `mPLUGOwl3Processor` returned by
            `mPLUGOwl3Model.init_processor`.
        :param datasets_path: The paths to the dataset directories to use. Each
            dataset directory should contain a `labels.csv` file. For exactly
            how to structure the datasets, please see
            https://huggingface.co/datasets/palapapa/iqa-project-dataset.
        :param processor: This is needed to produce the `mPLUGOwl3BatchFeature`
            needed in `PairDatasetImage`.
        :param tokenizer: This is needed according to the
            [example](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-241101) given
            by mPLUG-Owl3. This is needed to call `update` on the return value
            of `mPLUGOwl3Processor`.
        :param args: A `PairDatasetArguments` to customize the behavior of this
            `Dataset`.
        """
        super().__init__()
        self.dataset_paths = list(dataset_paths)
        self.processor = processor
        self.dataset_labels_data_frames = [
            read_csv(path / "labels.csv", keep_default_na=False, index_col="filename") # keep_default_na=False makes read_csv treat empty scene types as empty strings
            for path in dataset_paths
        ]
        """
        A `list` of `DataFrame`s where each element stores the `labels.csv` of
        one dataset.
        """
        self.dataset_image_counts = [len(data_frame.index) for data_frame in self.dataset_labels_data_frames]
        if any(dataset_image_count < 2 for dataset_image_count in self.dataset_image_counts):
            raise ValueError("Every dataset must have at least 2 images in it because image pairs can't be selected from a dataset with only 1 image.")
        self.cumulative_dataset_image_counts = list(accumulate(self.dataset_image_counts))
        """
        If there are 5 datasets and they contain 1, 2, 3, 4, and 5 images
        respectively, then this is [1, 3, 6, 10, 15]. This is useful for
        determining from which dataset to return image pairs, because a pair of
        images must be in the same dataset for their scores to be comparable.
        """

    def get_dataset_and_image_indices(self, getitem_index: int) -> tuple[int, int]:
        """
        Used in `__getitem__` to determine which dataset to get images from and
        which image in it to return. `getitem_index` must already be checked to
        not be out of bounds.

        :param getitem_index: The index argument to `__getitem__`.

        :returns: `(dataset_index, image_index)`
        """
        dataset_index = searchsorted(self.cumulative_dataset_image_counts, getitem_index, "right")
        if dataset_index == 0:
            return dataset_index.item(), getitem_index
        return dataset_index.item(), getitem_index - self.cumulative_dataset_image_counts[dataset_index - 1]

    @override
    def __getitem__(self, index: int) -> PairDatasetItem:
        if index < 0 or index >= len(self):
            raise IndexError("The index to PairDataset is out of bounds.")
        dataset_index, image_1_index = self.get_dataset_and_image_indices(index)
        # Gets the index of the second image in the same dataset randomly.
        while True:
            image_2_index = randint(0, self.dataset_image_counts[dataset_index] - 1)
            if image_2_index != image_1_index:
                break
        return PairDatasetItem(
            image_1=self.get_one_image(dataset_index, image_1_index),
            image_2=self.get_one_image(dataset_index, image_2_index)
        )
    
    def get_level_probabilities(self, mos: float, stddev: float) -> list[float]:


    def get_one_image(self, dataset_index: int, image_index: int) -> PairDatasetImage:
        possible_quality_questions = [
            "What do you think about the quality of this image?",
            "Can you rate the quality of this picture?",
            "Can you judge the quality of this image?",
            "How would you rate the quality of this image?",
            "How would you judge the quality of this image?",
            "What is your quality rating for this image?",
            "What's your opinion on the quality of this picture?",
            "Rate the quality of this image.",
            "Could you evaluate the quality of this image?",
            "How do you assess the quality of this image?"
        ]
        quality_message = [
            {
                "role": "user",
                "content": f"<|image|>\n{choice(possible_quality_questions)}"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
        distortion_type_message = [
            {
                "role": "user",
                "content": f"<|image|>\nWhat is the distortion type of this image?"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
        scene_type_message = [
            {
                "role": "user",
                "content": f"<|image|>\nWhat is the scene type of this image?"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
        image_path: str = self.dataset_labels_data_frames[dataset_index].iloc[image_index]["filename"]
        image = Image.open(self.dataset_paths[dataset_index] / "images" / image_path)
        mos = self.dataset_labels_data_frames[dataset_index].iloc[image_index]["mos"]
        stddev = self.dataset_labels_data_frames[dataset_index].iloc[image_index]["stddev"]
        level_probabilities = self.get_level_probabilities()
    
    def __len__(self) -> int:
        return self.cumulative_dataset_image_counts[-1]
