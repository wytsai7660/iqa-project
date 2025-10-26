from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import accumulate
from numpy import array, diff, ndarray, searchsorted, inner
from owl3.processing_mplugowl3 import mPLUGOwl3BatchFeature, mPLUGOwl3ImageProcessor, mPLUGOwl3Processor
from pandas import DataFrame, option_context, read_csv # pyright: ignore[reportUnknownVariableType]
from pathlib import PurePath
from PIL import Image
from random import choice, randint
from scipy.stats import norm # pyright: ignore[reportMissingTypeStubs]
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Qwen2Tokenizer # pyright: ignore[reportMissingTypeStubs]
from typing import Iterable, Literal, TypedDict, override

class PairDatasetImage(TypedDict):
    """
    The type of the values in a `PairDatasetItem` with the keys "image_1" and
    "image_2".
    """
    quality_message: mPLUGOwl3BatchFeature
    distortion_type_message: mPLUGOwl3BatchFeature | None
    scene_type_message: mPLUGOwl3BatchFeature | None
    level_probabilities: ndarray

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
    split: Literal["training", "validation", "testing"]
    use_scene_labels: bool
    use_distortion_labels: bool

class PairDataset(Dataset[PairDatasetItem]):
    def __init__(
        self,
        dataset_paths: Iterable[PurePath],
        processor: mPLUGOwl3Processor,
        tokenizer: Qwen2Tokenizer,
        args: PairDatasetArguments):
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
        self.tokenizer = tokenizer
        self.split = args["split"]
        self.use_scene_labels = args["use_scene_labels"]
        self.use_distortion_labels = args["use_distortion_labels"]
        all_dataset_labels_data_frames = [
            read_csv(path / "labels.csv", keep_default_na=False, index_col="filename") # keep_default_na=False makes read_csv treat empty scene types as empty strings
            for path in dataset_paths
        ]
        """
        A `list` of `DataFrame`s where each element stores the `labels.csv` of
        one dataset. We will use this and select only the image samples that
        belong in the set specified by the split parameter.
        """
        self.dataset_labels_data_frames: list[DataFrame] = []
        for data_frame in all_dataset_labels_data_frames:
            # We always use loc here to avoid pandas' SettingWithCopyWarning.
            data_frame.loc[:, :] = data_frame.loc[data_frame["set"] == self.split]
            min_mos = data_frame["mos"].min()
            max_mos = data_frame["mos"].max()
            # Maps the MOS to [0, 1] first, then * 4 + 1 transforms it into [1,
            # 5].
            data_frame.loc[:, "mos_normalized"] = (data_frame["mos"] - min_mos) / (max_mos - min_mos) * 4 + 1
            data_frame.loc[:, "stddev_normalized"] = data_frame["stddev"] / (max_mos - min_mos) * 4 # Only multiplications affect the standard deviation
            self.dataset_labels_data_frames.append(data_frame.loc[data_frame["set"] == self.split])
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

    @staticmethod
    def get_level_probabilities(mos: float, stddev: float) -> ndarray:
        cdf_points = array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        cdf_values = norm.cdf(cdf_points, loc=mos, scale=stddev) # pyright: ignore[reportUnknownMemberType]
        p_raw = diff(cdf_values)
        # Taken from 4th page of DeQA-Score, Post-adjustment section
        p_raw_sum = p_raw.sum()
        mu_rec = inner(array([1, 2, 3, 4, 5]), p_raw)
        alpha = (mos - 3) / (mu_rec - 3 * p_raw_sum + 1e-9)
        beta = (1 - alpha * p_raw_sum) / 5
        return p_raw * alpha + beta

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
        mos = self.dataset_labels_data_frames[dataset_index].iloc[image_index]["mos_normalized"]
        stddev = self.dataset_labels_data_frames[dataset_index].iloc[image_index]["stddev_normalized"]
        level_probabilities = array(self.get_level_probabilities(mos, stddev))
        level_names = ["bad", "low", "fair", "good", "awesome"]
        scene_type = self.dataset_labels_data_frames[dataset_index].iloc[image_index]["scene1"]
        distortion_type = self.dataset_labels_data_frames[dataset_index].iloc[image_index]["distortion"]
        image_prelude = [{
            "role": "user",
            "content": "<|image|>\n"
        }]
        scene_type_message = [
            {
                "role": "user",
                "content": "What is the scene type of this image?"
            },
            {
                "role": "assistant",
                "content": f"The scene type of this image is {scene_type}."
            }
        ]
        distortion_type_message = [
            {
                "role": "user",
                "content": "What is the distortion type of this image?"
            },
            {
                "role": "assistant",
                "content": f"The distortion type of this image is {distortion_type}."
            }
        ]
        quality_message = [
            {
                "role": "user",
                "content": f"{choice(possible_quality_questions)}"
            },
            {
                "role": "assistant",
                "content": f"This quality of this image is {level_names[level_probabilities.argmax()]}."
            }
        ]
        if self.use_scene_labels and self.use_distortion_labels:
            quality_message = scene_type_message + distortion_type_message + quality_message
        elif self.use_distortion_labels:
            quality_message = distortion_type_message + quality_message
        elif self.use_scene_labels:
            quality_message = scene_type_message + quality_message
        if self.use_distortion_labels:
            distortion_type_message = scene_type_message + distortion_type_message
        # self.use_scene_labels and self.use_distortion_labels => quality_message = image_prelude + scene_type_message + distortion_type_message + quality_message
        # not self.use_scene_labels and self.use_distortion_labels => quality_message = image_prelude + distortion_type_message + quality_message
        # self.use_scene_labels and not self.use_distortion_labels => quality_message = image_prelude + scene_type_message + quality_message
        # not self.use_scene_labels and not self.use_distortion_labels => quality_message = image_prelude + quality_message
        image_path: str = self.dataset_labels_data_frames[dataset_index].index[image_index]
        image = Image.open(self.dataset_paths[dataset_index] / "images" / image_path).convert("RGB")
        return {
            "quality_message": self.processor(images=[image], messages=deepcopy(image_prelude) + quality_message),
            "distortion_type_message": self.processor(images=[image], messages=deepcopy(image_prelude) + distortion_type_message) if self.use_distortion_labels else None,
            "scene_type_message": self.processor(images=[image], messages=image_prelude + scene_type_message) if self.use_scene_labels else None,
            "level_probabilities": level_probabilities,
        }
    
    def __len__(self) -> int:
        return self.cumulative_dataset_image_counts[-1]


def collate_fn(batch):
    pass


if __name__ == "__main__":
    result = PairDataset.get_level_probabilities(2.5, 1.0)
    print(result)
    print(sum(result))
    print(type(result))
    MODEL_DIR = "./src/owl3"
    plt.bar(range(len(result)), result)
    plt.xlabel("Level")
    plt.ylabel("Probability")
    plt.title("Level Probabilities Distribution")
    plt.xticks(range(len(result)), ["0", "1", "2", "3", "4"])
    plt.savefig("level_probabilities_distribution.png")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # Taken from mPLUGOwl3Model.init_processor
    image_processor = mPLUGOwl3ImageProcessor(image_size=378)
    processor = mPLUGOwl3Processor(image_processor=image_processor, tokenizer=tokenizer)
    dataset = PairDataset(
        dataset_paths=[PurePath("datasets/live")],
        processor=processor,
        tokenizer=tokenizer,
        args={
            "split": "training",
            "use_distortion_labels": True,
            "use_scene_labels": True
        }
    )
    print(dataset[0])
    print(f"Sum of image_1's level_probabilities: {sum(dataset[0]["image_1"]["level_probabilities"])}")
    print(f"Sum of image_2's level_probabilities: {sum(dataset[0]["image_2"]["level_probabilities"])}")
    with option_context("display.max_rows", None, "display.max_columns", None, "display.width", None):
        for data_frame in dataset.dataset_labels_data_frames:
            print(data_frame)
