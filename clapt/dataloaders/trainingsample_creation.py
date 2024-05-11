"""
Data augmentations rewritten from:
https://github.com/facebookresearch/contriever/blob/main/src/data.py
"""

import enum
import random
from typing import NamedTuple, Optional, cast

import numpy as np
import pydantic
import transformers
from typing_extensions import Self

from clapt.dataloaders import datamodels, validation


class Augmentations(str, enum.Enum):
    REPLACE = "replace"
    DELETE = "delete"
    SHUFFLE = "shuffle"


class AugmentationConfig(pydantic.BaseModel):
    crop_ratio_min: float = 0.1
    crop_ratio_max: float = 0.3
    augmentation_probas: dict[Augmentations, float] = {
        Augmentations.MASK: 0.15,
        Augmentations.REPLACE: 0.1,
        Augmentations.DELETE: 0.1,
        Augmentations.SHUFFLE: 0.1,
    }


class KeyQueryTuple(NamedTuple):
    key: list[int]
    query: list[int]


class TrainingDatapointFactory:
    """Create a training data point from a document."""

    def __init__(
        self,
        config: AugmentationConfig,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id

    @validation.validate
    def __call__(self, dp: datamodels.Document) -> datamodels.TrainingDataPoint:
        input_ids = self.tokenize(dp.content)
        key, query = self.augment(input_ids)
        return datamodels.TrainingDataPoint(**dp.model_dump(), key=key, query=query)

    def tokenize(self, text: str) -> list[int]:
        batch_encoding = self.tokenizer.encode_plus(
            text,
            return_tensors=transformers.TensorType.NUMPY,
        )
        input_ids = cast(list[int], batch_encoding.input_ids[0])
        return input_ids

    def augment(self, tokens: list[int]) -> KeyQueryTuple:
        q_tokens = randomcrop(
            tokens, self.config.crop_ratio_min, self.config.crop_ratio_max
        )
        k_tokens = randomcrop(
            tokens, self.config.crop_ratio_min, self.config.crop_ratio_max
        )
        q_tokens = apply_augmentations(q_tokens, self.config, self.vocab_size)
        q_tokens = add_bos_eos(q_tokens, self.bos, self.eos)
        k_tokens = apply_augmentations(k_tokens, self.config, self.vocab_size)
        k_tokens = add_bos_eos(k_tokens, self.bos, self.eos)
        return KeyQueryTuple(key=k_tokens, query=q_tokens)

    @classmethod
    def from_model_name(cls, model_name: str, config: AugmentationConfig) -> Self:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        return cls(config, cast(transformers.PreTrainedTokenizer, tokenizer))


def apply_augmentations(
    x: list[int], opt: AugmentationConfig, vocab_size: int
) -> list[int]:
    if replace_proba := opt.augmentation_probas.get(Augmentations.REPLACE):
        x = replaceword(x, max_rand=vocab_size - 1, p=replace_proba)
    if delete_proba := opt.augmentation_probas.get(Augmentations.DELETE):
        x = deleteword(x, p=delete_proba)
    if shuffle_proba := opt.augmentation_probas.get(Augmentations.SHUFFLE):
        x = shuffleword(x, p=shuffle_proba)
    return x


def randomcrop(x: list[int], ratio_min: float, ratio_max: float) -> list[int]:
    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end]
    return crop


def deleteword(x: list[int], p: float) -> list[int]:
    mask = np.random.rand(len(x))
    new_x = [e for e, m in zip(x, mask) if m > p]
    return new_x


def replaceword(x: list[int], max_rand: int, p: float, min_rand: int = 0) -> list[int]:
    mask = np.random.rand(len(x))
    new_x = [
        e if m > p else random.randint(min_rand, max_rand) for e, m in zip(x, mask)
    ]
    return new_x


def maskword(x: list[int], mask_id: int, p: float) -> list[int]:
    mask = np.random.rand(len(x))
    new_x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return new_x


def shuffleword(x: list[int], p: float) -> list[int]:
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x


def add_bos_eos(
    x: list[int],
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> list[int]:
    if bos_token_id is not None:
        x = [bos_token_id] + x
    if eos_token_id is not None:
        x = x + [eos_token_id]
    return x


if __name__ == "__main__":
    from clapt.dataloaders import wikipedia_loading

    dpfactory = TrainingDatapointFactory.from_model_name(
        "meta-llama/Meta-Llama-3-8B", AugmentationConfig()
    )
    dataset = wikipedia_loading.create_wikipedia_dataset()
    datapoint = dataset.take(1)[0]
    processed_dp = dpfactory(datapoint)
    print(processed_dp)
