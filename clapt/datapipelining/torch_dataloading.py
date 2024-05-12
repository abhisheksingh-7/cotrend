from typing import Iterable
from torch.utils import data

from clapt import modeling
from clapt.datapipelining import (
    datamodels,
    padding,
    trainingsample_creation,
    wikipedia_loading,
)


class MocoDataset(data.Dataset):
    def __init__(
        self,
        wikislicer: wikipedia_loading.WikiSlicer,
        samplefactory: trainingsample_creation.TrainingSampleFactory,
    ) -> None:
        self.wikislicer = wikislicer
        self.samplefactory = samplefactory

    def __len__(self):
        return len(self.wikislicer)

    def __getitem__(self, idx: int) -> datamodels.TrainingDataPoint:
        return self.samplefactory(self.wikislicer[idx])

    @classmethod
    def from_model_name(cls, model_name: str = modeling.MODEL_NAME) -> "MocoDataset":
        return MocoDataset(
            wikipedia_loading.WikiSlicer(),
            trainingsample_creation.TrainingSampleFactory.from_model_name(model_name),
        )


def collate_batch(
    batch: Iterable[datamodels.TrainingDataPoint],
) -> datamodels.TrainingBatch:
    batch = list(batch)
    keys = padding.collate_samples(dp.key for dp in batch)
    queries = padding.collate_samples(dp.query for dp in batch)
    return datamodels.TrainingBatch(
        key_tensor=keys.input_ids,
        key_mask=keys.attention_mask,
        query_tensor=queries.input_ids,
        query_mask=queries.attention_mask,
    )


if __name__ == "__main__":
    dataset = MocoDataset.from_model_name()
    collate_batch(dataset[i] for i in range(32))
