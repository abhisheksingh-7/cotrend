from typing import Iterable, NamedTuple
from torch.utils import data
import torch as T

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


class SapBertRow(NamedTuple):
    id_: int
    term1: str
    term2: str


class SapBertDataset(data.Dataset):
    def __init__(self, path_to_data: str) -> None:
        self.path_to_data = path_to_data
        with open(self.path_to_data, "r") as f:
            lines = f.readlines()
        self.rows = []
        self.query_ids = set()
        for line in lines:
            line = line.rstrip("\n")
            if line.count("||") != 2:
                continue
            row = SapBertRow(*line.split("||"))
            self.rows.append(row)
            self.query_ids.add(row.id_)
        self.query_id_2_index_id = {
            k: v for v, k in enumerate(list(set(self.query_ids)))
        }

    def __len__(self) -> None:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return row.term1, row.term2, self.query_id_2_index_id[row.id_]


def collate_fn_batch_encoding(batch, tokenizer):
    query1, query2, query_id = zip(*batch)
    query_encodings1 = tokenizer.batch_encode_plus(
        list(query1),
        max_length=50,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    query_encodings2 = tokenizer.batch_encode_plus(
        list(query2),
        max_length=50,
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    query_ids = T.tensor(list(query_id))
    return query_encodings1, query_encodings2, query_ids


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
