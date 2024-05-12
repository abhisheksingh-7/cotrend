from typing import Iterable, NamedTuple
import torch

from clapt import modeling
from clapt.datapipelining import datamodels, validation


class LlamaLastHiddenStateExtractor:
    def __init__(
        self,
        model_name: str,
        decoder_device: torch.device = torch.device("cuda"),
        output_device: torch.device = torch.device("cpu"),
    ) -> None:
        self.decoder_device = decoder_device
        self.output_device = output_device
        self.model = modeling.CLAPT(model_name, num_layers=0).to(decoder_device)

    @validation.validate_batch
    def __call__(
        self, dp_batch: list[datamodels.TrainingDataPoint]
    ) -> datamodels.TrainingBatch:
        batch = collate_batch(dp_batch)
        key_embeds = self._get_decoder_embeds(
            batch.key_tensor.to(self.decoder_device),
            batch.key_mask.to(self.decoder_device),
        )
        query_embeds = self._get_decoder_embeds(
            batch.query_tensor.to(self.decoder_device),
            batch.query_mask.to(self.decoder_device),
        )
        return datamodels.TrainingBatch(
            key_tensor=key_embeds,
            key_mask=batch.key_mask,
            query_tensor=query_embeds,
            query_mask=batch.query_mask,
        ).to(self.output_device)

    def _get_decoder_embeds(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.model.get_decoder_last_hidden_state(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )


def collate_batch(
    batch: list[datamodels.TrainingDataPoint],
) -> datamodels.TrainingBatch:
    keys = _collate(dp.key for dp in batch)
    queries = _collate(dp.query for dp in batch)
    return datamodels.TrainingBatch(
        key_tensor=keys.input_ids,
        key_mask=keys.attention_mask,
        query_tensor=queries.input_ids,
        query_mask=queries.attention_mask,
    )


class MaskedInputs(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def _collate(tensors: Iterable[list[int]]) -> MaskedInputs:
    tensors = list(tensors)
    max_len = max(len(t) for t in tensors)
    attention_mask = torch.ones((len(tensors), max_len))
    for i, t in enumerate(tensors):
        attention_mask[i, len(t) :] = 0
        tensors[i] = _pad(t, max_len)
    return MaskedInputs(torch.tensor(tensors), attention_mask)


def _pad(input_ids: list[int], max_len: int) -> list[int]:
    pad_len = max_len - len(input_ids)
    return input_ids + [0] * pad_len


if __name__ == "__main__":
    import json

    from clapt.datapipelining import trainingsample_creation, wikipedia_loading

    MODEL_NAME = modeling.MODEL_NAME
    dataset = (
        wikipedia_loading.create_wikipedia_dataset()
        .map(
            trainingsample_creation.TrainingSampleFactory.from_model_name(MODEL_NAME),
            concurrency=1,
        )
        .map_batches(
            LlamaLastHiddenStateExtractor(MODEL_NAME),
            batch_size=4,
            concurrency=2,
            num_gpus=1,
        )
    )

    datapoint = dataset.take(1)
    print(json.dumps(datapoint, indent=2))
