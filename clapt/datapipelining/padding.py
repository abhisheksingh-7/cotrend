from typing import Iterable, NamedTuple

import torch

from clapt.datapipelining import datamodels


class MaskedInputs(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return self.input_ids.shape

    @classmethod
    def from_keys(cls, batch: datamodels.TrainingBatch) -> "MaskedInputs":
        return cls(batch.key_tensor, batch.key_mask)

    @classmethod
    def from_queries(cls, batch: datamodels.TrainingBatch) -> "MaskedInputs":
        return cls(batch.query_tensor, batch.query_mask)


def collate_batches(
    batches: Iterable[MaskedInputs],  # (seq_len, emb_dim)
) -> MaskedInputs:  # (batch_size, max_seq_len, emb_dim)
    batches = list(batches)
    max_len = max(batch.shape[0] for batch in batches)
    padded_batches: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    for batch in batches:
        padding = torch.zeros(max_len - batch.shape[0], batch.shape[1])
        padded_batches.append(torch.concat([batch.input_ids, padding]))
        attention_masks.append(torch.concat([batch.attention_mask, padding[:, 0]]))
    return MaskedInputs(torch.stack(padded_batches), torch.stack(attention_masks))


def collate_samples(tensors: Iterable[list[int]]) -> MaskedInputs:
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
