import enum
from typing_extensions import Annotated
import uuid
from typing import Optional

import pydantic
import torch

from clapt.datapipelining import validation


class Sources(str, enum.Enum):
    WIKIPEDIA = "wikipedia"


class Document(pydantic.BaseModel):
    source: Sources
    doc_id: uuid.UUID
    title: Optional[str]
    content: str


class TrainingDataPoint(Document):
    """One training data point for the model"""

    key: list[int]  # (seq_len)
    query: list[int]  # (seq_len)


class TrainingBatch(pydantic.BaseModel):
    """One batch of training data for the model"""

    key_tensor: Annotated[torch.Tensor, validation.TENSOR_VALIDATOR]
    key_mask: Annotated[torch.Tensor, validation.TENSOR_VALIDATOR]
    query_tensor: Annotated[torch.Tensor, validation.TENSOR_VALIDATOR]
    query_mask: Annotated[torch.Tensor, validation.TENSOR_VALIDATOR]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.field_serializer("key_tensor", "query_tensor", "key_mask", "query_mask")
    @classmethod
    def tensor_serializer(cls, v: torch.Tensor) -> torch.Tensor:
        return v

    def to(self, device: torch.device) -> "TrainingBatch":
        return TrainingBatch(
            key_tensor=self.key_tensor.to(device),
            key_mask=self.key_mask.to(device),
            query_tensor=self.query_tensor.to(device),
            query_mask=self.query_mask.to(device),
        )

    def __len__(self):
        return self.key_tensor.shape[0]
