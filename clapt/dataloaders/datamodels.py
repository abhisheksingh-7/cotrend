import enum
import uuid
from typing import Optional

import numpy as np
import pydantic
import torch


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

    ## cnf
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.field_serializer("key", "query")
    def dont_serialize(self, v: torch.Tensor) -> torch.Tensor:
        return v
