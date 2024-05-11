import enum
import uuid
from typing import Optional

import pydantic


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
