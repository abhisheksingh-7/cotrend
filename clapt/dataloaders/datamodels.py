import enum
from typing import Optional
import pydantic
import uuid


class Sources(str, enum.Enum):
    WIKIPEDIA = "wikipedia"


class Document(pydantic.BaseModel):
    source: Sources
    doc_id: uuid.UUID
    title: Optional[str]
    content: str
