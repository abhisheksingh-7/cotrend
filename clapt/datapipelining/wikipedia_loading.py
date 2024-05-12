import os
import sqlite3
import uuid
from typing import Union

import pydantic
import ray.data
from typing_extensions import Annotated

from clapt import constants
from clapt.datapipelining import datamodels, validation

DEFAULT_DB_PATH = constants.REPO_ROOT / "data/wikipedia/docs.db"
QUERY = """SELECT * FROM documents LIMIT 100;"""


def create_wikipedia_dataset(
    path_to_wikipedia_db: Union[str, os.PathLike] = DEFAULT_DB_PATH,
) -> Annotated[ray.data.Dataset, datamodels.Document]:
    raw_dataset = _create_raw_wikipedia_dataset(path_to_wikipedia_db)
    return raw_dataset.map(parse_raw_wikipedia_data)


class RawWikipediaSchema(pydantic.BaseModel):
    """Schema for raw Wikipedia data as from the sql query."""

    id: str  # header e.g. Brazil
    text: str  # content e.g. Brazil\n\nBrazil (; ), officially the Federative Republic ...


@validation.validate
def parse_raw_wikipedia_data(raw_data: RawWikipediaSchema) -> datamodels.Document:
    return datamodels.Document(
        source=datamodels.Sources.WIKIPEDIA,
        doc_id=uuid.uuid4(),
        title=raw_data.id,
        content=raw_data.text,
    )


def _create_raw_wikipedia_dataset(
    path_to_wikipedia_db: Union[str, os.PathLike] = DEFAULT_DB_PATH,
) -> Annotated[ray.data.Dataset, RawWikipediaSchema]:
    def create_connection() -> sqlite3.Connection:
        return sqlite3.connect(path_to_wikipedia_db)

    return ray.data.read_sql(QUERY, create_connection)


if __name__ == "__main__":
    dataset = _create_raw_wikipedia_dataset()
    dataset.take(5)
