import os
import sqlite3
import unicodedata
import uuid
from typing import Union, Optional

import pydantic
import ray.data
from typing_extensions import Annotated

from clapt import constants
from clapt.datapipelining import datamodels, validation

DEFAULT_DB_PATH = (
    constants.REPO_ROOT / "data/wikipedia/docs.db"
    if os.uname().nodename == "koozie-00"
    else "/nfs/scratch/data/wiki-docs.db"
)
QUERY = """SELECT * FROM documents"""
SAMPLE_QUERY = """SELECT * FROM documents
                LIMIT {k}"""


class WikiSlicer:
    def __init__(
        self,
        path_to_wikipedia_db: Union[str, os.PathLike] = DEFAULT_DB_PATH,
    ) -> None:
        self.conn = sqlite3.connect(path_to_wikipedia_db)
        self._keys = self._load_keys()

    def __getitem__(self, idx: int) -> datamodels.Document:
        cursor = self.conn.cursor()
        id_ = self._keys[idx]
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (unicodedata.normalize("NFD", id_),),
        )
        unparsed_result, *_ = cursor.fetchone()
        result = datamodels.Document(
            content=unparsed_result,
            doc_id=uuid.uuid4(),
            source=datamodels.Sources.WIKIPEDIA,
            title=id_,
        )
        cursor.close()
        return result

    def __len__(self) -> int:
        return len(self._keys)

    def _load_keys(self) -> list[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents")
        results = cursor.fetchall()
        cursor.close()
        return [k for (k,) in results]


def create_wikipedia_dataset(
    path_to_wikipedia_db: Union[str, os.PathLike] = DEFAULT_DB_PATH,
    load_k_rows: Optional[int] = None,
) -> Annotated[ray.data.Dataset, datamodels.Document]:
    raw_dataset = _create_raw_wikipedia_dataset(path_to_wikipedia_db, load_k_rows)
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
    load_k_rows: Optional[int] = None,
) -> Annotated[ray.data.Dataset, RawWikipediaSchema]:
    def create_connection() -> sqlite3.Connection:
        return sqlite3.connect(path_to_wikipedia_db)

    query = SAMPLE_QUERY.format(k=load_k_rows) if load_k_rows is not None else QUERY
    return ray.data.read_sql(query, create_connection)


if __name__ == "__main__":
    a = WikiSlicer()[2]
    dataset = _create_raw_wikipedia_dataset()
    dataset.take(5)
