from typing import Optional
import numpy as np
import faiss
import pydantic


class IndexConfig(pydantic.BaseModel):
    nlist: Optional[int] = 100
    nprobe: Optional[int] = 10


class FaissConfig(pydantic.BaseModel):
    dim: int
    index_config: Optional[IndexConfig] = None
    faster_search: bool = False
    lower_memory: bool = False
    quantization_factor: Optional[int] = 4


class VectorStore:
    def __init__(
        self,
        config: FaissConfig,
        gpu_id: int,
    ):
        self.dim = config.dim
        self.gpu_id = gpu_id
        self.faster_search = config.faster_search
        self.lower_memory = config.lower_memory
        self.quantization_factor = config.quantization_factor
        if config.index_config:
            self.nlist = config.index_config.nlist
            self.nprobe = config.index_config.nprobe
        self.index = self.create_index()

    def create_index(self):
        index = None
        if not self.faster_search:
            return faiss.IndexFlatL2(self.dim)

        quantizer = faiss.IndexFlatL2(self.dim)
        if self.lower_memory:
            index = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.nlist,
                self.dim // self.quantization_factor,
                8,
            )
        else:
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        assert not index.is_trained, "Index is already trained"
        index.train(np.random.random((self.nlist, self.dim)).astype("float32"))

        return index

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query, k):
        return self.index.search(query, k)
