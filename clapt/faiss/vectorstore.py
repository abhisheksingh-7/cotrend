from typing import Optional
import numpy as np
import faiss
import pydantic


class IndexConfig(pydantic.BaseModel):
    nlist: Optional[int] = 100
    nprobe: Optional[int] = 10


class FaissConfig(pydantic.BaseModel):
    dim: int
    gpu_id: int
    gpu_resource: faiss.StandardGpuResources = faiss.StandardGpuResources()
    index_config: Optional[IndexConfig] = None
    faster_search: bool = False
    lower_memory: bool = False
    quantization_factor: Optional[int] = 4


class VectorStore:
    def __init__(
        self,
        config: FaissConfig,
    ):
        self.dim = config.dim
        self.gpu_id = config.gpu_id
        self.gpu_resource = config.gpu_resource
        self.faster_search = config.faster_search
        self.lower_memory = config.lower_memory
        self.quantization_factor = config.quantization_factor
        if config.index_config:
            self.nlist = config.index_config.nlist
            self.nprobe = config.index_config.nprobe
        self.index = self.create_index()

    def create_index(self):
        # TODO: index_factory
        index = None
        if not self.faster_search:
            index = faiss.IndexFlatL2(self.dim)
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resource, self.gpu_id, index)
            return gpu_index

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

        gpu_index = faiss.index_cpu_to_gpu(self.gpu_resource, self.gpu_id, index)
        gpu_index.train(np.random.random((self.nlist, self.dim)).astype("float32"))

        return gpu_index

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query, k):
        return self.index.search(query, k)
