from typing import Optional
import numpy as np
import faiss


class VectorStore:
    def __init__(
        self,
        dim: int,
        index: Optional[faiss.Index],
        gpu_id: int,
        faster_search: bool = False,
        lower_memory: bool = False,
        nlist: int = 100,
        nprobe: int = 10,
        quantization_factor: int = 4,
    ):
        self.dim = dim
        self.gpu_id = gpu_id
        self.use_ivf = faster_search
        self.lower_memory = lower_memory
        self.nlist = nlist
        self.nprobe = nprobe
        self.quantization_factor = quantization_factor
        self.index = self.create_index(index)

    def create_index(self, index):
        if self.faster_search:
            quantizer = faiss.IndexFlatL2(self.dim)
            if self.lower_memory:
                index = faiss.IndexIVFPQ(
                    quantizer,
                    self.dim,
                    self.nlist,
                    self.dim / self.quantization_factor,
                    8,
                )
            else:
                index = faiss.IndexIVFFlat(
                    quantizer, self.dim, self.nlist, faiss.METRIC_L2
                )
            assert not index.is_trained
            index.train(np.random.random((self.nlist, self.dim)).astype("float32"))

        else:
            if not index:
                index = faiss.IndexFlatL2(self.dim)
            index = faiss.index_factory(self.dim, index)

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query, k):
        return self.index.search(query, k)
