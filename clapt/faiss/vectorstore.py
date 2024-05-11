from typing import Optional, Tuple, Union
import numpy as np
import faiss
import pydantic
import torch as T
import loguru

logger = loguru.logger


class IVFIndexConfig(pydantic.BaseModel):
    nlist: Optional[int] = 100
    nprobe: Optional[int] = 10


class FaissConfig(pydantic.BaseModel):
    dim: int
    gpu_id: int
    ivf_index_config: Optional[IVFIndexConfig] = None
    faster_search: bool = False
    lower_memory: bool = False
    quantization_factor: Optional[int] = 4


class VectorStore:
    def __init__(
        self,
        config: FaissConfig,
    ) -> None:
        self.dim = config.dim
        self.gpu_id = config.gpu_id
        self.gpu_resource = faiss.StandardGpuResources()
        self.faster_search = config.faster_search
        self.lower_memory = config.lower_memory
        self.quantization_factor = config.quantization_factor
        if config.ivf_index_config:
            self.nlist = config.ivf_index_config.nlist
            self.nprobe = config.ivf_index_config.nprobe
        self.index: faiss.Index = self.create_index()

    def create_index(self) -> faiss.Index:
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

        gpu_index: faiss.Index = faiss.index_cpu_to_gpu(
            self.gpu_resource, self.gpu_id, index
        )
        gpu_index.train(np.random.random((self.nlist, self.dim)).astype("float32"))

        return gpu_index

    def add(self, vectors: Union[np.ndarray, T.Tensor]) -> None:
        if isinstance(vectors, T.Tensor):
            vectors = vectors.cpu().numpy().astype("float32")
        assert vectors.dtype == np.float32, "vectors must be float32 type"
        assert (
            vectors.shape[1] == self.dim
        ), "Vectors must have the same dimension as the index"
        self.index.add(vectors)

    def search(
        self, query: Union[np.ndarray, T.Tensor], k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(query, T.Tensor):
            query = query.cpu().numpy().astype("float32")
        return self.index.search(query, k)


def main() -> None:
    faiss_config = FaissConfig(dim=128, gpu_id=0)
    vs = VectorStore(faiss_config)

    nb_vectors = 1000
    k = 4

    db_vectors = T.rand(nb_vectors, faiss_config.dim).cuda()
    vs.add(db_vectors)

    query_vector = T.rand(1, faiss_config.dim).cuda()
    D, I = vs.search(query_vector, k)

    logger.info(f"Distances: {D}")
    logger.info(f"Indices: {I}")


if __name__ == "__main__":
    main()
