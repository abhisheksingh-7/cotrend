import enum
from typing import Optional, Tuple, Union
import numpy as np
import faiss
import pydantic
import torch as T
import loguru

logger = loguru.logger


class IVFIndexParams(pydantic.BaseModel):
    nlist: Optional[int] = 100
    nprobe: Optional[int] = 10
    quantization_factor: Optional[int] = 4


class HNSWIndexParams(pydantic.BaseModel):
    M: Optional[int] = 8  # A larger M is more accurate but uses more memory
    efConstruction: Optional[int] = 64  # depth of exploration at add time
    efSearch: Optional[int] = 20  # depth of exploration of the search


class IndexType(enum.Enum):
    IVFFLAT = faiss.IndexIVFFlat
    IVFPQ = faiss.IndexIVFPQ
    HNSW = faiss.IndexHNSW
    FLATIP = faiss.IndexFlatIP
    FLATL2 = faiss.IndexFlatL2


class IndexConfig(pydantic.BaseModel):
    index_type: Optional[IndexType] = IndexType.FLATL2
    index_params: Optional[Union[IVFIndexParams, HNSWIndexParams]] = None


class FaissConfig(pydantic.BaseModel):
    dim: int
    gpu_id: int
    index_config: Optional[IndexConfig] = None
    metric: int = faiss.METRIC_L2


class VectorStore:
    def __init__(
        self,
        config: FaissConfig,
    ) -> None:
        self.dim = config.dim
        self.gpu_id = config.gpu_id
        self.gpu_resource = faiss.StandardGpuResources()
        self.metric = config.metric
        if config.index_config:
            self.index_type = config.index_config.index_type
            self.index_params = config.index_config.index_params
        self.index: faiss.Index = self.create_index()

    def create_index(self) -> faiss.Index:
        # TODO: index_factory
        index = None

        if self.index_type == IndexType.HNSW:
            index: faiss.IndexHNSWFlat = faiss.IndexHNSWFlat(
                self.dim, self.index_params.M, self.metric
            )
            index.hnsw.efConstruction = self.index_params.efConstruction
            index.hnsw.efSearch = self.index_params.efSearch

        elif self.index_type in [IndexType.FLATIP, IndexType.FLATL2]:
            index = faiss.IndexFlatIP(self.dim)

        else:
            quantizer = faiss.IndexFlatL2(self.dim)
            if self.index_type == IndexType.IVFFLAT:
                index = faiss.IndexIVFFlat(
                    quantizer, self.dim, self.index_params.nlist, self.metric
                )
            elif self.index_type == IndexType.IVFPQ:
                index = faiss.IndexIVFPQ(
                    quantizer,
                    self.dim,
                    self.index_params.nlist,
                    self.dim // self.index_params.quantization_factor,
                    8,
                )
            index.train(np.random.rand(10000, self.dim).astype("float32"))

        gpu_index = faiss.index_cpu_to_gpu(self.gpu_resource, self.gpu_id, index)
        return gpu_index

    def add(self, vectors: Union[np.ndarray, T.Tensor]) -> None:
        if isinstance(vectors, T.Tensor):
            vectors = vectors.cpu().numpy().astype("float32")
        assert vectors.dtype == np.float32, "vectors must be float32 type"
        assert (
            vectors.shape[1] == self.dim
        ), "Vectors must have the same dimension as the index"
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(
        self, query: Union[np.ndarray, T.Tensor], k: int
    ) -> Tuple[np.ndarray, np.ndarray]:

        if isinstance(query, T.Tensor):
            query = query.cpu().numpy().astype("float32")

        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(query)
        return self.index.search(query, k)


def main() -> None:
    index_config = IndexConfig(
        index_type=IndexType.IVFFLAT, index_params=IVFIndexParams()
    )

    faiss_config = FaissConfig(
        dim=128,
        gpu_id=0,
        metric=faiss.METRIC_L2,
        index_config=index_config,
        arbitrary_types_allowed=True,
    )

    vs = VectorStore(faiss_config)

    nb_vectors = 1000
    k = 4

    db_vectors = T.rand(nb_vectors, faiss_config.dim).cuda()
    vs.add(db_vectors)

    query_vector = T.rand(1, faiss_config.dim).cuda()
    logger.info(f"Query Vector: {query_vector}")
    D_1, I_1 = vs.search(query_vector, k)

    logger.info(f"L2 Distances: {D_1}")
    logger.info(f"Indices: {I_1}")


if __name__ == "__main__":
    main()
