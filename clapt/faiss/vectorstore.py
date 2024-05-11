import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim, index, gpu_id, use_ivf=False, nlist=100, nprobe=10):
        self.dim = dim
        self.gpu_id = gpu_id
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = self.create_index(index)

    def create_index(self, index):
        if self.use_ivf:
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        else:
            index = faiss.index_factory(self.dim, index)

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query, k):
        return self.index.search(query, k)
