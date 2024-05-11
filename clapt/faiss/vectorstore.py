import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim, index, gpu_id):
        self.dim = dim
        self.gpu_id = gpu_id
        self.index = self.create_index(index)

    def create_index(self, index): ...

    def add(self, vectors):
        self.index.add(vectors)

    def search(self, query, k):
        return self.index.search(query, k)
