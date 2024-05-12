from typing import List
import torch
import torch as T
import pydantic
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from clapt import modeling


class Document(pydantic.BaseModel):
    id_: int
    title: str
    abstract: str

    def __hash__(self) -> int:
        return hash((hash(self.id_), hash(self.text)))


class VectorStore(nn.Module):
    def __init__(
        self,
        embedding_model: modeling.CLAPT,
        tokenizer: AutoTokenizer,
        device: T.device,
    ) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self._index_to_doc = {}
        self._indx = 0
        self.embeddings = None
        self.tokenizer = tokenizer
        self.device = device

    def set_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self._index_to_doc[self._indx] = doc
            self._indx += 1
        texts = [doc.title for doc in documents]
        embeddings = []
        i = 0
        bs = 6
        with T.inference_mode():
            for b in batch(texts, batch_size=bs):
                print((bs * i) / len(texts))
                i += 1
                tokens = self.tokenizer(b, return_tensors="pt", padding="longest")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                output = self.embedding_model(**tokens)
                embeddings.append(output)
            self.embeddings = torch.cat(embeddings)
            return embeddings

    def search(self, text: str, k: int = 3) -> List[Document]:
        tokens = self.tokenizer(text, return_tensors="pt", padding="longest")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        embed = self.embedding_model(**tokens)
        scores = self.embeddings @ embed.T
        result = T.topk(scores, k=k, dim=0)
        docs = [self._index_to_doc[i[0]] for i in result.indices.tolist()]
        return docs, result.values.tolist()


def batch(lst, batch_size: int):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def main():
    PATH = "/data/clapt-head-epoch-3.pt"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    clapt = modeling.CLAPT(llm, num_layers=2)
    clapt.clapt_head.load_state_dict(torch.load(PATH))
