import math, re
import os
import pickle
from collections import Counter
from Chunker import Chunker
import chromadb
import uuid

from chromadb import (
    Schema,
    SparseVectorIndexConfig,
    K
)

from rank_bm25 import BM25Okapi

class VectorIndex:
    def __init__(self, chunker: Chunker):
        self.chunker = chunker
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="vector_collection")
        # self.docs, self.vectors = [], []

    def is_built(self):
        # Chroma persists to ./chroma_db, so a non-empty collection means the
        # embeddings already exist and voyageai doesn't need to be called again
        return self.collection.count() > 0

    def reset(self):
        # Drop the collection (and its stored embeddings) and start empty
        self.client.delete_collection(name="vector_collection")
        self.collection = self.client.get_or_create_collection(name="vector_collection")

    def add_documents(self, docs):
        self.collection.add(
            # Deterministic ids so the same chunk can't be stored twice
            ids=[f'{doc["source"]}#{doc["chunk"]}' for doc in docs],
            embeddings=[self.chunker.generate_embedding(doc["content"], input_type="document") for doc in docs],
            documents=[doc["content"] for doc in docs],
            metadatas=[{"source": doc["source"], "chunk": doc["chunk"]} for doc in docs]
        )

    def search(self, query, k=1):
        q = self.chunker.generate_embedding(query, input_type="query")
        res = self.collection.query(
            query_texts=[query],
            query_embeddings=[q],
            n_results=k
        )
        return [
            ({"content": text, "source": meta["source"], "chunk": meta["chunk"]}, 1 - dist)
            for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

class BM25Index:
    def __init__(self, k1=1.5, b=0.75, path="./bm25_index.pkl"):
        self.k1, self.b = k1, b
        self.path = path
        self.docs, self.tfs, self.df = [], [], Counter()
        self.bm25 = None
        self._load()

    @staticmethod
    def _tok(text):
        return re.findall(r"\w+", text.lower())

    def is_built(self):
        return self.bm25 is not None

    def reset(self):
        self.docs, self.bm25 = [], None
        if os.path.exists(self.path):
            os.remove(self.path)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                state = pickle.load(f)
            self.docs, self.bm25 = state["docs"], state["bm25"]

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump({"docs": self.docs, "bm25": self.bm25}, f)

    def add_documents(self, docs):
        # BM25Okapi can't be updated incrementally, so append and rebuild over everything
        self.docs.extend({"content": d["content"], "source": d["source"], "chunk": d["chunk"]} for d in docs)
        self.bm25 = BM25Okapi([self._tok(d["content"]) for d in self.docs], k1=self.k1, b=self.b)
        self._save()

    def search(self, query, k=1):
        tokenized_query = self._tok(query)
        scores = self.bm25.get_scores(tokenized_query)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]