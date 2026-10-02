import math, re
from collections import Counter
from Chunker import Chunker

class VectorIndex:
    def __init__(self, chunker: Chunker):
        self.chunker = chunker
        self.docs, self.vectors = [], []

    def add_document(self, doc):
        self.docs.append(doc)
        self.vectors.append(self.chunker.generate_embedding(doc["content"], input_type="document"))

    def search(self, query, k=1):
        q = self.chunker.generate_embedding(query, input_type="query")
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))
        scored = [(d, cos(q, v)) for d, v in zip(self.docs, self.vectors)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs, self.tfs, self.df = [], [], Counter()

    @staticmethod
    def _tok(text):
        return re.findall(r"\w+", text.lower())

    def add_document(self, doc):
        tokens = self._tok(doc["content"])
        self.docs.append(doc)
        self.tfs.append((Counter(tokens), len(tokens)))
        self.df.update(set(tokens))

    def search(self, query, k=1):
        n = len(self.docs)
        avgdl = sum(l for _, l in self.tfs) / max(n, 1)
        q = self._tok(query)
        scored = []
        for doc, (tf, dl) in zip(self.docs, self.tfs):
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / avgdl))
            scored.append((doc, s))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]