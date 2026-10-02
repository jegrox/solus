from typing import Any, Dict, List, Protocol, Tuple


class SearchIndex(Protocol):
    def add_document(self, document: Dict[str, Any]) -> None: ...

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]: ...


class Retriever:
    def __init__(self, *indexes: SearchIndex):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    def add_document(self, document: Dict[str, Any]):
        for index in self._indexes:
            index.add_document(document)

    def search(self, query_text: str, k: int = 1, k_rrf: int = 60):
        # Get results from all indexes
        # Over-fetch so documents ranked lower in one index can still surface
        all_results = [index.search(query_text, k=k * 5) for index in self._indexes]

        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                # Track document ranks across indexes
                # add_document passes the same dict to every index, so id() identifies it
                doc_id = id(doc)
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = {
                        "doc": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                doc_ranks[doc_id]["ranks"][idx] = rank + 1

        # Apply RRF scoring formula: score = sum(1 / (k_rrf + rank)) over indexes
        scored = [
            (
                entry["doc"],
                sum(1.0 / (k_rrf + r) for r in entry["ranks"] if r != float("inf")),
            )
            for entry in doc_ranks.values()
        ]

        # Return merged and sorted results
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
