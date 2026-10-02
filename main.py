from pathlib import Path
import anthropic
from Chunker import Chunker
from Retriever import Retriever
from indexes import VectorIndex, BM25Index

chunker = Chunker()
retriever = Retriever(VectorIndex(chunker), BM25Index())

# Ingest: one dict per chunk, with metadata for citations
for path in Path("data").glob("**/*.md"):
    text = path.read_text(encoding="utf-8")
    for i, chunk in enumerate(Chunker.chunk_by_sentence(text, 5, 1)):
        if chunk.strip():
            retriever.add_document({"content": chunk, "source": str(path), "chunk": i})

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

def answer(question, k=5):
    hits = retriever.search(question, k=k)
    context = "\n\n".join(
        f'<doc source="{d["source"]}#{d["chunk"]}">\n{d["content"]}\n</doc>' for d, _ in hits
    )
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        # betas=["server-side-fallback-2026-07-01"],
        # fallbacks="default",  # retries on another model if a safety filter declines
        system="Answer using only the provided documents. Cite sources. "
               "If the documents don't contain the answer, say so.",
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    if response.stop_reason == "refusal":
        return "The request was declined."
    return "".join(b.text for b in response.content if b.type == "text")

if __name__ == "__main__":
    while (q := input("\nAsk: ").strip()):
        print(answer(q))