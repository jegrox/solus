import csv
import re
import sys
from pathlib import Path
import anthropic
import pymupdf4llm
from Agent import Agent
from Chunker import Chunker
from Retriever import Retriever
from indexes import VectorIndex, BM25Index

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval


judge = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

chunker = Chunker()
retriever = Retriever(VectorIndex(chunker), BM25Index())
agent = Agent(retriever)

def load_markdown(path):
    """Return a file's contents as markdown, converting PDFs with pymupdf4llm."""
    if path.suffix.lower() == ".pdf":
        return pymupdf4llm.to_markdown(str(path))
    return path.read_text(encoding="utf-8")

def build_index(force=False):
    # Ingest: one dict per chunk, with metadata for citations.
    # Indexes already on disk (chroma_db/, bm25_index.pkl) are left alone
    # unless force is set, which wipes them and rebuilds from scratch.
    if force:
        retriever.reset()

    chunks = []
    for path in Path("data").glob("**/*"):
        if path.suffix.lower() not in {".md", ".pdf"}:
            continue
        text = load_markdown(path)
        for i, chunk in enumerate(chunker.chunk_by_section(text)):
            if chunk.strip():
                chunks.append({"content": chunk, "source": str(path), "chunk": i})

    retriever.build(chunks)

def grade(question, expected, actual):
    """Ask Claude to score how closely `actual` matches `expected`, 0-100."""
    response = judge.messages.create(
        model="claude-opus-5-5",
        max_tokens=16,
        system="You grade answers against a reference answer. Score 0-100 for how well "
               "the candidate covers the facts in the reference: 100 = all key facts present "
               "and correct, 0 = all facts wrong or missing. Ignore wording, citations and extra detail "
               "unless it contradicts the reference. Reply with only the integer score.",
        messages=[{"role": "user", "content":
                   f"<question>{question}</question>\n"
                   f"<reference>{expected}</reference>\n"
                   f"<candidate>{actual}</candidate>"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    match = re.search(r"\d+", text)
    return min(100, int(match.group())) if match else 0


def get_test_cases(path="eval-set.txt"):
    # Each line is a CSV row: "question", "answer"
    with open(path, encoding="utf-8") as f:
        pairs = [row for row in csv.reader(f, skipinitialspace=True) if row]

    # scores = []
    test_cases = []
    for i, (question, expected) in enumerate(pairs, 1):
        actual = agent.answer(question)
        test_case = LLMTestCase(
            input=question,
            actual_output=actual,
            expected_output=expected
        )
        test_cases.append(test_case)
    return test_cases

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == "build-index":
        build_index(force="--force" in sys.argv[2:])
        sys.exit()
    if not retriever.is_built():
        sys.exit("Indexes not built yet, run: python main.py build-index")

    if command == "eval":
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            threshold=0.5
        )
        
        test_cases = get_test_cases()
        evaluate(test_cases, [correctness_metric])
    else:
        while (q := input("\nAsk: ").strip()):
            print(agent.answer(q))