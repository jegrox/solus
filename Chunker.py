from dotenv import load_dotenv
import voyageai
import re


class Chunker:
    def __init__(self, embedding_model="voyage-3-large"):
        self.embedding_model = embedding_model
        self._client = None

    @property
    def client(self):
        # Created on first use so importing/instantiating doesn't require an API key
        if self._client is None:
            load_dotenv()
            self._client = voyageai.Client()
        return self._client

    @staticmethod
    def chunk_by_char(text, chunk_size=150, chunk_overlap=20):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")

        chunks = []
        start_idx = 0

        while start_idx < len(text):
            end_idx = min(start_idx + chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]
            chunks.append(chunk_text)

            start_idx = (
                end_idx - chunk_overlap if end_idx < len(text) else len(text)
            )

        return chunks

    @staticmethod
    def chunk_by_section(document_text):
        pattern = r"\n## "
        return re.split(pattern, document_text)

    @staticmethod
    def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
        if max_sentences_per_chunk <= 0:
            raise ValueError("max_sentences_per_chunk must be positive")
        if not 0 <= overlap_sentences < max_sentences_per_chunk:
            raise ValueError(
                "overlap_sentences must be >= 0 and less than max_sentences_per_chunk"
            )

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        start_idx = 0

        while start_idx < len(sentences):
            end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))
            current_chunk = sentences[start_idx:end_idx]
            chunks.append(" ".join(current_chunk))

            # Stop once the last sentence is included, otherwise the overlap
            # would produce a trailing chunk that's already fully covered
            if end_idx == len(sentences):
                break

            start_idx += max_sentences_per_chunk - overlap_sentences

        return chunks

    def generate_embedding(self, text, input_type="query"):
        result = self.client.embed([text], model=self.embedding_model, input_type=input_type)
        return result.embeddings[0]
