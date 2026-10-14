import anthropic
from dotenv import load_dotenv

class Agent():
    def __init__(self, retriever):
        self.agent = ClaudeAgent(retriever)

    def answer(self, question, k=5):
        return self.agent.answer(question, k)

class ClaudeAgent:
    def __init__(self, retriever):
        load_dotenv()
        self.client = anthropic.Anthropic()
        self.retriever = retriever

    def answer(self, question, k=5):
        hits = self.retriever.search(question, k=k)
        context = "\n\n".join(
            f'<doc source="{d["source"]}#{d["chunk"]}">\n{d["content"]}\n</doc>' for d, _ in hits
        )
        response = self.client.beta.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            # betas=["server-side-fallback-2026-07-01"],
            # fallbacks="default",  # retries on another model if a safety filter declines
            system="You are an expert in github, tasked with answering client questions about Github Desktop in natural language."
                "Answer using only the provided documents. Answer concisely in a single paragraph. "
                "If the documents contain a partial answer, provide the answer to the best of your abilities."
                "Validate that all facts are supported by the documentation."
                "Review answer to ensure facts are not missing"
                "If the documents don't contain the answer at all, say so.",
            messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
        )
        if response.stop_reason == "refusal":
            return "The request was declined."
        return "".join(b.text for b in response.content if b.type == "text")