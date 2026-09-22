import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunk_data import build_all_chunks


MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = build_all_chunks()

        texts = [chunk["text"] for chunk in self.chunks]

        print(f"Creating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.embeddings = embeddings.astype("float32")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        print(
            f"FAISS index created: "
            f"{self.index.ntotal} vectors, "
            f"dimension {dimension}"
        )

    def _keyword_score(self, query, text):
        query_words = set(query.lower().split())
        text_lower = text.lower()

        score = 0

        for word in query_words:
            word = word.strip(".,?!:;()[]")

            if len(word) < 4:
                continue

            if word in text_lower:
                score += 1

        return score

    def _question_type_score(self, query, text):
        query_lower = query.lower()
        text_lower = text.lower()

        score = 0

        if "barriers" in query_lower:
            barrier_terms = [
                "barrier",
                "barriers",
                "biggest issue",
                "biggest barrier",
                "cost",
                "capital",
                "budget",
                "funding",
                "hospital finances",
                "training capacity",
                "adoption stalls",
                "under pressure",
                "competing capital priorities"
            ]

            for term in barrier_terms:
                if term in text_lower:
                    score += 6

            future_terms = [
                "i am quite positive",
                "continued growth",
                "growth will be gradual",
                "adoption could accelerate",
                "15 percent",
                "20 percent",
                "high single digits",
                "low double digits"
            ]

            for term in future_terms:
                if term in text_lower:
                    score -= 8

        if (
            "budget" in query_lower
            or "roi" in query_lower
            or "purchasing decisions" in query_lower
        ):
            economic_terms = [
                "roi",
                "economic",
                "economics",
                "finance",
                "budget",
                "capital",
                "pay for itself",
                "cost",
                "total cost of ownership",
                "procedure volume",
                "maintenance",
                "service contracts"
            ]

            for term in economic_terms:
                if term in text_lower:
                    score += 5

        if (
            "training" in query_lower
            or "clinical outcomes" in query_lower
        ):
            training_terms = [
                "training",
                "trained",
                "surgeon",
                "theatre staff",
                "utilisation",
                "outcomes",
                "clinical outcomes"
            ]

            for term in training_terms:
                if term in text_lower:
                    score += 5

        if (
            "3–5" in query_lower
            or "3-5" in query_lower
            or "future" in query_lower
            or "trend" in query_lower
        ):
            future_terms = [
                "expect",
                "future",
                "growth",
                "annually",
                "accelerate",
                "continued growth",
                "gradual",
                "positive",
                "next"
            ]

            for term in future_terms:
                if term in text_lower:
                    score += 5

        if (
            "timeline" in query_lower
            or "decision-making" in query_lower
        ):
            timeline_terms = [
                "months",
                "timeline",
                "budget cycle",
                "procurement",
                "capital cycle",
                "six months",
                "nine months",
                "twelve months",
                "eighteen months"
            ]

            for term in timeline_terms:
                if term in text_lower:
                    score += 6

        return score

    def search(self, query, top_k=5, country=None):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            self.index.ntotal
        )

        candidates = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            chunk = self.chunks[index].copy()

            if country is not None:
                if chunk["country"] != country:
                    continue

            semantic_score = float(score)

            keyword_score = self._keyword_score(
                query,
                chunk["text"]
            )

            question_score = self._question_type_score(
                query,
                chunk["text"]
            )

            final_score = (
                semantic_score
                + (0.03 * keyword_score)
                + (0.08 * question_score)
            )

            chunk["semantic_score"] = semantic_score
            chunk["keyword_score"] = keyword_score
            chunk["question_score"] = question_score
            chunk["score"] = final_score

            candidates.append(chunk)

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return candidates[:top_k]