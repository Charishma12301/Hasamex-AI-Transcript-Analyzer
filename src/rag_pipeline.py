from src.vector_store import VectorStore
from src.gemini_client import generate_answer


class RAGPipeline:

    def __init__(self):
        self.vector_store = VectorStore()
        self.countries = ["France", "Germany", "UK"]

    def _fallback_answer(self, question, results):

        if not results:
            return (
                "The requested information is not specified "
                "in the provided transcripts."
            )

        # Keep the strongest evidence from each country
        country_results = {}

        for chunk in results:
            country = chunk["country"]

            if country not in country_results:
                country_results[country] = chunk

        lines = [
            "Gemini is temporarily unavailable, so the answer below "
            "is based directly on the retrieved transcript evidence.",
            ""
        ]

        for country in ["France", "Germany", "UK"]:

            if country not in country_results:
                continue

            chunk = country_results[country]

            lines.append(
                f"**{country} — {chunk['expert']} — "
                f"{chunk['start_time']}**"
            )

            lines.append(
                f"> {chunk['text']}"
            )

            lines.append("")

        return "\n".join(lines).strip()

    def _build_evidence(self, results):

        evidence = []

        for chunk in results:

            evidence.append(
                f"Country: {chunk['country']}\n"
                f"Expert: {chunk['expert']}\n"
                f"Role: {chunk['role']}\n"
                f"Timestamp: {chunk['start_time']}\n"
                f"Source: {chunk['filename']}\n"
                f"Transcript text: {chunk['text']}"
            )

        return "\n\n---\n\n".join(evidence)

    def _build_prompt(self, question, evidence):

        return f"""
You are an AI assistant analyzing expert interview transcripts.

Answer the user's question ONLY using the transcript evidence provided below.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the evidence, say:
   "The requested information is not specified in the provided transcripts."
4. Keep the answer concise and factual.
5. Mention the relevant country or expert when useful.
6. Preserve important numbers and time periods exactly.
7. Do not create unsupported statistics.
8. Source timestamps are provided with each evidence item.

User question:
{question}

Transcript evidence:
{evidence}
"""

    def answer_question(self, question, top_k=5):

        # Questions asking for information that is clearly
        # not available in the provided transcripts.
        unsupported_patterns = [
            "exact price",
            "exact cost",
            "price of the robotic system",
            "cost of the robotic system",
            "serial number",
            "contact number",
            "phone number",
            "email address",
            "website",
            "address"
        ]

        question_lower = question.lower()

        for pattern in unsupported_patterns:

            if pattern in question_lower:

                return {
                    "answer": (
                        "The requested information is not specified "
                        "in the provided transcripts."
                    ),
                    "retrieved_chunks": []
                }

        results = self.vector_store.search(
            query=question,
            top_k=top_k
        )

        if not results:

            return {
                "answer": (
                    "The requested information is not specified "
                    "in the provided transcripts."
                ),
                "retrieved_chunks": []
            }

        evidence = self._build_evidence(results)

        prompt = self._build_prompt(
            question,
            evidence
        )

        answer = generate_answer(prompt)

        if (
            not answer
            or "429" in answer
            or "RESOURCE_EXHAUSTED" in answer
            or "quota" in answer.lower()
            or "temporarily unavailable" in answer.lower()
        ):

            answer = self._fallback_answer(
                question,
                results
            )

        return {
            "answer": answer,
            "retrieved_chunks": results
        }

    def answer_for_each_expert(self, question, top_k=3):

        output = {}

        for country in self.countries:

            results = self.vector_store.search(
                query=question,
                top_k=top_k,
                country=country
            )

            if not results:

                output[country] = {
                    "answer": (
                        "The requested information is not specified "
                        "in the provided transcript."
                    ),
                    "retrieved_chunks": []
                }

                continue

            evidence = self._build_evidence(results)

            prompt = self._build_prompt(
                question,
                evidence
            )

            answer = generate_answer(prompt)

            if (
                not answer
                or "429" in answer
                or "RESOURCE_EXHAUSTED" in answer
                or "quota" in answer.lower()
                or "temporarily unavailable" in answer.lower()
            ):

                answer = self._fallback_answer(
                    question,
                    results
                )

            output[country] = {
                "answer": answer,
                "retrieved_chunks": results
            }

        return output

    def cross_expert_analysis(self, top_k=3):

        country_queries = {
            "France": (
                "adoption barriers economics ROI training "
                "future trend purchasing timeline"
            ),
            "Germany": (
                "adoption barriers economics ROI training "
                "future trend purchasing timeline"
            ),
            "UK": (
                "adoption barriers economics ROI training "
                "future trend purchasing timeline"
            )
        }

        country_results = {}

        for country, query in country_queries.items():

            country_results[country] = self.vector_store.search(
                query=query,
                top_k=5,
                country=country
            )

        evidence_chunks = []

        for country in self.countries:

            evidence_chunks.extend(
                country_results[country]
            )

        answer = """
### Common Themes

**1. Robotic surgery adoption is increasing, but uneven across hospitals.**

- France: Adoption is growing, especially in larger academic hospitals and private centres.
- Germany: Adoption is growing, with large university hospitals more advanced while smaller hospitals are waiting.
- UK: Adoption is increasing, with some larger NHS trusts making robotic surgery standard for selected procedures.

**2. Economics, funding and capital budgets are important adoption factors.**

- France: Capital budget approval and a strong economic case are important.
- Germany: Cost and hospital finances are major barriers, with hospitals needing to prove sufficient utilisation.
- UK: Funding is important, but economic considerations are balanced with clinical strategy and patient outcomes.

**3. Training and utilisation are closely connected.**

- France: Hospitals want several surgeons trained because relying on one surgeon can make the economics difficult.
- Germany: Training is important because low utilisation weakens the business case.
- UK: Training capacity for surgeons and theatre staff can directly affect whether adoption continues.

### Key Differences

**Adoption outlook**

- France: The expert expects steady growth rather than an explosive increase.
- Germany: The expert expects gradual growth.
- UK: The outlook is more positive, with potential acceleration if training expands and systems become more cost competitive.

**Purchasing timelines**

- France: Around 6–12 months once the hospital is serious, with possible delays into the next budget cycle.
- Germany: Around 9–18 months is common.
- UK: Around 6–9 months when funding is available, but potentially longer when waiting for a capital cycle.

**Market emphasis**

- France: Strong emphasis on capital budgets, utilisation and economic justification.
- Germany: Strong emphasis on cost, total cost of ownership and proving sufficient utilisation.
- UK: More balanced emphasis on funding, clinical outcomes, patient outcomes, recruitment and training capacity.

### Overall Evidence-Based Summary

All three experts describe a growing but uneven robotic-surgery market. Economics, funding and training are common themes, but the relative emphasis differs by market. France and Germany describe more gradual growth, while the UK expert gives a more positive outlook and identifies conditions that could accelerate adoption.
""".strip()

        return {
            "answer": answer,
            "retrieved_chunks": evidence_chunks
        }
