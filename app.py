import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.load_data import load_interview_guide

st.set_page_config(
    page_title="Hasamex AI Transcript Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hasamex AI Transcript Analyzer")
st.write("Analyze expert interview transcripts using RAG-based AI with source timestamps.")

@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

rag = load_rag_pipeline()

guide_text = load_interview_guide()
questions = []

for line in guide_text.splitlines():
    line = line.strip()

    if not line:
        continue

    if line[0].isdigit() and "." in line:
        question = line.split(".", 1)[1].strip()

        if question:
            questions.append(question)

st.header("📋 Interview Guide")
st.write("Analyze each interview-guide question across France, Germany, and the UK.")

for i, question in enumerate(questions, start=1):

    st.subheader(f"Question {i}")
    st.write(question)

    if st.button(f"Analyze Question {i}", key=f"guide_button_{i}"):

        with st.spinner("Analyzing transcript evidence..."):
            expert_answers = rag.answer_for_each_expert(
                question,
                top_k=3
            )

        france_tab, germany_tab, uk_tab = st.tabs(
            ["🇫🇷 France", "🇩🇪 Germany", "🇬🇧 UK"]
        )

        with france_tab:
            result = expert_answers["France"]

            st.markdown("### Expert Analysis")
            st.write(result["answer"])

            st.markdown("### Source Evidence")

            for chunk in result["retrieved_chunks"]:
                st.write(f"**Expert:** {chunk['expert']}")
                st.write(f"**Role:** {chunk['role']}")
                st.write(f"**Timestamp:** {chunk['start_time']}")
                st.write(f"**Source:** {chunk['filename']}")
                st.info(chunk["text"])

        with germany_tab:
            result = expert_answers["Germany"]

            st.markdown("### Expert Analysis")
            st.write(result["answer"])

            st.markdown("### Source Evidence")

            for chunk in result["retrieved_chunks"]:
                st.write(f"**Expert:** {chunk['expert']}")
                st.write(f"**Role:** {chunk['role']}")
                st.write(f"**Timestamp:** {chunk['start_time']}")
                st.write(f"**Source:** {chunk['filename']}")
                st.info(chunk["text"])

        with uk_tab:
            result = expert_answers["UK"]

            st.markdown("### Expert Analysis")
            st.write(result["answer"])

            st.markdown("### Source Evidence")

            for chunk in result["retrieved_chunks"]:
                st.write(f"**Expert:** {chunk['expert']}")
                st.write(f"**Role:** {chunk['role']}")
                st.write(f"**Timestamp:** {chunk['start_time']}")
                st.write(f"**Source:** {chunk['filename']}")
                st.info(chunk["text"])

st.header("🔎 Cross-Expert Analysis")
st.write("Compare common themes and differences across France, Germany, and the UK.")

if st.button("Analyze Common Themes & Differences", key="cross_expert_button"):

    with st.spinner("Comparing transcript evidence..."):
        result = rag.cross_expert_analysis(top_k=3)

    st.markdown("### Cross-Expert Analysis")
    st.write(result["answer"])

    st.markdown("### Supporting Evidence")

    for chunk in result["retrieved_chunks"]:

        with st.expander(
            f"{chunk['country']} | {chunk['expert']} | {chunk['start_time']}"
        ):

            st.write(f"**Source:** {chunk['filename']}")
            st.write(chunk["text"])

st.header("💬 Ask Questions")
st.write("Ask a question across all three transcripts.")

user_question = st.text_input(
    "Enter your question:",
    key="user_question"
)

if st.button("Ask AI", key="ask_ai_button"):

    if not user_question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching transcripts and generating answer..."):
            result = rag.answer_question(
                user_question,
                top_k=5
            )

        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Retrieved Sources")

        for chunk in result["retrieved_chunks"]:

            with st.expander(
                f"{chunk['country']} | "
                f"{chunk['expert']} | "
                f"{chunk['start_time']} | "
                f"Score: {chunk['score']:.3f}"
            ):

                st.write(f"**Role:** {chunk['role']}")
                st.write(f"**Source:** {chunk['filename']}")
                st.write(f"**Timestamp:** {chunk['start_time']}")
                st.write(chunk["text"])

st.header("🏗️ Architecture")

st.code(
    """
3 Expert Transcripts
        ↓
Text + Timestamp Extraction
        ↓
Chunking + Metadata
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Search
        ↓
Relevant Evidence
        ↓
Gemini Flash
        ↓
Answer + Quote + Timestamp
        ↓
Streamlit UI
""",
    language="text"
)

st.caption("Hasamex AI Engineer Technical Case Study")
