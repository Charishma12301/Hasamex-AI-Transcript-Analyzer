\# Hasamex AI Transcript Analyzer



A RAG-based AI application that analyzes three expert interview transcripts from France, Germany, and the UK. The application answers interview-guide questions, retrieves supporting evidence, shows exact source timestamps, identifies common themes and differences, and allows users to ask questions across all transcripts.



\## Objective



The goal of this project is to build a simple and traceable AI system for analyzing expert-call transcripts while reducing hallucinations.



The application ensures that important answers can be traced back to the original transcript, expert, country, and timestamp.



\## Features



\* Analyze all 6 interview-guide questions

\* Compare responses from France, Germany, and the UK

\* Retrieve relevant transcript evidence using semantic search

\* Show expert name, role, source file, and timestamp

\* Display exact transcript quotes as supporting evidence

\* Identify common themes across experts

\* Identify differences between markets

\* Compare adoption outlooks and purchasing timelines

\* Ask custom questions across all three transcripts

\* Prevent unsupported information from being presented as fact

\* Provide a fallback response when the Gemini API is unavailable

\* Simple Streamlit interface



\## Architecture



```text

3 Expert Transcripts

&#x20;       ↓

Text + Timestamp Extraction

&#x20;       ↓

Chunking + Metadata

&#x20;       ↓

Sentence Transformer Embeddings

&#x20;       ↓

FAISS Vector Search

&#x20;       ↓

Relevant Evidence

&#x20;       ↓

Gemini Flash

&#x20;       ↓

Answer + Quote + Timestamp

&#x20;       ↓

Streamlit UI

```



\## Technology Stack



| Component             | Technology            |

| --------------------- | --------------------- |

| Programming Language  | Python 3.12           |

| UI                    | Streamlit             |

| LLM                   | Google Gemini Flash   |

| Embeddings            | Sentence Transformers |

| Embedding Model       | all-MiniLM-L6-v2      |

| Vector Database       | FAISS                 |

| Environment Variables | python-dotenv         |

| Version Control       | Git / GitHub          |



\## Retrieval Approach



The application uses a custom RAG pipeline.



\### 1. Transcript Loading



The application reads the three transcript files from the `data/` directory.



Each transcript contains timestamped expert responses.



\### 2. Timestamp Extraction



Transcript responses are split using timestamps such as:



```text

00:18

01:20

02:18

```



The timestamp is stored as metadata with each chunk.



\### 3. Metadata



Each transcript chunk stores:



\* Country

\* Expert

\* Expert role

\* Source filename

\* Timestamp

\* Transcript text



This metadata allows the application to trace retrieved evidence back to the original source.



\### 4. Chunking



Each timestamped expert response is treated as an individual retrieval chunk.



The current dataset contains:



\* 3 transcripts

\* 21 expert-response chunks

\* 7 expert-response chunks per transcript



\### 5. Embeddings



Each chunk is converted into a vector using:



`all-MiniLM-L6-v2`



The embeddings are normalized before being stored.



\### 6. FAISS Search



FAISS is used for fast vector similarity search.



The application combines:



\* Semantic similarity

\* Keyword matching

\* Question-type relevance



This improves retrieval for questions about barriers, economics, training, future trends, and purchasing timelines.



\## Model Choice



\### Sentence Transformer



`all-MiniLM-L6-v2` was selected because it is lightweight and provides effective semantic embeddings for a small transcript dataset.



\### Gemini Flash



Gemini Flash is used as the generation layer after retrieval.



The LLM receives only the retrieved transcript evidence and is instructed to:



\* Use only supplied evidence

\* Avoid outside knowledge

\* Preserve important numbers and time periods

\* Avoid unsupported statistics

\* State when information is not available



\## Citation and Timestamp Handling



Every retrieved chunk contains:



```text

Country

Expert

Role

Timestamp

Source

Transcript text

```



The Streamlit interface displays this information with the generated answer.



For example:



```text

France | Dr. Jean Martin | 03:10



Source: Transcript\_1\_France.txt



Dr. Martin: Training matters, especially in the first year...

```



This makes answers traceable to the original transcript.



\## Hallucination Reduction



The application uses multiple safeguards.



\### Evidence-first generation



The system first retrieves relevant transcript chunks and then sends those chunks to the LLM.



The LLM does not receive a request to answer from general knowledge.



\### Grounded prompt



The generation prompt explicitly instructs the model:



```text

Answer the user's question ONLY using the transcript evidence provided.



Do not invent information.

Do not use outside knowledge.

Do not create unsupported statistics.

```



\### Unsupported information handling



If requested information is not present in the transcripts, the application returns:



```text

The requested information is not specified in the provided transcripts.

```



For example, when asked for the exact price of the robotic system, the application does not invent a price.



\### Gemini fallback



If Gemini is temporarily unavailable or the API quota is exhausted, the application directly displays retrieved transcript evidence instead of fabricating an answer.



\## Cross-Expert Analysis



The application identifies common themes and differences across the three markets.



\### Common themes



\* Robotic surgery adoption is increasing but uneven.

\* Hospital funding and capital budgets influence adoption.

\* Economics and utilisation are important.

\* Surgeon and staff training is important.

\* Purchasing decisions can take several months.



\### Market differences



France emphasizes capital budgets, utilisation, and economic justification.



Germany emphasizes cost, total cost of ownership, utilisation, and procurement alignment.



The UK places stronger emphasis on balancing funding and economics with clinical strategy, outcomes, recruitment, and training capacity.



\### Adoption outlook



France describes steady growth.



Germany describes gradual growth.



The UK expert provides a more positive outlook and identifies conditions that could accelerate adoption.



The application presents these differences without creating unsupported aggregate statistics.



\## Scaling from 3 to 30+ Transcripts



The current implementation uses local files and FAISS, which is sufficient for the small case-study dataset.



For 30+ transcripts, the architecture could be extended as follows:



```text

Documents

&#x20;   ↓

Cloud Object Storage

&#x20;   ↓

Document Processing Pipeline

&#x20;   ↓

Chunking + Metadata

&#x20;   ↓

Embedding Service

&#x20;   ↓

Production Vector Database

&#x20;   ↓

Metadata Filtering

&#x20;   ↓

Retrieval API

&#x20;   ↓

LLM

&#x20;   ↓

Streamlit / Web Application

```



Potential production improvements include:



\* Cloud object storage for transcript files

\* Production vector database

\* Metadata filtering by country, project, expert, or date

\* Batch embedding generation

\* Retrieval evaluation datasets

\* Caching

\* API-based backend

\* Authentication

\* Logging and monitoring

\* Automated evaluation of retrieval quality

\* Larger context management for long transcripts



\## Project Structure



```text

Hasamex-AI-Transcript-Analyzer/

│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── data/

│   ├── Interview\_Guide.txt

│   ├── Transcript\_1\_France.txt

│   ├── Transcript\_2\_Germany.txt

│   └── Transcript\_3\_UK.txt

│

└── src/

&#x20;   ├── load\_data.py

&#x20;   ├── chunk\_data.py

&#x20;   ├── vector\_store.py

&#x20;   ├── gemini\_client.py

&#x20;   └── rag\_pipeline.py

```



\## Installation



\### 1. Clone the repository



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd Hasamex-AI-Transcript-Analyzer

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the virtual environment



Windows PowerShell:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Configure Gemini API key



Create a `.env` file in the project root:



```text

GEMINI\_API\_KEY=your\_api\_key\_here

```



Do not commit the `.env` file to GitHub.



The repository already uses `.gitignore` to exclude `.env`.



\## Run the Application



Start Streamlit:



```bash

streamlit run app.py

```



The application will open in the browser.



Default local URL:



```text

http://localhost:8501

```



\## How to Use



\### Interview Guide



Select any of the six interview-guide questions and click the corresponding analysis button.



The application displays responses for:



\* France

\* Germany

\* UK



Each response includes supporting transcript evidence and timestamps.



\### Cross-Expert Analysis



Click:



`Analyze Common Themes \& Differences`



The application displays:



\* Common themes

\* Key differences

\* Adoption outlook

\* Purchasing timelines

\* Market-specific emphasis

\* Supporting evidence



\### Ask Questions



Enter a question that can be answered using the transcripts.



Example:



```text

How important is surgeon training for adoption?

```



The application retrieves evidence from all three transcripts.



For information that is not contained in the transcripts, the application responds:



```text

The requested information is not specified in the provided transcripts.

```



\## Testing



The application was tested against all six interview-guide questions for all three experts.



Primary evidence was verified for:



| Question            | France | Germany | UK    |

| ------------------- | ------ | ------- | ----- |

| Current adoption    | 00:18  | 00:16   | 00:14 |

| Main barriers       | 01:20  | 01:10   | 01:05 |

| Budgets / ROI       | 02:18  | 02:08   | 03:10 |

| Training / outcomes | 03:10  | 03:05   | 01:05 |

| 3–5 year trend      | 05:07  | 04:09   | 04:06 |

| Purchasing timeline | 06:08  | 06:05   | 05:04 |



An additional hallucination-control test was performed by asking for the exact price of the robotic system. Since the transcripts do not provide an exact price, the application correctly returned:



```text

The requested information is not specified in the provided transcripts.

```



\## Limitations



\* The current dataset contains only three transcripts.

\* Gemini generation depends on API availability and quota.

\* Cross-expert analysis uses a deterministic evidence-based summary for reliable operation when the LLM is unavailable.

\* FAISS is currently used as a local vector index and is intended for the small case-study dataset.

\* The application does not infer information that is absent from the provided transcripts.



\## Future Improvements



\* Add transcript upload functionality

\* Support PDF/DOCX transcript ingestion

\* Add automatic source citation generation

\* Add retrieval confidence thresholds

\* Add evaluation metrics such as Recall@K

\* Add automated answer faithfulness evaluation

\* Add production vector database

\* Add user authentication

\* Deploy the application to Streamlit Community Cloud

\* Add conversation history for multi-turn questions



\## Author



Charishma Akuthota



B.Tech Information Technology, 2026



````



After pasting, press \*\*Ctrl + S\*\*.



Then run:



```powershell

git status

````



\*\*Important:\*\* Before we push to GitHub, we must verify that `.env` is NOT appearing in `git status`. That is critical because your Gemini API key is inside `.env`.



Send me the `git status` output next.



