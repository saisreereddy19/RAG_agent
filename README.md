# RAG Agent – Retrieval-Augmented Generation Q&A System

## Overview
This project is a **Retrieval-Augmented Generation (RAG) AI Agent** built to answer questions from a small knowledge base (PDF or text documents). The agent demonstrates a full AI workflow, including query planning, document retrieval, answer generation, and relevance reflection.

The project uses:
- **ChromaDB** as a local vector database.
- **Sentence Transformers** for embeddings.
- **Google Gemini 2.0** for LLM-based answer generation.
- **LangGraph-style nodes** for modular AI agent workflow: **Plan → Retrieve → Answer → Reflect**.
- Optional **trace/logging and evaluation** for answer relevance.

---

## Features

1. **Query Interpretation (Plan Node)**  
   - Detects short casual messages (e.g., "Hi", "Hello") and generates a suitable response without querying the database.  
   - For other queries, decides if retrieval is needed.

2. **Document Retrieval (Retrieve Node)**  
   - Splits PDF files into chunks.
   - Embeds chunks into vectors using SentenceTransformers.
   - Stores and queries embeddings in ChromaDB.

3. **Answer Generation (Answer Node)**  
   - Uses **Google Gemini 2.0 LLM** to generate answers based on retrieved context.
   - Can generate detailed responses even for long queries.

4. **Answer Reflection (Reflect Node)**  
   - Evaluates the answer for relevance using a simple keyword-based heuristic.
   - Optional automated evaluation using a second LLM or metrics like BLEU/ROUGE/BERTScore.

5. **Interactive Terminal Chat**  
   - Simple user interface showing **User → Bot** style Q&A.
   - Handles exit commands (`exit`, `quit`, `bye`).

6. **Logging & Trace** *(Bonus)*  
   - Tracks each step of the workflow: plan, retrieval, answer, and reflection.
   - Can be extended to use **LangSmith** or **TruLens** for advanced evaluation and monitoring.

---

## Installation

1. Clone the repository:

git clone https://github.com/saisreereddy19/RAG_agent.git
cd RAG_agent

2.Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

3.Install dependencies:
pip install -r requirements.txt

4.Add your Google Gemini API key to a .env file:
GOOGLE_API_KEY=your_api_key_here
---

## Enhancements & Improvements

Plan Node Enhancement
Short casual messages are handled without querying the database.
Queries are classified as RAG-required or direct-response.
Answer Generation Improvements
Uses Gemini 2.0 Flash model with max output tokens set.
Produces detailed, context-aware answers.
Tracks each stage of the agent pipeline.
Can be extended with LangSmith/TruLens for advanced evaluation.
Checks relevance automatically by comparing keywords in query and answer.
Can be enhanced using a second LLM or BLEU/ROUGE/BERTScore metrics.
