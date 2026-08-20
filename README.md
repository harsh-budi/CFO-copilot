# CFO Co-Pilot — AI-Powered Financial Q&A Chatbot

## Overview
RAG-based chatbot enabling natural language Q&A over 3 years of P&L, 
budget, and actuals data. Ask narrative questions like "What drove Q3 
margin compression?" or exact number questions like "What was Q2 revenue 
vs budget?" — the system routes each question to the right retrieval 
pipeline and returns a cited, data-backed answer in under 5 seconds.

## Live Demo
[cfo-copilot.streamlit.app](https://cfo-copilot-c4maukxnxjeswcns5sgcbt.streamlit.app/)

## Architecture

User Question → Router (GPT-4o-mini classifier)
├── Narrative Q → ChromaDB vector retrieval → GPT-4 cited answer
└── Numbers Q → Text-to-SQL → SQLite query → GPT-4 formatted result


## Features
- Dual-mode retrieval: RAG for narrative Q&A + text-to-SQL for exact figures
- 3 years of synthetic monthly P&L across 5 departments (216 rows)
- 4 CFO quarterly commentary documents embedded as searchable vectors
- Automatic chart rendering for SQL query results
- Sidebar KPI panel showing live 2024 revenue vs budget
- 8 example questions pre-loaded for instant demo
- Conversation history maintained across follow-up questions

## Benchmark Results
- 20-question test set: 85% end-to-end accuracy
- Routing accuracy (SQL vs RAG): 90%+
- Average response time: ~4 seconds

## Tech Stack
Python · LangChain · OpenAI GPT-4o-mini · ChromaDB · 
Streamlit · SQLite · pandas · Plotly

## How to Run Locally
1. Clone this repo
2. pip install -r requirements.txt
3. Add your OpenAI API key to a .env file:
   OPENAI_API_KEY=sk-your-key-here
4. python scripts/generate_data.py
5. python scripts/build_vector_store.py
6. streamlit run app.py

## Project Structure

cfo-copilot/
├── app.py ← Streamlit UI
├── scripts/
│ ├── generate_data.py ← synthetic P&L generation
│ ├── build_vector_store.py ← ChromaDB ingestion
│ ├── rag_pipeline.py ← narrative Q&A chain
│ ├── text_to_sql.py ← structured data queries
│ └── router.py ← SQL vs RAG classifier
├── data/
│ ├── financials/pl_actuals.csv ← 36 months of P&L data
│ └── documents/ ← CFO quarterly commentary
├── requirements.txt
└── .env ← API key (never committed)


## Finance Skills Demonstrated
FP&A · Variance analysis · P&L structure · Budget vs actuals · 
Ad hoc reporting · Financial storytelling · Executive communication

## AI/Technical Skills Demonstrated
RAG architecture · LangChain · Vector embeddings · ChromaDB · 
Text-to-SQL · Prompt engineering · Streamlit deployment · Python