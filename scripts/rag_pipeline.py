from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── System prompt: tells GPT-4 how to behave ─────────────────
# This is the most important thing to tune for quality answers.
# The rules prevent hallucination and keep answers finance-specific.
SYSTEM_PROMPT = """You are a CFO Co-Pilot assistant with expertise in
corporate FP&A and financial analysis.

Answer the question using ONLY the financial context provided below.
Do not use any outside knowledge.

Rules you must follow:
- Be specific — always cite exact figures from the context
- Always mention the time period (Q3 2024, FY2023, etc.)
- Format dollar amounts with $ and M/K (e.g. $5.8M, $340K)
- Format percentages with % and 1 decimal place (e.g. 23.8%)
- If the context does not contain enough information to answer,
  say exactly: "I don't have that data in my current knowledge base.
  Try asking about a specific quarter or time period."
- Do NOT make up numbers or percentages

Financial context retrieved from documents:
{context}
"""

def load_vectorstore():
    """Load the existing ChromaDB vector store from disk"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vectorstore

def build_rag_chain():
    """Build the full RAG pipeline chain"""
    vectorstore = load_vectorstore()

    # retriever: takes a question, returns top 4 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",  # use gpt-4o for higher quality (higher cost)
        temperature=0         # 0 = deterministic, no creativity, just facts
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    def format_docs(docs):
        # joins retrieved chunks with a separator for readability
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # chain: question → retriever → format → prompt → GPT → string output
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ── Test the RAG chain directly ───────────────────────────────
if __name__ == "__main__":
    print("Building RAG chain...")
    chain = build_rag_chain()

    test_questions = [
        "What drove the Q3 2024 margin compression?",
        "What is the Engineering headcount plan?",
        "What was FY2023 EBITDA margin?",
        "What are the key risks mentioned for Q4?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"A: {chain.invoke(q)}")