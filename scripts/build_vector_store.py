import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # loads your API key from .env

# ── Step 1: Load all .txt documents from data/documents/ ──────
docs = []
doc_dir = 'data/documents'

for filename in os.listdir(doc_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(doc_dir, filename)
        loader = TextLoader(filepath, encoding='utf-8')
        loaded = loader.load()
        docs.extend(loaded)
        print(f"Loaded: {filename} ({len(loaded[0].page_content)} chars)")

print(f"\nTotal documents loaded: {len(docs)}")

# ── Step 2: Split documents into chunks ───────────────────────
# Why? GPT-4 has a token limit. We break docs into 500-char chunks
# so we only send the most relevant pieces, not the entire document.
# chunk_overlap=50 means adjacent chunks share 50 chars — prevents
# cutting a sentence in half and losing context.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# ── Step 3: Create embeddings and store in ChromaDB ───────────
# OpenAIEmbeddings converts each chunk to a vector of numbers.
# Chroma.from_documents stores all vectors locally in chroma_db/
# persist_directory tells ChromaDB where to save the database file.
print("\nCreating embeddings (this takes ~20 seconds)...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

total_vectors = vectorstore._collection.count()
print(f"Vector store created with {total_vectors} vectors")

# ── Step 4: Test retrieval to confirm it works ────────────────
# This is the most important test — ask a real question and see
# if the returned chunks are relevant. If they are, the RAG will work.
print("\n─── Test retrieval ───")
test_questions = [
    "What drove Q3 2024 margin compression?",
    "What is the Q4 revenue outlook?",
    "How did Engineering costs change?"
]

for q in test_questions:
    results = vectorstore.similarity_search(q, k=2)
    print(f"\nQ: {q}")
    print(f"Top result ({len(results[0].page_content)} chars):")
    print(results[0].page_content[:200] + "...")

print("\nVector store build complete.")