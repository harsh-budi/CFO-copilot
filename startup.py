import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')

def ensure_vector_store():
    """Build vector store if it doesn't exist or is empty"""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        count = vectorstore._collection.count()

        if count == 0:
            print("Vector store empty — building now...")
            _build_vector_store()
        else:
            print(f"Vector store ready — {count} vectors loaded")

    except Exception as e:
        print(f"Vector store check failed: {e} — building now...")
        _build_vector_store()

def _build_vector_store():
    """Full rebuild of the vector store from documents"""
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import os

    docs = []
    doc_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'documents'
    )

    if not os.path.exists(doc_dir):
        print(f"Documents folder not found at {doc_dir}")
        return

    for filename in os.listdir(doc_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(doc_dir, filename)
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                loaded = loader.load()
                docs.extend(loaded)
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    if not docs:
        print("No documents found — cannot build vector store")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Vector store built — {vectorstore._collection.count()} vectors")