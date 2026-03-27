import argparse
import shutil
import uuid
from pathlib import Path
from typing import List


def supported_data_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return [p for p in sorted(data_dir.iterdir()) if p.suffix.lower() in {".pdf", ".txt"}]


def load_documents(data_dir: str = "data"):
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs = []
    for path in supported_data_files(Path(data_dir)):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No .pdf or .txt documents found in {data_dir}")
    return docs


def build_vectorstore(
    docs,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: str = "chroma_db",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    reset_index: bool = True,
):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents and created {len(chunks)} chunks")

    persist_path = Path(persist_directory)
    collection_name = "meeting_summary"
    if reset_index and persist_path.exists():
        try:
            shutil.rmtree(persist_path)
            print(f"Reset existing vector index at {persist_path}")
        except PermissionError:
            # On Windows, Chroma files can stay locked briefly; use a fresh collection instead.
            collection_name = f"meeting_summary_{uuid.uuid4().hex[:8]}"
            print(
                "Could not delete existing index directory due to a file lock. "
                f"Using fresh collection '{collection_name}' in {persist_path}"
            )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def build_rag_chain(vectorstore, llm_model: str = "mistral", top_k: int = 3, temperature: float = 0.0):
    from langchain_ollama import OllamaLLM
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = OllamaLLM(model=llm_model, temperature=temperature)

    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    summary_prompt = PromptTemplate.from_template(
        """You are an expert Project Manager helping summarize meetings.

Use only the provided context. Do not invent facts, owners, deadlines, or decisions.
If information is missing or ambiguous, state it explicitly.

Return your answer using EXACTLY this structure:

SUMMARY:
- Exactly 3 sentences with a high-level overview.

KEY_DECISIONS:
- Bullet list of finalized decisions from the meeting.
- If none are clearly finalized, write: - None clearly finalized in context.

ACTION_ITEMS:
- Bullet list in this exact format: [Owner] - [Task Description]
- If owner is unknown, use [Unassigned].
- If no action items are found, write: - [Unassigned] - No explicit action items found in context.

OPEN_QUESTIONS:
- Bullet list of unresolved questions or risks.
- If none, write: - None identified from context.

Context:
{context}

User request:
{question}
"""
    )

    return (
        {"context": retriever | RunnableLambda(_format_docs), "question": RunnablePassthrough()}
        | summary_prompt
        | llm
        | StrOutputParser()
    )


def summarize_meeting(rag_chain) -> None:
    query = (
        "Summarize this meeting transcript. Focus on key decisions, action items, "
        "owners, deadlines, and open questions."
    )
    print("\nMeeting Summary:\n")
    print(rag_chain.invoke(query))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local RAG index and summarize the meeting.")
    parser.add_argument("--data-dir", default="data", help="Folder with .pdf and .txt documents.")
    parser.add_argument("--persist-dir", default="chroma_db", help="Folder where Chroma DB is stored.")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="mistral")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--keep-existing-index",
        action="store_true",
        help="Keep existing Chroma index instead of rebuilding from current data files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_documents(data_dir=args.data_dir)
    vectorstore = build_vectorstore(
        docs,
        embedding_model=args.embedding_model,
        persist_directory=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset_index=not args.keep_existing_index,
    )
    rag_chain = build_rag_chain(
        vectorstore,
        llm_model=args.llm_model,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    summarize_meeting(rag_chain)


if __name__ == "__main__":
    main()
