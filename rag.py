from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os

# 1. Load documents
docs = []

for file in os.listdir("data"):
    path = os.path.join("data", file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

print(f"Loaded {len(docs)} documents")

# 2. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5. Load local LLM
llm = Ollama(
    model="mistral",
    temperature=0
)

# 6. Create RAG chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template(
    """Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}
"""
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break

    answer = rag_chain.invoke(query)
    print("\nAnswer:\n", answer)


# 7. Ask questions
while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break

    result = rag_chain.invoke(query)

    print("\nAnswer:\n", result)
