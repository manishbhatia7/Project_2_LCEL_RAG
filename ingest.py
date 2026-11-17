from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from pathlib import Path
import os
load_dotenv()
embedding_model=os.getenv("EMBEDDING_MODEL")
collection_name=os.getenv("CHROMA_COLLECTION")
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
model=OpenAIEmbeddings(model=embedding_model)
pdf1 = PyPDFLoader(str(BASE_DIR / "data" / "IT_Company_Policies.pdf"))
pdf2 = PyPDFLoader(str(BASE_DIR / "data" / "HR_Policy.pdf"))
pdf3 = PyPDFLoader(str(BASE_DIR / "data" / "Leave_Policy.pdf"))
document1=pdf1.load()
document2=pdf2.load()
document3=pdf3.load()
complete_document=document1+document2+document3
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=splitter.split_documents(complete_document)
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=model,
    persist_directory=CHROMA_PATH
)

vector_store.add_documents(documents=chunks,ids=[str(i) for i in range(len(chunks))])
print("Stored records:", vector_store._collection.count())
