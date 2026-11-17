
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------------
# 1. Load environment variables
# -----------------------------------------

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL")
)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    temperature=0
)

collection_name = os.getenv("CHROMA_COLLECTION")


# -----------------------------------------
# 2. Connect to Chroma database
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_PATH)
)


# -----------------------------------------
# 3. Configure MMR retriever
# -----------------------------------------

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 15
    }
)


# -----------------------------------------
# 4. Format retrieved documents
# -----------------------------------------

def format_docs(documents):
    context = []

    for document in documents:
        context.append(document.page_content)

    return "\n\n".join(context)


# -----------------------------------------
# 5. Extract source metadata
# -----------------------------------------

def format_docs_metadata(documents):
    sources = []

    for document in documents:
        metadata = document.metadata

        source = Path(
            metadata.get("source", "")
        ).name

        page_no = metadata.get("page_label")

        source_info = {
            "source": source,
            "page_no": page_no
        }

        if source_info not in sources:
            sources.append(source_info)

    return sources


# -----------------------------------------
# 6. Create prompt template
# -----------------------------------------

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a Company Policy Assistant.

Answer employee questions using only the provided
company policy context.

Instructions:
1. Use only information present in the context.
2. Do not invent company policies or benefits.
3. If the context does not contain the answer,
   respond: "Sorry, this information is not
   available in the provided company policies."
4. Identify the policy most relevant to the question.
5. Respect the eligibility conditions for each policy.
6. Do not recommend Sick Leave for non-medical
   reasons unless the policy explicitly permits it.
7. If the question is ambiguous, explain the
   applicable options or ask for clarification.
8. Keep answers clear and concise.

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])


# -----------------------------------------
# 7. Create LCEL chain
# -----------------------------------------

final_chain = (
    prompt_template
    | llm
    | StrOutputParser()
)


# -----------------------------------------
# 8. RAG question-answering function
# -----------------------------------------

def ask_question(question):

    if not isinstance(question, str) or not question.strip():
        return {
            "question": question,
            "answer": "Please enter a valid question.",
            "retrieved_sources": []
        }

    question = question.strip()

    # Retrieve documents only once
    documents = retriever.invoke(question)

    # Prepare context
    context = format_docs(documents)

    # Generate answer using LCEL
    answer = final_chain.invoke({
        "context": context,
        "question": question
    })

    # Extract source metadata
    sources = format_docs_metadata(documents)

    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": sources
    }


# -----------------------------------------
# 9. Test application
# -----------------------------------------

if __name__ == "__main__":

    questions = [
        "How many casual leaves are employees entitled to?",
        "How many days can I take off for personal reasons?",
        "What is the company's work-from-home allowance?"
    ]

    for question in questions:

        result = ask_question(question)

        print("\n" + "=" * 60)
        print("Question:", result["question"])
        print("Answer:", result["answer"])

        print("\nRetrieved Sources:")

        for source in result["retrieved_sources"]:
            print(
                "PDF: {} | Page: {}".format(
                    source["source"],
                    source["page_no"]
                )
            )
