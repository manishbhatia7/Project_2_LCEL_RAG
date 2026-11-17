
# Company Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers employee questions using company policy PDF documents.

## Project Overview

This project helps employees find information from HR, Leave, and IT policy documents without manually searching through multiple PDFs.

The application retrieves relevant document chunks from Chroma and uses an OpenAI language model to generate context-grounded answers.

## Tech Stack

- Python
- LangChain and LCEL
- OpenAI GPT-4.1-mini
- OpenAI Embeddings
- Chroma Vector Database
- MMR Retrieval
- PyPDF

## RAG Architecture

### Document Ingestion Pipeline

PDF Documents
→ PyPDFLoader
→ RecursiveCharacterTextSplitter
→ OpenAI Embeddings
→ Chroma Vector Database

### Question-Answering Pipeline

User Question
→ MMR Retriever
→ Relevant Document Chunks
→ Context Formatting
→ ChatPromptTemplate
→ OpenAI LLM
→ StrOutputParser
→ Answer + Retrieved Sources

## Features

- Supports multiple policy PDFs
- Semantic document retrieval
- MMR search for result diversity
- Context-based answer generation
- PDF filename and page metadata
- Handling of unavailable information
- Basic input validation

## Project Structure

    Project_2_LCEL_RAG/
    ├── data/
    │   ├── HR_Policy.pdf
    │   ├── Leave_Policy.pdf
    │   └── IT_Company_Policies.pdf
    ├── company_policy_assistant.py
    ├── ingest.py
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore
    └── README.md

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

    pip install -r requirements.txt

4. Create a `.env` file using `.env.example`.
5. Add your OpenAI API key.
6. Run the ingestion script:

    python ingest.py

7. Run the assistant:

    python company_policy_assistant.py

## Example Questions

- How many casual leaves are employees entitled to?
- How many sick leaves are available?
- Can earned leave be carried forward?
- What is the company's work-from-home allowance?

## Testing

Manual testing covers direct policy questions, unsupported questions, and paraphrased questions.

Verified examples:

- Casual Leave: 8 days per year
- Work-from-home allowance: Correctly reports unavailable information

## Known Limitations

- Some paraphrased questions may retrieve incomplete context.
- The question "How many days can I take off for personal reasons?" may omit Casual Leave information.
- Retrieved source pages are not verified claim-level citations.
- The application currently uses a command-line interface.

## Future Improvements

- Hybrid search
- Automated RAG evaluation
- Verified citations
- Streamlit interface
- Deployment
