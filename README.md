# Streamlit-based Document Processing and Querying with LLAMA

This Streamlit application is designed to upload, process, and store documents in ChromaDB, and query these documents using a LLAMA language model. It utilizes `transformers`, `PyPDF2`, `Chroma`, and other libraries for processing and querying documents.

## Features

- **Document Processing**: Upload PDF documents, extract their text, and store them in ChromaDB.
- **ChromaDB Integration**: Initialize and interact with ChromaDB for document persistence and retrieval.
- **LLAMA Integration**: Utilize LLAMA, a powerful causal language model, to generate answers based on the context from the documents stored in ChromaDB.
- **Interactive UI**: Easy-to-use Streamlit interface for uploading documents and querying.

## Requirements

- Python 3
- Streamlit (`streamlit`)
- Transformers (`transformers`)
- PyPDF2 (`PyPDF2`)
- LangChain (`langchain`)
- ChromaDB (`chromadb`)
- Other dependencies as required by these packages

## Setup

1. **Install Dependencies**: Use `pip install streamlit transformers PyPDF2 langchain chromadb`.
2. **Run the Application**: Execute `streamlit run main.py` to start the Streamlit application.

## Usage

- **Upload Documents**: Use the file uploader to add PDF documents to ChromaDB.
- **Query with LLAMA**: Enter your query, and the application will search ChromaDB for relevant documents and use LLAMA to generate context-based answers.

## Customization

- **ChromaDB Directory**: Set the directory for ChromaDB storage. Default is `./chromadb`.
- **Model for Embeddings**: The application uses `all-MiniLM-L6-v2` for sentence embeddings. This can be customized in the code.

## Notes
- Ensure that ChromaDB is initialized and has documents before querying.
- The application handles basic error scenarios like empty ChromaDB directories. Nothing too crazy of an error handling is going here otherwise. 
