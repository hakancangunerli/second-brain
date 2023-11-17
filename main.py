from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import PyPDF2
import io
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import chromadb
import os 
# Initialize Chroma for persistence
persist_directory = "./chromadb"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# try to load a chromadb vectordb, if you can't find it, tell the user to create one.

try:
    st.write("Loading ChromaDB...")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    # check if the folder is empty, if it's empty tell them to add some files first.
    if os.listdir(persist_directory) == []:
        st.error("Please create a ChromaDB first.")    
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please create a ChromaDB first.")

# Function to process PDF file and extract text
def process_pdf(uploaded_file):
    file_stream = io.BytesIO(uploaded_file.getvalue())
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text_data = ""
    for page_num in range(len(pdf_reader.pages)):
        text_data += pdf_reader.pages[page_num].extract_text()
    return text_data


# Streamlit UI for file upload
st.header("Upload to ChromaDB")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Extract text from the PDF file
    text_data = process_pdf(uploaded_file)

    # Attempt to add the extracted text to the ChromaDB collection
    try:
        vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        
        vector_db.add_texts([text_data])
        
        st.success(
            f"File '{uploaded_file.name}' successfully uploaded and processed!")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Function to generate answer using LLAMA
def generate_answer_with_context(user_input, context):
    from ctransformers import AutoModelForCausalLM
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF")

    return llm(f"""
                [INST]
                <<SYS>>
                 The following is your context, {context}
                 Answer the question:
            <</SYS>>
                {user_input}[/INST]
             """)


# Streamlit UI for file upload and LLAMA querying

# Functionality to query the collection and LLAMA
st.header("Query Documents with LLAMA")
question = st.text_input("Enter your query here:")
if st.button("Search"):
    with st.spinner('Searching...'):
        # Query ChromaDB and retrieve documents
        # Assuming `collection.query` returns a list of dictionaries with 'text' key for documents
        results = vector_db.similarity_search(question)

        if results:
            # For demonstration, we're taking just the top result for context
            answer = generate_answer_with_context(
                question, results[0].page_content)
            st.write(answer)
        else:
            st.write("No results found.")
