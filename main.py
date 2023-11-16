import streamlit as st
import PyPDF2
import io
import chromadb

# Initialize the Chroma Client
chroma_client = chromadb.Client()
collection_name = "my_collection"

# Attempt to get or create the collection
try:
    collection = chroma_client.get_collection(name=collection_name)
    st.write(f"Collection '{collection_name}' retrieved successfully.")
except Exception as e:
    st.error(f"Failed to retrieve collection: {e}. Attempting to create a new collection.")
    try:
        collection = chroma_client.create_collection(name=collection_name)
        st.write(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        st.error(f"Failed to create collection: {e}")


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
        collection.add(
            documents=[text_data],
            metadatas=[{"source": "pdf_upload"}],
            ids=[uploaded_file.name]  # Use the file name as the unique ID
        )
        st.success(f"File '{uploaded_file.name}' successfully uploaded and processed!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM


# Function to generate answer using LLAMA
def generate_answer_with_context(question, context):
    from transformers import AutoTokenizer, TextStreamer
    model_name = "Intel/neural-chat-7b-v3"    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_tokens = str(question) + str(context)
    inputs = tokenizer(input_tokens, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
    
    return outputs

# Streamlit UI for file upload and LLAMA querying

# Functionality to query the collection and LLAMA
st.header("Query Documents with LLAMA")
query = st.text_input("Enter your query here:")
if st.button("Search"):
    with st.spinner('Searching...'):
        # Query ChromaDB and retrieve documents
        # Assuming `collection.query` returns a list of dictionaries with 'text' key for documents
        results = collection.query(
            query_texts=[query],
            n_results=5  # Assuming we want the top 5 results
        )
        
        if results:
            # For demonstration, we're taking just the top result for context
            context = results
            answer = generate_answer_with_context(query, context)
            st.write(answer)
        else:
            st.write("No results found.")
