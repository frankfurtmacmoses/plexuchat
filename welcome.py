import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
OPEN_API_KEY ="sk-SocOEFgUl3kbRFLknTq8XkhZAm0DY-Az32dBr7OKWoT3BlbkFJ3gVUlxE45M5h1b28xwhBDFZflslhBZHd0YpVsqGhoAaaaaaaaa"
# Set up Streamlit app title
st.title("PDF Uploader and Embedding with FAISS Vector Store")

# Add a sidebar for multiple PDF file uploads
st.sidebar.header("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

# Check if any files have been uploaded
if uploaded_files:
    st.success(f"{len(uploaded_files)} PDF file(s) uploaded successfully!")
    
    # Initialize the OpenAI Embeddings model
    embedding = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

    # Loop through each uploaded PDF file
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing File: {uploaded_file.name}")
        
        # Read the PDF using PyPDF2
        pdf_reader = PdfReader(uploaded_file)
        
        # Extract text from all pages in the PDF
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
        
        # Check if any text was extracted
        if pdf_text.strip():
            st.write(f"Total characters extracted: {len(pdf_text)}")
            
            # Initialize the LangChain text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Define the size of each chunk (e.g., 1000 characters)
                chunk_overlap=100  # Define the overlap size between chunks (e.g., 100 characters)
            )
            
            # Split the extracted text into chunks
            chunks = text_splitter.split_text(pdf_text)
            st.write(f"Total number of chunks: {len(chunks)}")
            
            # Display the chunks
            for i, chunk in enumerate(chunks):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk)
                st.write("---")  # Divider between chunks
            
            # Generate embeddings for the chunks and create a FAISS vector store
            vector_store = FAISS.from_texts(chunks, embedding)
            st.success("FAISS Vector Store created for this PDF!")
            
        else:
            st.warning("No text could be extracted from this PDF. It might be an image-based PDF.")
        
        st.write("------")  # Divider between files
else:
    st.info("Please upload one or more PDF documents using the sidebar.")

user_text = st.text_input("Type your question here")
if user_text:
    match = vector_store.similarity_search()
    st.write(match)
    llm = ChatOpenAI(openai_api_key = OPEN_API_KEY, temperature=0,max_tokens=1000,model_name='gpt-4-turbo')
    chain = load_qa_chain(llm=llm,chain_type='stuff')
    response = chain.run(input_documents=match,question=user_text)
    st.write(response)          


