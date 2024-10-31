import streamlit as st
import PyPDF2
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

class LangChainSemanticChunker:
    def __init__(self, openai_api_key: str, number_of_chunks: int = 30, min_chunk_size: int = 0):
        self.number_of_chunks = number_of_chunks
        self.min_chunk_size = min_chunk_size
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.text_splitter = SemanticChunker(self.embeddings, 
                                             number_of_chunks=self.number_of_chunks,
                                             min_chunk_size=self.min_chunk_size)
        
    def chunk_content(self, content: str, max_chunk_length: int = 2000, min_chunk_length: int = 0) -> List[str]:
        # Configure chunker parameters
        self.text_splitter.chunk_size = max_chunk_length
        
        # Create documents
        chunk_documents = self.text_splitter.create_documents([content])
        chunks = [doc.page_content for doc in chunk_documents]
        
        # Filter chunks by minimum length if specified
        if min_chunk_length > 0:
            chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
            
        return chunks

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Semantic Text Chunker")
    st.write("Upload a PDF or paste text to split it into semantic chunks")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        # Chunking parameters
        st.subheader("Chunking Parameters")
        num_chunks = st.number_input("Number of chunks", min_value=1, value=40)
        min_chunk_size = st.number_input("Minimum chunk size (characters)", 
                                       min_value=20, 
                                       value=20, 
                                       step=10)
        
        st.markdown("""
        ### Instructions
        1. Enter your OpenAI API key
        2. Adjust chunking parameters:
           - Number of chunks to create
           - Minimum characters to keep a chunk
        3. Either upload a PDF or paste text
        4. View color-coded chunks below
        """)

    # Input methods
    tab1, tab2 = st.tabs(["Upload PDF", "Paste Text"])
    
    with tab1:
        pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])
        if pdf_file:
            text_content = extract_text_from_pdf(pdf_file)
    
    with tab2:
        text_input = st.text_area("Or paste your text here", height=200)
        if text_input:
            text_content = text_input

    # Process button
    if st.button("Process Text") and openai_api_key:
        if 'text_content' in locals() and text_content.strip():
            with st.spinner("Processing..."):
                try:
                    chunker = LangChainSemanticChunker(
                        openai_api_key=openai_api_key,
                        number_of_chunks=num_chunks,
                        min_chunk_size=min_chunk_size
                    )
                    chunks = chunker.chunk_content(
                        content=text_content
                    )
                    
                    # Display chunks with alternating colors
                    st.subheader(f"{len(chunks)} chunks in total")
                    
                    # Create columns for chunk display
                    chunk_container = st.container()
                    
                    colors = ["#E6E6FA", "#98FB98"]  # Light purple and light green
                    
                    for i, chunk in enumerate(chunks):
                        with chunk_container:
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: {colors[i % 2]};
                                    padding: 10px;
                                    border-radius: 5px;
                                    margin: 5px 0;
                                    min-height: 50px;
                                ">
                                <p style="margin: 0;"><strong>= Chunk {i+1}: {len(chunk)} characters</strong></p>
                                <p style="margin: 0;">{chunk}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
        else:
            st.warning("Please provide some text to process")
    elif not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar")

if __name__ == "__main__":
    main()