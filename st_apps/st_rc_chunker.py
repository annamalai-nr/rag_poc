import streamlit as st
import PyPDF2
from typing import List
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LangChainRCChunker:
    def __init__(self):
        """Initialize the LangChain Chunker"""
        logging.info("Initialized LangChain Chunker")
        
    def chunk_content(self,
                     content: str,
                     max_chunk_length: int = 500,
                     min_chunk_length: int = 0,
                     chunk_overlap: int = 100,
                     max_input_chars: int = None) -> List[str]:
        try:
            # Truncate content if max_input_chars is specified
            if max_input_chars:
                content = content[:max_input_chars]
            
            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_length,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
            # Create documents and extract text
            chunk_documents = text_splitter.create_documents([content])
            chunks = [doc.page_content for doc in chunk_documents]
            
            # Filter chunks by minimum length if specified
            if min_chunk_length > 0:
                chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_length]
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error in chunk_content: {str(e)}")
            raise

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Recursive Character Text Chunker")
    st.write("Upload a PDF or paste text to split it into chunks")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Chunking Parameters")
        
        # Main parameters
        max_chunk_size = st.number_input(
            "Maximum chunk size (characters)", 
            min_value=100, 
            value=500, 
            step=100,
            help="Maximum number of characters in each chunk"
        )
        
        chunk_overlap = st.number_input(
            "Chunk overlap (characters)", 
            min_value=0, 
            value=100, 
            step=50,
            help="Number of characters to overlap between chunks"
        )
        
        # Advanced parameters with expander
        with st.expander("Advanced Parameters"):
            min_chunk_size = st.number_input(
                "Minimum chunk size (characters)", 
                min_value=0, 
                value=0, 
                step=50,
                help="Minimum number of characters required to keep a chunk"
            )
            
            max_input_chars = st.number_input(
                "Maximum input characters", 
                min_value=0, 
                value=None, 
                help="Maximum number of characters to process (0 for no limit)"
            )
            if max_input_chars == 0:
                max_input_chars = None
        
        st.markdown("""
        ### Instructions
        1. Set chunking parameters:
           - Maximum chunk size
           - Overlap between chunks
           - (Optional) Advanced parameters
        2. Either upload a PDF or paste text
        3. View color-coded chunks below
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
    if st.button("Process Text"):
        if 'text_content' in locals() and text_content.strip():
            with st.spinner("Processing..."):
                try:
                    chunker = LangChainRCChunker()
                    chunks = chunker.chunk_content(
                        content=text_content,
                        max_chunk_length=int(max_chunk_size),
                        min_chunk_length=int(min_chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        max_input_chars=max_input_chars
                    )
                    
                    # Display stats
                    total_chars = sum(len(chunk) for chunk in chunks)
                    avg_chunk_size = total_chars / len(chunks) if chunks else 0
                    
                    st.subheader("Chunking Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", len(chunks))
                    with col2:
                        st.metric("Average Chunk Size", f"{avg_chunk_size:.0f}")
                    with col3:
                        st.metric("Total Characters", total_chars)
                    
                    # Display chunks with alternating colors
                    st.subheader("Generated Chunks")
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

if __name__ == "__main__":
    main()