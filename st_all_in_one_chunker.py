import streamlit as st
import os
from pathlib import Path
import json
import tempfile
from all_in_one_pce import AllInOnePCE
import config


def init_semantic_chunker(api_key: str, model: str, temperature: float, max_tokens: int):
    """Initialize the All In One PCE"""
    return AllInOnePCE(
        openai_api_key=api_key,
        tmp_pdf_png_path=config.PDF_PNG_PATH,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
def display_chunk(chunk: dict, index: int, colors: list):
    """Display a single chunk with formatting"""
    bg_color = colors[index % len(colors)]
    
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        ">
        <p style="margin: 0;"><strong>Chunk {index + 1}</strong></p>
        """,
        unsafe_allow_html=True
    )
    
    # Display chunk contents
    for key, value in chunk.items():
        if key == 'metadata':
            st.markdown("**Metadata:**")
            for meta_key, meta_value in value.items():
                st.markdown(f"- {meta_key}: {meta_value}")
        else:
            st.markdown(f"**{key}:**")
            st.markdown(value)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
def display_metrics(metrics):
    """Display processing metrics"""
    st.subheader("Processing Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pages", metrics.total_pages)
        st.metric("PDF to PNG Time", f"{metrics.pdf_to_png_time:.2f}s")
    with col2:
        st.metric("Successful Pages", metrics.successful_pages)
        st.metric("Encoding Time", f"{metrics.encoding_time:.2f}s")
    with col3:
        st.metric("Failed Pages", metrics.failed_pages)
        st.metric("LLM Processing Time", f"{metrics.llm_processing_time:.2f}s")

def main():
    st.title("LLM based Parser, Semantic Chunker and Chunk Enricher")
    st.write("Upload a PDF to prase, extract semantic chunks and enrich the same using GPT-4o and GPT-4o-mini")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Model parameters
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o"],
            help="OpenAI model to use"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Controls randomness in the output"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1000,
            max_value=16000,
            value=16000,
            step=100,
            help="Maximum tokens in response"
        )
        
        cleanup_images = st.checkbox(
            "Cleanup Images",
            value=True,
            help="Delete temporary PNG files after processing"
        )
        
        save_output = st.checkbox(
            "Save Output",
            value=True,
            help="Save results to JSON file"
        )
        
        st.markdown("""
        ### Instructions
        1. Enter your OpenAI API key
        2. Configure model parameters
        3. Upload a PDF file
        4. View extracted semantic chunks
        5. (Optional) Save results to JSON
        """)

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if uploaded_file and st.button("Process PDF"):
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar")
            return
            
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
            
            # Initialize chunker with configuration
            chunker = init_semantic_chunker(
                api_key=openai_api_key,
                model=model, 
                temperature=temperature, 
                max_tokens=max_tokens
            )
            
            # Process PDF
            with st.spinner("Processing PDF..."):
                chunks, metrics = chunker.process_pdf(
                    pdf_path=pdf_path,
                    save_output=save_output,
                    cleanup_images=cleanup_images
                )
            
            # Display metrics
            display_metrics(metrics)
            
            # Display chunks
            st.subheader("Extracted Chunks")
            colors = ["#E6E6FA", "#98FB98"]  # Light purple and light green
            
            chunk_container = st.container()
            with chunk_container:
                for i, chunk in enumerate(chunks):
                    display_chunk(chunk, i, colors)
            
            # Option to download results
            if chunks:
                st.download_button(
                    "Download JSON",
                    json.dumps(chunks, indent=2),
                    file_name="semantic_chunks.json",
                    mime="application/json"
                )
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            # Cleanup temporary file
            if 'pdf_path' in locals():
                os.unlink(pdf_path)

if __name__ == "__main__":
    main()