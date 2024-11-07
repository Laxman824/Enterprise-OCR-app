import streamlit as st
from processors.document_processor import EnterpriseDocumentProcessor
from utils.visualization import DocumentVisualizer
import io
import traceback
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Enterprise Document Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize visualizer
    visualizer = DocumentVisualizer()
    
    # Custom CSS
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stButton>button { width: 100%; }
        .upload-box {
            border: 2px dashed #cccccc;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Document Settings")
        doc_type = st.selectbox(
            "Select Document Type",
            ["Auto Detect", "Invoice", "Bank Statement", "Pay Slip", "Expense Report"]
        )
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    st.title("Enterprise Document Processor")
    st.write("Upload documents for intelligent information extraction")

    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = EnterpriseDocumentProcessor()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    image = Image.open(uploaded_file)
                    
                    results = st.session_state.processor.process_document(
                        image=image,
                        doc_type=None if doc_type == "Auto Detect" else doc_type,
                        confidence_threshold=confidence_threshold
                    )
                    
                    if results:
                        tabs = st.tabs(["Visual Analysis", "Results", "Validation"])
                        
                        with tabs[0]:
                            visualizer.create_interactive_display(image, results)
                        
                        with tabs[1]:
                            visualizer.create_summary_view(results)
                        
                        with tabs[2]:
                            visualizer.create_validation_view(results)
                            
                        # Download buttons
                        st.download_button(
                            "Download Results (JSON)",
                            visualizer.get_json_results(results),
                            f"{uploaded_file.name}_results.json",
                            "application/json"
                        )

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.code(traceback.format_exc())

    else:
        st.info("👆 Upload a document to get started!")

if __name__ == "__main__":
    main()