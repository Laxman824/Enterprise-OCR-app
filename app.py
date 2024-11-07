import streamlit as st
from src.processors.document_processor import EnterpriseDocumentProcessor
from src.utils.visualization import DocumentVisualizer
import io
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
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
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
        
        # Document type selection
        doc_type = st.selectbox(
            "Select Document Type",
            ["Auto Detect", "Invoice", "Bank Statement", "Pay Slip", "Expense Report"],
            help="Select specific document type or let system detect automatically"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Minimum confidence score for field extraction"
        )

    # Main content
    st.title("Enterprise Document Processor")
    st.write("Upload documents for intelligent information extraction")

    # Initialize processor if not in session state
    if 'processor' not in st.session_state:
        st.session_state.processor = EnterpriseDocumentProcessor()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
        accept_multiple_files=True,
        help="Upload one or more documents for processing"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")
            
            try:
                # Create processing placeholder
                with st.spinner("Processing document..."):
                    # Read image
                    image = Image.open(uploaded_file)
                    
                    # Process document
                    results = st.session_state.processor.process_document(
                        image=image,
                        doc_type=None if doc_type == "Auto Detect" else doc_type,
                        confidence_threshold=confidence_threshold
                    )
                    
                    if results:
                        # Display results using tabs
                        tabs = st.tabs([
                            "Visual Analysis", 
                            "Extracted Data", 
                            "Validation Results",
                            "Export Options"
                        ])
                        
                        with tabs[0]:
                            # Visual analysis with interactive elements
                            visualizer.create_interactive_display(image, results)
                        
                        with tabs[1]:
                            # Extracted data in structured format
                            visualizer.create_summary_view(results)
                        
                        with tabs[2]:
                            # Validation results and warnings
                            visualizer.display_validation_results(results)
                        
                        with tabs[3]:
                            # Export options
                            visualizer.create_export_buttons(results, uploaded_file.name)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.code(traceback.format_exc())

    # Display usage instructions when no file is uploaded
    else:
        st.info("""
        👆 Upload a document to get started!
        
        Supported document types:
        - Invoices
        - Bank Statements
        - Pay Slips
        - Expense Reports
        
        Supported file formats: PDF, PNG, JPG, JPEG, TIFF
        """)

if __name__ == "__main__":
    main()