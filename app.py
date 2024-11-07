import streamlit as st
from src.processors.document_processor import EnterpriseDocumentProcessor
from src.utils.visualization import create_download_buttons
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
        
        # Document type selection (if not auto-detected)
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
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            enable_table_extraction = st.checkbox("Enable Table Extraction", value=True)
            enable_auto_correction = st.checkbox("Enable Auto Correction", value=True)
            enable_validation = st.checkbox("Enable Validation", value=True)

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
                        confidence_threshold=confidence_threshold,
                        enable_table_extraction=enable_table_extraction,
                        enable_auto_correction=enable_auto_correction,
                        enable_validation=enable_validation
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
                            st.session_state.processor.create_interactive_display(
                                image, results
                            )
                        
                        with tabs[1]:
                            # Extracted data in structured format
                            st.session_state.processor.display_extracted_data(
                                results
                            )
                        
                        with tabs[2]:
                            # Validation results and warnings
                            st.session_state.processor.display_validation_results(
                                results
                            )
                        
                        with tabs[3]:
                            # Export options
                            create_download_buttons(results, uploaded_file.name)

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

    # Footer
    st.markdown("""---""")
    st.markdown(
        """
        <div style="text-align: center">
            <p>Enterprise Document Processor v1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()