from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
import re
from doctr.models import ocr_predictor
from easyocr import Reader
import pandas as pd
from datetime import datetime
import traceback

@dataclass
class DocumentField:
    name: str
    pattern: str
    description: str
    example: str
    validation_type: str
    category: str
    is_key_field: bool = False
    location_hints: List[str] = None
    related_fields: List[str] = None
    custom_validator: callable = None

    def __post_init__(self):
        if self.location_hints is None:
            self.location_hints = []
        if self.related_fields is None:
            self.related_fields = []

# @dataclass
# class DocumentTemplate:
#     name: str
#     description: str
#     fields: Dict[str, DocumentField]
#     key_identifiers: List[str]
#     validation_rules: Dict[str, List[str]]
#     sample_formats: List[str]

# # Define document templates for different types
# DOCUMENT_TEMPLATES = {
#     "invoice": DocumentTemplate(
#         name="Invoice",
#         description="Commercial invoices and bills",
#         key_identifiers=["invoice", "bill to", "amount due", "tax", "total"],
#         fields={
#             "invoice_number": DocumentField(
#                 name="Invoice Number",
#                 pattern=r"(?i)(?:invoice|bill)\s*(?:#|number|num|no)?[:.]?\s*([A-Z0-9-]+)",
#                 description="Unique invoice identifier",
#                 example="INV-12345",
#                 validation_type="invoice_number",
#                 category="Header",
#                 is_key_field=True
#             ),
#             "date": DocumentField(
#                 name="Invoice Date",
#                 pattern=r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",
#                 description="Invoice issue date",
#                 example="01/01/2024",
#                 validation_type="date",
#                 category="Header",
#                 is_key_field=True
#             ),
#             "total_amount": DocumentField(
#                 name="Total Amount",
#                 pattern=r"\$?\s*\d{1,3}(?:,\d{3})*\.\d{2}",
#                 description="Total invoice amount",
#                 example="$1,234.56",
#                 validation_type="amount",
#                 category="Summary",
#                 is_key_field=True
#             )
#         },
#         validation_rules={
#             "date": ["valid_date", "not_future"],
#             "total_amount": ["positive_amount", "matches_items"]
#         },
#         sample_formats=["pdf", "png", "jpg"]
#     ),
    
#     "bank_statement": DocumentTemplate(
#         name="Bank Statement",
#         description="Bank account statements",
#         key_identifiers=["account statement", "balance", "withdrawal", "deposit"],
#         fields={
#             "account_number": DocumentField(
#                 name="Account Number",
#                 pattern=r"\b\d{8,12}\b",
#                 description="Bank account number",
#                 example="1234567890",
#                 validation_type="account_number",
#                 category="Header",
#                 is_key_field=True
#             ),
#             "statement_period": DocumentField(
#                 name="Statement Period",
#                 pattern=r"(?i)statement period:?\s*(.*)",
#                 description="Statement period",
#                 example="Jan 1 - Jan 31, 2024",
#                 validation_type="date_range",
#                 category="Header",
#                 is_key_field=True
#             )
#         },
#         validation_rules={
#             "balance": ["valid_amount", "matches_calculations"]
#         },
#         sample_formats=["pdf", "png", "jpg"]
#     ),
    
#     "pay_slip": DocumentTemplate(
#         name="Pay Slip",
#         description="Employee payment slips",
#         key_identifiers=["salary", "pay slip", "earnings", "deductions"],
#         fields={
#             "employee_id": DocumentField(
#                 name="Employee ID",
#                 pattern=r"(?i)emp(?:loyee)?\s*(?:id|number|no)[:.]?\s*([A-Z0-9]+)",
#                 description="Employee identification number",
#                 example="EMP123",
#                 validation_type="employee_id",
#                 category="Header",
#                 is_key_field=True
#             ),
#             "net_pay": DocumentField(
#                 name="Net Pay",
#                 pattern=r"\$?\s*\d{1,3}(?:,\d{3})*\.\d{2}",
#                 description="Net payment amount",
#                 example="$3,500.00",
#                 validation_type="amount",
#                 category="Summary",
#                 is_key_field=True
#             )
#         },
#         validation_rules={
#             "net_pay": ["valid_amount", "matches_calculations"]
#         },
#         sample_formats=["pdf", "png", "jpg"]
#     ),
    
#     "expense_report": DocumentTemplate(
#         name="Expense Report",
#         description="Business expense reports",
#         key_identifiers=["expense", "reimbursement", "receipt"],
#         fields={
#             "report_id": DocumentField(
#                 name="Report ID",
#                 pattern=r"(?i)report\s*(?:id|number|no)[:.]?\s*([A-Z0-9-]+)",
#                 description="Expense report identifier",
#                 example="EXP-2024-001",
#                 validation_type="report_id",
#                 category="Header",
#                 is_key_field=True
#             ),
#             "total_expenses": DocumentField(
#                 name="Total Expenses",
#                 pattern=r"\$?\s*\d{1,3}(?:,\d{3})*\.\d{2}",
#                 description="Total expense amount",
#                 example="$1,234.56",
#                 validation_type="amount",
#                 category="Summary",
#                 is_key_field=True
#             )
#         },
#         validation_rules={
#             "total_expenses": ["valid_amount", "matches_items"]
#         },
#         sample_formats=["pdf", "png", "jpg"]
#     )
# }

class EnterpriseDocumentProcessor:
    def __init__(self):
        self.doctr_model = self._load_doctr_model()
        self.easyocr_reader = self._load_easyocr()
        self.templates = DOCUMENT_TEMPLATES
        self.current_template = None
        self.extracted_data = {}
        self.bounding_boxes = []

    @st.cache_resource
    def _load_doctr_model(self):
        try:
            return ocr_predictor(pretrained=True)
        except Exception as e:
            st.error(f"Error loading DocTR model: {str(e)}")
            st.stop()

    @st.cache_resource
    def _load_easyocr(self):
        try:
            return Reader(['en'])
        except Exception as e:
            st.error(f"Error loading EasyOCR: {str(e)}")
            st.stop()

    def detect_document_type(self, text: str) -> str:
        """Detect document type based on key identifiers"""
        scores = {}
        for doc_type, template in self.templates.items():
            score = 0
            for identifier in template.key_identifiers:
                if re.search(identifier, text, re.IGNORECASE):
                    score += 1
            scores[doc_type] = score

        if not scores:
            return None

        # Return document type with highest score
        return max(scores.items(), key=lambda x: x[1])[0]


##############

    def process_document(self, image: Image.Image) -> Dict:
        """Main document processing pipeline"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract text using both OCR engines
            extracted_text, boxes = self._extract_text(processed_image)
            
            # Detect document type
            doc_type = self.detect_document_type(extracted_text)
            if doc_type:
                st.success(f"Detected document type: {doc_type.replace('_', ' ').title()}")
                self.current_template = self.templates[doc_type]
            else:
                # Let user select document type
                doc_type = st.selectbox(
                    "Please select document type:",
                    list(self.templates.keys()),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                self.current_template = self.templates[doc_type]

            # Extract fields based on template
            results = self._process_with_template(extracted_text, boxes)
            
            # Store bounding boxes for visualization
            self.bounding_boxes = boxes
            
            return results

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.code(traceback.format_exc())
            return None

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_np = np.array(image)
        
        # Grayscale conversion
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(binary)

    def _extract_text(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        """Extract text using both OCR engines"""
        # DocTR extraction
        doctr_boxes = self._extract_with_doctr(image)
        
        # EasyOCR extraction
        easyocr_boxes = self._extract_with_easyocr(image)
        
        # Combine results
        combined_boxes = self._combine_results(doctr_boxes, easyocr_boxes)
        
        # Create full text from boxes
        full_text = ' '.join([box['text'] for box in combined_boxes])
        
        return full_text, combined_boxes

    def _process_with_template(self, text: str, boxes: List[Dict]) -> Dict:
        """Process document using current template"""
        results = {
            'document_type': self.current_template.name,
            'fields': {},
            'confidence_scores': {},
            'validation_results': {},
            'warnings': [],
            'table_data': [],
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'template_version': '1.0'
            }
        }
        
        # Process each field in template
        for field_name, field_def in self.current_template.fields.items():
            field_result = self._extract_field_value(text, boxes, field_def)
            if field_result:
                results['fields'][field_name] = field_result['value']
                results['confidence_scores'][field_name] = field_result['confidence']
                results['validation_results'][field_name] = field_result['is_valid']
                
                if not field_result['is_valid']:
                    results['warnings'].append(
                        f"{field_name}: {field_result['validation_message']}"
                    )

        # Extract table data if present
        table_data = self._extract_table_data(boxes)
        if table_data:
            results['table_data'] = table_data

        return results

    def create_interactive_display(self, image: Image.Image, results: Dict):
        """Create interactive display with enhanced features"""
        st.subheader("Document Analysis Results")
        
        # Create tabs for different views
        tabs = st.tabs(["Visual Analysis", "Extracted Data", "Validation Results", "JSON View"])
        
        with tabs[0]:
            self._create_visual_analysis(image, results)
            
        with tabs[1]:
            self._create_data_view(results)
            
        with tabs[2]:
            self._create_validation_view(results)
            
        with tabs[3]:
            self._create_json_view(results)

    def _create_visual_analysis(self, image: Image.Image, results: Dict):
        """Create interactive visual analysis view"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image with interactive bounding boxes
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(image)
            
            # Draw bounding boxes
            for box in self.bounding_boxes:
                x1, y1, x2, y2 = box['box']
                confidence = box.get('confidence', 0)
                field_name = box.get('field_name', '')
                
                color = self._get_confidence_color(confidence)
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.5
                )
                ax.add_patch(rect)
                
                # Add hover text
                ax.text(x1, y1-5, field_name, fontsize=8, color=color)
            
            ax.axis('off')
            st.pyplot(fig)
            
            # Add custom CSS for hover effects
            st.markdown("""
                <style>
                .hover-info {
                    position: absolute;
                    background: white;
                    border: 1px solid black;
                    padding: 5px;
                    display: none;
                }
                </style>
                """, unsafe_allow_html=True)
            
        with col2:
            # Field highlighting controls
            st.subheader("Field Controls")
            for field_name in results['fields']:
                if st.checkbox(field_name, value=True):
                    # Update visualization to highlight selected field
                    self._highlight_field(field_name)

    def _create_data_view(self, results: Dict):
        """Create structured data view"""
        st.subheader("Extracted Information")
        
        # Group fields by category
        grouped_fields = {}
        for field_name, value in results['fields'].items():
            category = self.current_template.fields[field_name].category
            if category not in grouped_fields:
                grouped_fields[category] = []
            grouped_fields[category].append((field_name, value))
        
        # Display grouped fields
        for category, fields in grouped_fields.items():
            with st.expander(category, expanded=True):
                for field_name, value in fields:
                    confidence = results['confidence_scores'][field_name]
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{field_name}:**")
                    with col2:
                        st.write(value)
                    with col3:
                        st.progress(confidence)

        # Display table data if present
        if results.get('table_data'):
            st.subheader("Table Data")
            st.dataframe(pd.DataFrame(results['table_data']))

    def _create_validation_view(self, results: Dict):
        """Create validation results view"""
        st.subheader("Validation Results")
        
        # Display validation status for each field
        for field_name, is_valid in results['validation_results'].items():
            color = 'green' if is_valid else 'red'
            status = '✓' if is_valid else '✗'
            st.markdown(
                f'<p style="color: {color}">{status} {field_name}</p>',
                unsafe_allow_html=True
            )
        
        # Display warnings
        if results['warnings']:
            st.subheader("Warnings")
            for warning in results['warnings']:
                st.warning(warning)

    def _create_json_view(self, results: Dict):
        """Create JSON view with download option"""
        st.subheader("JSON Output")
        
        # Display formatted JSON
        st.json(results)
        
        # Add download button
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    st.set_page_config(page_title="Enterprise Document Processor", layout="wide")
    
    st.title("Enterprise Document Processing System")
    st.write("Upload any document for intelligent information extraction")
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = EnterpriseDocumentProcessor()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
        help="Upload a document for processing"
    )
    
    if uploaded_file:
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Process document
            results = st.session_state.processor.process_document(image)
            
            if results:
                # Create interactive display
                st.session_state.processor.create_interactive_display(image, results)
                
                # Export options
                st.subheader("Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export as Excel
                    excel_data = pd.DataFrame({
                        'Field': results['fields'].keys(),
                        'Value': results['fields'].values(),
                        'Confidence': [results['confidence_scores'].get(k, 0) 
                                     for k in results['fields'].keys()]
                    })
                    
                    st.download_button(
                        "Download Excel",
                        excel_data.to_excel(index=False).encode(),
                        f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # Export as CSV
                    st.download_button(
                        "Download CSV",
                        excel_data.to_csv(index=False),
                        f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
