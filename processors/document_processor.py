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


from typing import Dict, List, Optional, Tuple
from doctr.models import ocr_predictor
import easyocr

from utils.validation import DocumentValidator

# class EnterpriseDocumentProcessor:
#     def __init__(self):
#         self.doctr_model = self._load_doctr_model()
#         self.easyocr_reader = self._load_easyocr()
#         self.templates = DOCUMENT_TEMPLATES
#         self.current_template = None
#         self.extracted_data = {}
#         self.bounding_boxes = []

#     @st.cache_resource
#     def _load_doctr_model(self):
#         try:
#             return ocr_predictor(pretrained=True)
#         except Exception as e:
#             st.error(f"Error loading DocTR model: {str(e)}")
#             st.stop()

#     @st.cache_resource
#     def _load_easyocr(self):
#         try:
#             return Reader(['en'])
#         except Exception as e:
#             st.error(f"Error loading EasyOCR: {str(e)}")
#             st.stop()

#     def detect_document_type(self, text: str) -> str:
#         """Detect document type based on key identifiers"""
#         scores = {}
#         for doc_type, template in self.templates.items():
#             score = 0
#             for identifier in template.key_identifiers:
#                 if re.search(identifier, text, re.IGNORECASE):
#                     score += 1
#             scores[doc_type] = score

#         if not scores:
#             return None

#         # Return document type with highest score
#         return max(scores.items(), key=lambda x: x[1])[0]


# ##############

#     def process_document(self, image: Image.Image) -> Dict:
#         """Main document processing pipeline"""
#         try:
#             # Preprocess image
#             processed_image = self._preprocess_image(image)
            
#             # Extract text using both OCR engines
#             extracted_text, boxes = self._extract_text(processed_image)
            
#             # Detect document type
#             doc_type = self.detect_document_type(extracted_text)
#             if doc_type:
#                 st.success(f"Detected document type: {doc_type.replace('_', ' ').title()}")
#                 self.current_template = self.templates[doc_type]
#             else:
#                 # Let user select document type
#                 doc_type = st.selectbox(
#                     "Please select document type:",
#                     list(self.templates.keys()),
#                     format_func=lambda x: x.replace('_', ' ').title()
#                 )
#                 self.current_template = self.templates[doc_type]

#             # Extract fields based on template
#             results = self._process_with_template(extracted_text, boxes)
            
#             # Store bounding boxes for visualization
#             self.bounding_boxes = boxes
            
#             return results

#         except Exception as e:
#             st.error(f"Error processing document: {str(e)}")
#             st.code(traceback.format_exc())
#             return None

#     def _preprocess_image(self, image: Image.Image) -> Image.Image:
#         """Enhanced image preprocessing"""
#         # Convert to RGB if needed
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Convert to numpy array
#         img_np = np.array(image)
        
#         # Grayscale conversion
#         gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
#         # Noise reduction
#         denoised = cv2.fastNlMeansDenoising(gray)
        
#         # Contrast enhancement
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         enhanced = clahe.apply(denoised)
        
#         # Adaptive thresholding
#         binary = cv2.adaptiveThreshold(
#             enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY, 11, 2
#         )
        
#         return Image.fromarray(binary)

#     def _extract_text(self, image: Image.Image) -> Tuple[str, List[Dict]]:
#         """Extract text using both OCR engines"""
#         # DocTR extraction
#         doctr_boxes = self._extract_with_doctr(image)
        
#         # EasyOCR extraction
#         easyocr_boxes = self._extract_with_easyocr(image)
        
#         # Combine results
#         combined_boxes = self._combine_results(doctr_boxes, easyocr_boxes)
        
#         # Create full text from boxes
#         full_text = ' '.join([box['text'] for box in combined_boxes])
        
#         return full_text, combined_boxes

#     def _process_with_template(self, text: str, boxes: List[Dict]) -> Dict:
#         """Process document using current template"""
#         results = {
#             'document_type': self.current_template.name,
#             'fields': {},
#             'confidence_scores': {},
#             'validation_results': {},
#             'warnings': [],
#             'table_data': [],
#             'metadata': {
#                 'processing_date': datetime.now().isoformat(),
#                 'template_version': '1.0'
#             }
#         }
        
#         # Process each field in template
#         for field_name, field_def in self.current_template.fields.items():
#             field_result = self._extract_field_value(text, boxes, field_def)
#             if field_result:
#                 results['fields'][field_name] = field_result['value']
#                 results['confidence_scores'][field_name] = field_result['confidence']
#                 results['validation_results'][field_name] = field_result['is_valid']
                
#                 if not field_result['is_valid']:
#                     results['warnings'].append(
#                         f"{field_name}: {field_result['validation_message']}"
#                     )

#         # Extract table data if present
#         table_data = self._extract_table_data(boxes)
#         if table_data:
#             results['table_data'] = table_data

#         return results

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

#     def _create_visual_analysis(self, image: Image.Image, results: Dict):
#         """Create interactive visual analysis view"""
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             # Display image with interactive bounding boxes
#             fig, ax = plt.subplots(figsize=(12, 8))
#             ax.imshow(image)
            
#             # Draw bounding boxes
#             for box in self.bounding_boxes:
#                 x1, y1, x2, y2 = box['box']
#                 confidence = box.get('confidence', 0)
#                 field_name = box.get('field_name', '')
                
#                 color = self._get_confidence_color(confidence)
#                 rect = patches.Rectangle(
#                     (x1, y1), x2-x1, y2-y1,
#                     linewidth=2, edgecolor=color, facecolor='none', alpha=0.5
#                 )
#                 ax.add_patch(rect)
                
#                 # Add hover text
#                 ax.text(x1, y1-5, field_name, fontsize=8, color=color)
            
#             ax.axis('off')
#             st.pyplot(fig)
            
#             # Add custom CSS for hover effects
#             st.markdown("""
#                 <style>
#                 .hover-info {
#                     position: absolute;
#                     background: white;
#                     border: 1px solid black;
#                     padding: 5px;
#                     display: none;
#                 }
#                 </style>
#                 """, unsafe_allow_html=True)
            
#         with col2:
#             # Field highlighting controls
#             st.subheader("Field Controls")
#             for field_name in results['fields']:
#                 if st.checkbox(field_name, value=True):
#                     # Update visualization to highlight selected field
#                     self._highlight_field(field_name)

#     def _create_data_view(self, results: Dict):
#         """Create structured data view"""
#         st.subheader("Extracted Information")
        
#         # Group fields by category
#         grouped_fields = {}
#         for field_name, value in results['fields'].items():
#             category = self.current_template.fields[field_name].category
#             if category not in grouped_fields:
#                 grouped_fields[category] = []
#             grouped_fields[category].append((field_name, value))
        
#         # Display grouped fields
#         for category, fields in grouped_fields.items():
#             with st.expander(category, expanded=True):
#                 for field_name, value in fields:
#                     confidence = results['confidence_scores'][field_name]
#                     col1, col2, col3 = st.columns([2, 2, 1])
#                     with col1:
#                         st.write(f"**{field_name}:**")
#                     with col2:
#                         st.write(value)
#                     with col3:
#                         st.progress(confidence)

#         # Display table data if present
#         if results.get('table_data'):
#             st.subheader("Table Data")
#             st.dataframe(pd.DataFrame(results['table_data']))

#     def _create_validation_view(self, results: Dict):
#         """Create validation results view"""
#         st.subheader("Validation Results")
        
#         # Display validation status for each field
#         for field_name, is_valid in results['validation_results'].items():
#             color = 'green' if is_valid else 'red'
#             status = '✓' if is_valid else '✗'
#             st.markdown(
#                 f'<p style="color: {color}">{status} {field_name}</p>',
#                 unsafe_allow_html=True
#             )
        
#         # Display warnings
#         if results['warnings']:
#             st.subheader("Warnings")
#             for warning in results['warnings']:
#                 st.warning(warning)

#     def _create_json_view(self, results: Dict):
#         """Create JSON view with download option"""
#         st.subheader("JSON Output")
        
#         # Display formatted JSON
#         st.json(results)
        
#         # Add download button
#         json_str = json.dumps(results, indent=2)
#         st.download_button(
#             label="Download JSON",
#             data=json_str,
#             file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#             mime="application/json"
#         )

# @dataclass
# class DocumentField:
#     name: str
#     pattern: str
#     description: str
#     example: str
#     validation_type: str
#     category: str
#     is_key_field: bool = False
#     location_hints: List[str] = None
#     related_fields: List[str] = None
#     custom_validator: callable = None

#     def __post_init__(self):
#         if self.location_hints is None:
#             self.location_hints = []
#         if self.related_fields is None:
#             self.related_fields = []



# def main():
#     st.set_page_config(page_title="Enterprise Document Processor", layout="wide")
    
#     st.title("Enterprise Document Processing System")
#     st.write("Upload any document for intelligent information extraction")
    
#     # Initialize processor
#     if 'processor' not in st.session_state:
#         st.session_state.processor = EnterpriseDocumentProcessor()
    
#     # File upload
#     uploaded_file = st.file_uploader(
#         "Upload Document",
#         type=['pdf', 'png', 'jpg', 'jpeg', 'tiff'],
#         help="Upload a document for processing"
#     )
    
#     if uploaded_file:
#         try:
#             # Read image
#             image = Image.open(uploaded_file)
            
#             # Process document
#             results = st.session_state.processor.process_document(image)
            
#             if results:
#                 # Create interactive display
#                 st.session_state.processor.create_interactive_display(image, results)
                
#                 # Export options
#                 st.subheader("Export Options")
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     # Export as Excel
#                     excel_data = pd.DataFrame({
#                         'Field': results['fields'].keys(),
#                         'Value': results['fields'].values(),
#                         'Confidence': [results['confidence_scores'].get(k, 0) 
#                                      for k in results['fields'].keys()]
#                     })
                    
#                     st.download_button(
#                         "Download Excel",
#                         excel_data.to_excel(index=False).encode(),
#                         f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
#                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                     )
                
#                 with col2:
#                     # Export as CSV
#                     st.download_button(
#                         "Download CSV",
#                         excel_data.to_csv(index=False),
#                         f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                         "text/csv"
#                     )
                
#         except Exception as e:
#             st.error(f"Error processing document: {str(e)}")
#             st.code(traceback.format_exc())

# if __name__ == "__main__":
#     main()
# import cv2
# import numpy as np
# from PIL import Image
# from typing import Dict, List, Optional, Tuple
# from doctr.models import ocr_predictor
# import easyocr
# import streamlit as st
# from utils.validation import DocumentValidator
# import re
# from datetime import datetime

class EnterpriseDocumentProcessor:
    def __init__(self):
        # Initialize models using static methods
        self.doctr_model = EnterpriseDocumentProcessor.load_doctr_model()
        self.easyocr_reader = EnterpriseDocumentProcessor.load_easyocr()
        self.validator = DocumentValidator()
        self.bounding_boxes = []
        self.extracted_data = {}

    @staticmethod
    @st.cache_resource
    def load_doctr_model():
        """Load and cache DocTR model"""
        try:
            return ocr_predictor(pretrained=True)
        except Exception as e:
            st.error(f"Error loading DocTR model: {str(e)}")
            return None

    @staticmethod
    @st.cache_resource
    def load_easyocr():
        """Load and cache EasyOCR reader"""
        try:
            return easyocr.Reader(['en'])
        except Exception as e:
            st.error(f"Error loading EasyOCR: {str(e)}")
            return None

    def process_document(self, image: Image.Image, 
                        doc_type: Optional[str] = None,
                        confidence_threshold: float = 0.5) -> Dict:
        """Process document and extract information"""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract text using both OCR engines
            extracted_text, boxes = self._extract_text(processed_image)
            
            # Detect document type if not specified
            if not doc_type or doc_type == "Auto Detect":
                doc_type = self._detect_document_type(extracted_text)
                if doc_type:
                    st.info(f"Detected document type: {doc_type}")
            
            # Process based on document type
            results = self._process_by_type(
                doc_type, 
                extracted_text, 
                boxes,
                confidence_threshold
            )
            
            # Store bounding boxes for visualization
            self.bounding_boxes = boxes
            
            return results
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return None

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR"""
        # Convert to numpy array
        img_np = np.array(image)
        
        # Convert to grayscale
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Thresholding
        _, binary = cv2.threshold(
            enhanced, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
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
        
        # Create full text
        full_text = ' '.join([box['text'] for box in combined_boxes])
        
        return full_text, combined_boxes

    def _extract_with_doctr(self, image: Image.Image) -> List[Dict]:
        """Extract text using DocTR"""
        if not self.doctr_model:
            return []
            
        try:
            result = self.doctr_model([np.array(image)])
            boxes = []
            
            for block in result.pages[0].blocks:
                for line in block.lines:
                    for word in line.words:
                        box = {
                            'text': word.value,
                            'box': [
                                int(word.geometry[0][0] * image.width),
                                int(word.geometry[0][1] * image.height),
                                int(word.geometry[1][0] * image.width),
                                int(word.geometry[1][1] * image.height)
                            ],
                            'confidence': float(word.confidence),
                            'source': 'doctr'
                        }
                        boxes.append(box)
                        
            return boxes
            
        except Exception as e:
            st.warning(f"DocTR extraction warning: {str(e)}")
            return []

    def _extract_with_easyocr(self, image: Image.Image) -> List[Dict]:
        """Extract text using EasyOCR"""
        if not self.easyocr_reader:
            return []
            
        try:
            result = self.easyocr_reader.readtext(np.array(image))
            boxes = []
            
            for detection in result:
                # Convert EasyOCR box format to [x1, y1, x2, y2]
                box_points = detection[0]
                x1 = min(point[0] for point in box_points)
                y1 = min(point[1] for point in box_points)
                x2 = max(point[0] for point in box_points)
                y2 = max(point[1] for point in box_points)
                
                box = {
                    'text': detection[1],
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(detection[2]),
                    'source': 'easyocr'
                }
                boxes.append(box)
                
            return boxes
            
        except Exception as e:
            st.warning(f"EasyOCR extraction warning: {str(e)}")
            return []

    def _combine_results(self, doctr_boxes: List[Dict], 
                        easyocr_boxes: List[Dict]) -> List[Dict]:
        """Combine and deduplicate results from both OCR engines"""
        all_boxes = []
        used_boxes = set()
        
        # Helper function to create box key
        def get_box_key(box):
            return f"{box['box'][0]},{box['box'][1]},{box['box'][2]},{box['box'][3]}"
        
        # Add DocTR results
        for box in doctr_boxes:
            box_key = get_box_key(box)
            if box_key not in used_boxes:
                used_boxes.add(box_key)
                all_boxes.append(box)
        
        # Add unique EasyOCR results
        for box in easyocr_boxes:
            box_key = get_box_key(box)
            if box_key not in used_boxes:
                used_boxes.add(box_key)
                all_boxes.append(box)
        
        return all_boxes

    def _detect_document_type(self, text: str) -> Optional[str]:
        """Detect document type based on content"""
        # Define keywords for each document type
        type_keywords = {
            'invoice': ['invoice', 'bill to', 'payment terms', 'qty', 'quantity'],
            'bank_statement': ['account statement', 'balance', 'withdrawal', 'deposit'],
            'payslip': ['pay slip', 'earnings', 'deductions', 'net pay'],
            'expense_report': ['expense', 'reimbursement', 'receipt', 'claim']
        }
        
        # Count keywords for each type
        type_scores = {doc_type: 0 for doc_type in type_keywords}
        
        for doc_type, keywords in type_keywords.items():
            for keyword in keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    type_scores[doc_type] += 1
        
        # Get type with highest score
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0] if best_type[1] > 0 else None

    def _process_by_type(self, doc_type: str, text: str, 
                        boxes: List[Dict],
                        confidence_threshold: float) -> Dict:
        """Process document based on its type"""
        if not doc_type:
            return self._process_generic(text, boxes, confidence_threshold)
            
        processor = getattr(
            self,
            f'_process_{doc_type}',
            self._process_generic
        )
        
        results = processor(text, boxes, confidence_threshold)
        results['document_type'] = doc_type
        results['processing_date'] = datetime.now().isoformat()
        
        # Validate results
        validation_results = self.validator.validate_document(results)
        results.update(validation_results)
        
        return results


    def _process_generic(self, text: str, boxes: List[Dict],
                            confidence_threshold: float) -> Dict:
            """Generic document processing with field extraction"""
            results = {
                'text': text,
                'boxes': [box for box in boxes if box['confidence'] >= confidence_threshold],
                'confidence_threshold': confidence_threshold,
                'fields': {},
                'validation_results': {},
                'warnings': []
            }
            
            # Extract common fields
            self._extract_common_fields(text, boxes, results, confidence_threshold)
            
            return results

    def _process_invoice(self, text: str, boxes: List[Dict],
                        confidence_threshold: float) -> Dict:
        """Process invoice documents"""
        results = self._process_generic(text, boxes, confidence_threshold)
        
        # Invoice-specific patterns
        patterns = {
            'invoice_number': r'(?i)invoice\s*(?:#|number|num|no)?[\s:]*([A-Z0-9-]+)',
            'invoice_date': r'(?i)(?:invoice\s*date|date)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            'due_date': r'(?i)(?:due\s*date|payment\s*due)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            'total_amount': r'(?i)(?:total|amount\s*due|grand\s*total)[\s:]*[$£€]?\s*([\d,]+\.?\d{0,2})',
            'tax_amount': r'(?i)(?:tax|vat|gst)[\s:]*[$£€]?\s*([\d,]+\.?\d{0,2})',
            'subtotal': r'(?i)(?:subtotal|net\s*amount)[\s:]*[$£€]?\s*([\d,]+\.?\d{0,2})'
        }
        
        # Extract invoice-specific fields
        for field, pattern in patterns.items():
            matches = re.search(pattern, text)
            if matches:
                value = matches.group(1).strip()
                results['fields'][field] = value
        
        # Extract line items if present
        line_items = self._extract_line_items(text, boxes)
        if line_items:
            results['fields']['line_items'] = line_items
        
        return results

    def _process_bank_statement(self, text: str, boxes: List[Dict],
                              confidence_threshold: float) -> Dict:
        """Process bank statement documents"""
        results = self._process_generic(text, boxes, confidence_threshold)
        
        # Bank statement patterns
        patterns = {
            'account_number': r'(?i)account\s*(?:#|number|no)[\s:]*(\d{8,12})',
            'statement_period': r'(?i)statement\s*period[\s:]*([^\n]+)',
            'opening_balance': r'(?i)(?:opening|previous)\s*balance[\s:]*[$£€]?\s*([\d,]+\.?\d{2})',
            'closing_balance': r'(?i)(?:closing|ending)\s*balance[\s:]*[$£€]?\s*([\d,]+\.?\d{2})',
            'total_deposits': r'(?i)total\s*deposits[\s:]*[$£€]?\s*([\d,]+\.?\d{2})',
            'total_withdrawals': r'(?i)total\s*withdrawals[\s:]*[$£€]?\s*([\d,]+\.?\d{2})'
        }
        
        # Extract bank statement fields
        for field, pattern in patterns.items():
            matches = re.search(pattern, text)
            if matches:
                value = matches.group(1).strip()
                results['fields'][field] = value
        
        # Extract transactions
        transactions = self._extract_transactions(text, boxes)
        if transactions:
            results['fields']['transactions'] = transactions
        
        return results

    def _process_payslip(self, text: str, boxes: List[Dict],
                        confidence_threshold: float) -> Dict:
        """Process payslip documents"""
        results = self._process_generic(text, boxes, confidence_threshold)
        
        # Payslip patterns
        patterns = {
            'employee_id': r'(?i)employee\s*(?:id|number|no)[\s:]*([A-Z0-9]+)',
            'pay_period': r'(?i)pay\s*period[\s:]*([^\n]+)',
            'gross_pay': r'(?i)gross\s*pay[\s:]*[$£€]?\s*([\d,]+\.?\d{2})',
            'net_pay': r'(?i)net\s*pay[\s:]*[$£€]?\s*([\d,]+\.?\d{2})',
            'total_deductions': r'(?i)total\s*deductions[\s:]*[$£€]?\s*([\d,]+\.?\d{2})'
        }
        
        # Extract payslip fields
        for field, pattern in patterns.items():
            matches = re.search(pattern, text)
            if matches:
                value = matches.group(1).strip()
                results['fields'][field] = value
        
        # Extract earnings and deductions
        earnings = self._extract_earnings(text, boxes)
        deductions = self._extract_deductions(text, boxes)
        
        if earnings:
            results['fields']['earnings'] = earnings
        if deductions:
            results['fields']['deductions'] = deductions
        
        return results

    def _extract_common_fields(self, text: str, boxes: List[Dict], 
                             results: Dict, confidence_threshold: float):
        """Extract common fields found in most documents"""
        # Common patterns
        patterns = {
            'date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\+?1?[-.]?\d{3}[-.)]\d{3}[-.]?\d{4}\b',
            'address': r'\b\d+\s+[A-Za-z0-9\s,]+\b[A-Za-z]{2}\b\s+\d{5}\b'
        }
        
        for field, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                value = match.group(0)
                # Find corresponding box
                box = self._find_box_for_text(value, boxes)
                if box and box['confidence'] >= confidence_threshold:
                    results['fields'][field] = value
                    break

    def _extract_line_items(self, text: str, boxes: List[Dict]) -> List[Dict]:
        """Extract line items from invoice"""
        line_items = []
        
        # Try to find table structure
        table_region = self._detect_table_region(boxes)
        if table_region:
            # Process boxes within table region
            table_boxes = [
                box for box in boxes 
                if self._is_box_in_region(box, table_region)
            ]
            
            # Group boxes into rows
            rows = self._group_boxes_into_rows(table_boxes)
            
            # Process each row
            for row in rows:
                item = self._process_line_item_row(row)
                if item:
                    line_items.append(item)
        
        return line_items

    def _extract_transactions(self, text: str, boxes: List[Dict]) -> List[Dict]:
        """Extract transactions from bank statement"""
        transactions = []
        
        # Find transaction table region
        table_region = self._detect_table_region(boxes)
        if table_region:
            table_boxes = [
                box for box in boxes 
                if self._is_box_in_region(box, table_region)
            ]
            
            # Group boxes into rows
            rows = self._group_boxes_into_rows(table_boxes)
            
            # Process each row
            for row in rows:
                transaction = self._process_transaction_row(row)
                if transaction:
                    transactions.append(transaction)
        
        return transactions

    def _detect_table_region(self, boxes: List[Dict]) -> Optional[Dict]:
        """Detect table region in document"""
        # Find potential table headers
        header_patterns = [
            r'(?i)description',
            r'(?i)amount',
            r'(?i)quantity',
            r'(?i)price',
            r'(?i)date'
        ]
        
        header_boxes = []
        for box in boxes:
            if any(re.search(pattern, box['text'], re.IGNORECASE) 
                   for pattern in header_patterns):
                header_boxes.append(box)
        
        if len(header_boxes) >= 2:
            # Calculate table region
            x1 = min(box['box'][0] for box in header_boxes)
            y1 = min(box['box'][1] for box in header_boxes)
            x2 = max(box['box'][2] for box in header_boxes)
            y2 = max(box['box'][3] for box in header_boxes)
            
            # Extend region downwards
            max_y = max(box['box'][3] for box in boxes)
            
            return {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': max_y
            }
        
        return None

    def _is_box_in_region(self, box: Dict, region: Dict) -> bool:
        """Check if box is within region"""
        return (box['box'][0] >= region['x1'] and
                box['box'][1] >= region['y1'] and
                box['box'][2] <= region['x2'] and
                box['box'][3] <= region['y2'])

    def _group_boxes_into_rows(self, boxes: List[Dict]) -> List[List[Dict]]:
        """Group boxes into rows based on vertical position"""
        # Sort boxes by y-coordinate
        sorted_boxes = sorted(boxes, key=lambda x: x['box'][1])
        
        rows = []
        current_row = []
        current_y = None
        
        for box in sorted_boxes:
            y = box['box'][1]
            
            if current_y is None:
                current_y = y
                current_row.append(box)
            elif abs(y - current_y) < 20:  # Adjust threshold as needed
                current_row.append(box)
            else:
                if current_row:
                    # Sort boxes in row by x-coordinate
                    current_row.sort(key=lambda x: x['box'][0])
                    rows.append(current_row)
                current_row = [box]
                current_y = y
        
        if current_row:
            current_row.sort(key=lambda x: x['box'][0])
            rows.append(current_row)
        
        return rows

    def _find_box_for_text(self, text: str, boxes: List[Dict]) -> Optional[Dict]:
        """Find box containing exact text"""
        for box in boxes:
            if box['text'].strip() == text.strip():
                return box
        return None