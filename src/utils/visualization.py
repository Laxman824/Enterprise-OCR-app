import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import json
from datetime import datetime
import io
from typing import Dict, List, Optional
from PIL import Image

class DocumentVisualizer:
    def __init__(self):
        self.colors = {
            'header': '#1f77b4',      # Blue
            'amount': '#2ca02c',      # Green
            'date': '#d62728',        # Red
            'table': '#9467bd',       # Purple
            'text': '#7f7f7f',        # Gray
            'key_field': '#ff7f0e'    # Orange
        }

    def create_interactive_display(self, image: Image.Image, 
                                 results: Dict, boxes: List[Dict]):
        """Create interactive display with bounding boxes and tooltips"""
        col1, col2 = st.columns([2, 1])

        with col1:
            # Display image with bounding boxes
            fig, ax = self._create_annotated_image(image, boxes, results)
            st.pyplot(fig)

            # Add interactive hover functionality
            self._add_hover_functionality(boxes, results)

        with col2:
            self._display_field_details(results)

    def _create_annotated_image(self, image: Image.Image, 
                              boxes: List[Dict], results: Dict):
        """Create annotated image with bounding boxes"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)

        for box in boxes:
            self._draw_box(ax, box, results)

        ax.axis('off')
        return fig

    def _draw_box(self, ax, box: Dict, results: Dict):
        """Draw a single bounding box with proper styling"""
        x1, y1, x2, y2 = box['box']
        field_type = box.get('field_type', 'text')
        confidence = box.get('confidence', 0)

        # Get color based on field type
        color = self.colors.get(field_type, self.colors['text'])
        
        # Adjust alpha based on confidence
        alpha = min(0.9, max(0.3, confidence))

        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=alpha
        )
        ax.add_patch(rect)

        # Add label if it's a key field
        if box.get('is_key_field'):
            ax.text(
                x1, y1-5,
                box.get('field_name', ''),
                fontsize=8,
                color=color,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    def _display_field_details(self, results: Dict):
        """Display extracted field details"""
        st.subheader("Extracted Information")

        # Group fields by category
        grouped_fields = {}
        for field_name, value in results['fields'].items():
            category = results.get('field_categories', {}).get(field_name, 'Other')
            if category not in grouped_fields:
                grouped_fields[category] = []
            grouped_fields[category].append((field_name, value))

        # Display fields by category
        for category, fields in grouped_fields.items():
            with st.expander(category, expanded=True):
                for field_name, value in fields:
                    confidence = results['confidence_scores'].get(field_name, 0)
                    is_valid = results['validation_results'].get(field_name, False)

                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.text(field_name)
                    with col2:
                        st.text(value)
                    with col3:
                        color = self._get_confidence_color(confidence)
                        st.markdown(
                            f'<div style="background-color: {color}; '
                            f'padding: 5px; border-radius: 5px; '
                            f'text-align: center;">{confidence:.1%}</div>',
                            unsafe_allow_html=True
                        )

    def create_summary_view(self, results: Dict):
        """Create summary view of extraction results"""
        st.subheader("Extraction Summary")

        # Create metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_fields = len(results['fields'])
            st.metric("Total Fields", total_fields)

        with col2:
            valid_fields = sum(results['validation_results'].values())
            st.metric("Valid Fields", f"{valid_fields}/{total_fields}")

        with col3:
            avg_confidence = np.mean(list(results['confidence_scores'].values()))
            st.metric("Average Confidence", f"{avg_confidence:.1%}")

        # Create confidence distribution chart
        fig = self._create_confidence_chart(results)
        st.pyplot(fig)

    def _create_confidence_chart(self, results: Dict):
        """Create confidence distribution chart"""
        fields = list(results['confidence_scores'].keys())
        scores = list(results['confidence_scores'].values())

        fig, ax = plt.subplots(figsize=(10, max(4, len(fields)*0.3)))
        
        # Create horizontal bars
        y_pos = np.arange(len(fields))
        bars = ax.barh(y_pos, scores)

        # Color bars based on confidence
        for i, bar in enumerate(bars):
            bar.set_color(self._get_confidence_color(scores[i]))

        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fields)
        ax.set_xlabel('Confidence Score')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(scores):
            ax.text(v, i, f'{v:.1%}', va='center')

        plt.tight_layout()
        return fig

    @staticmethod
    def _get_confidence_color(confidence: float) -> str:
        """Get color based on confidence score"""
        if confidence >= 0.8:
            return '#28a745'  # Green
        elif confidence >= 0.6:
            return '#ffc107'  # Yellow
        else:
            return '#dc3545'  # Red

    def create_export_buttons(self, results: Dict, filename: str):
        """Create download buttons for different formats"""
        st.subheader("Export Results")

        # JSON Export
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"{filename}_results.json",
            mime="application/json"
        )

        # Create DataFrame for Excel/CSV
        df = pd.DataFrame({
            'Field': results['fields'].keys(),
            'Value': results['fields'].values(),
            'Confidence': [results['confidence_scores'].get(k, 0) 
                         for k in results['fields'].keys()],
            'Valid': [results['validation_results'].get(k, False) 
                     for k in results['fields'].keys()]
        })

        # Excel Export
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            if results.get('table_data'):
                pd.DataFrame(results['table_data']).to_excel(
                    writer, 
                    sheet_name='Table Data',
                    index=False
                )

        st.download_button(
            label="Download Excel",
            data=buffer,
            file_name=f"{filename}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )