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
class DocumentTemplate:
    name: str
    description: str
    fields: Dict[str, DocumentField]
    key_identifiers: List[str]
    validation_rules: Dict[str, List[str]]
    sample_formats: List[str]

# Define document templates for different types
DOCUMENT_TEMPLATES = {
    "invoice": DocumentTemplate(
        name="Invoice",
        description="Commercial invoices and bills",
        key_identifiers=["invoice", "bill to", "amount due", "tax", "total"],
        fields={
            "invoice_number": DocumentField(
                name="Invoice Number",
                pattern=r"(?i)(?:invoice|bill)\s*(?:#|number|num|no)?[:.]?\s*([A-Z0-9-]+)",
                description="Unique invoice identifier",
                example="INV-12345",
                validation_type="invoice_number",
                category="Header",
                is_key_field=True
            ),
            "date": DocumentField(
                name="Invoice Date",
                pattern=r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",
                description="Invoice issue date",
                example="01/01/2024",
                validation_type="date",
                category="Header",
                is_key_field=True
            ),
            "total_amount": DocumentField(
                name="Total Amount",
                pattern=r"\$?\s*\d{1,3}(?:,\d{3})*\.\d{2}",
                description="Total invoice amount",
                example="$1,234.56",
                validation_type="amount",
                category="Summary",
                is_key_field=True
            )
        },
        validation_rules={
            "date": ["valid_date", "not_future"],
            "total_amount": ["positive_amount", "matches_items"]
        },
        sample_formats=["pdf", "png", "jpg"]
    ),
    
    "bank_statement": DocumentTemplate(
        name="Bank Statement",
        description="Bank account statements",
        key_identifiers=["account statement", "balance", "withdrawal", "deposit"],
        fields={
            "account_number": DocumentField(
                name="Account Number",
                pattern=r"\b\d{8,12}\b",
                description="Bank account number",
                example="1234567890",
                validation_type="account_number",
                category="Header",
                is_key_field=True
            ),
            "statement_period": DocumentField(
                name="Statement Period",
                pattern=r"(?i)statement period:?\s*(.*)",
                description="Statement period",
                example="Jan 1 - Jan 31, 2024",
                validation_type="date_range",
                category="Header",
                is_key_field=True
            )
        },
        validation_rules={
            "balance": ["valid_amount", "matches_calculations"]
        },
        sample_formats=["pdf", "png", "jpg"]
    ),
    
    "pay_slip": DocumentTemplate(
        name="Pay Slip",
        description="Employee payment slips",
        key_identifiers=["salary", "pay slip", "earnings", "deductions"],
        fields={
            "employee_id": DocumentField(
                name="Employee ID",
                pattern=r"(?i)emp(?:loyee)?\s*(?:id|number|no)[:.]?\s*([A-Z0-9]+)",
                description="Employee identification number",
                example="EMP123",
                validation_type="employee_id",
                category="Header",
                is_key_field=True
            ),
            "net_pay": DocumentField(
                name="Net Pay",
                pattern=r"\$?\s*\d{1,3}(?:,\d{3})*\.\d{2}",
                description="Net payment amount",
                example="$3,500.00",
                validation_type="amount",
                category="Summary",
                is_key_field=True
            )
        },
        validation_rules={
            "net_pay": ["valid_amount", "matches_calculations"]
        },
        sample_formats=["pdf", "png", "jpg"]
    ),
    
    "expense_report": DocumentTemplate(
        name="Expense Report",
        description="Business expense reports",
        key_identifiers=["expense", "reimbursement", "receipt"],
        fields={
            "report_id": DocumentField(
                name="Report ID",
                pattern=r"(?i)report\s*(?:id|number|no)[:.]?\s*([A-Z0-9-]+)",
                description="Expense report identifier",
                example="EXP-2024-001",
                validation_type="report_id",
                category="Header",
                is_key_field=True
            ),
            "total_expenses": DocumentField(
                name="Total Expenses",
                pattern=r"\$?\s*\d{1,3}(?:,\d{3})*\.\d{2}",
                description="Total expense amount",
                example="$1,234.56",
                validation_type="amount",
                category="Summary",
                is_key_field=True
            )
        },
        validation_rules={
            "total_expenses": ["valid_amount", "matches_items"]
        },
        sample_formats=["pdf", "png", "jpg"]
    )
}
