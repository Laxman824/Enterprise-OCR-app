from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import pandas as pd
import numpy as np

class DocumentValidator:
    def __init__(self):
        self.validation_rules = {
            'date': {
                'patterns': [
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{2}/\d{2}/\d{4}',
                    r'\d{2}-\d{2}-\d{4}'
                ],
                'formats': ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
            },
            'amount': {
                'pattern': r'^\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?$',
                'decimal_places': 2
            },
            'phone': {
                'pattern': r'^\+?1?\d{10,12}$'
            },
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            }
        }

    def validate_document(self, results: Dict) -> Dict:
        """Validate entire document extraction results"""
        validation_results = {}
        warnings = []

        for field_name, value in results['fields'].items():
            is_valid, message = self.validate_field(
                value,
                field_name,
                results.get('field_types', {}),
                results
            )
            validation_results[field_name] = is_valid
            if not is_valid:
                warnings.append(f"{field_name}: {message}")

        # Add cross-field validations
        cross_field_warnings = self._validate_cross_fields(results)
        warnings.extend(cross_field_warnings)

        return {
            'validation_results': validation_results,
            'warnings': warnings
        }

    def validate_field(self, value: str, field_name: str, 
                      field_types: Dict, context: Dict) -> Tuple[bool, str]:
        """Validate individual field value"""
        if not value:
            return False, "Empty value"

        field_type = field_types.get(field_name, 'text')
        validator = getattr(
            self,
            f'_validate_{field_type}',
            self._validate_text
        )

        return validator(value, context)

    def _validate_date(self, value: str, context: Dict) -> Tuple[bool, str]:
        """Validate date fields"""
        for pattern in self.validation_rules['date']['patterns']:
            if not re.match(pattern, value):
                continue

            for date_format in self.validation_rules['date']['formats']:
                try:
                    date = datetime.strptime(value, date_format)
                    
                    # Check if date is in valid range
                    if context.get('max_date'):
                        max_date = datetime.strptime(
                            context['max_date'],
                            '%Y-%m-%d'
                        )
                        if date > max_date:
                            return False, f"Date cannot be later than {context['max_date']}"

                    if context.get('min_date'):
                        min_date = datetime.strptime(
                            context['min_date'],
                            '%Y-%m-%d'
                        )
                        if date < min_date:
                            return False, f"Date cannot be earlier than {context['min_date']}"

                    return True, ""
                except ValueError:
                    continue

        return False, "Invalid date format"

    def _validate_amount(self, value: str, context: Dict) -> Tuple[bool, str]:
        """Validate amount fields"""
        # Remove currency symbols and commas
        clean_value = value.replace('$', '').replace(',', '').strip()
        
        try:
            amount = float(clean_value)
            
            # Check range if specified
            if context.get('min_amount') is not None:
                if amount < context['min_amount']:
                    return False, f"Amount below minimum: {context['min_amount']}"
                    
            if context.get('max_amount') is not None:
                if amount > context['max_amount']:
                    return False, f"Amount exceeds maximum: {context['max_amount']}"

            # Validate decimal places
            decimal_places = len(clean_value.split('.')[-1]) if '.' in clean_value else 0
            if decimal_places > self.validation_rules['amount']['decimal_places']:
                return False, f"Too many decimal places: {decimal_places}"

            return True, ""
        except ValueError:
            return False, "Invalid amount format"

    def _validate_phone(self, value: str, context: Dict) -> Tuple[bool, str]:
        """Validate phone numbers"""
        # Remove common separators
        clean_value = re.sub(r'[\s\-\(\)]', '', value)
        
        if re.match(self.validation_rules['phone']['pattern'], clean_value):
            return True, ""
        return False, "Invalid phone number format"

    def _validate_email(self, value: str, context: Dict) -> Tuple[bool, str]:
        """Validate email addresses"""
        if re.match(self.validation_rules['email']['pattern'], value):
            return True, ""
        return False, "Invalid email format"

    def _validate_text(self, value: str, context: Dict) -> Tuple[bool, str]:
        """Validate text fields"""
        if context.get('min_length') and len(value) < context['min_length']:
            return False, f"Text too short (min: {context['min_length']})"
            
        if context.get('max_length') and len(value) > context['max_length']:
            return False, f"Text too long (max: {context['max_length']})"
            
        if context.get('pattern') and not re.match(context['pattern'], value):
            return False, "Text doesn't match required pattern"
            
        return True, ""

    def _validate_cross_fields(self, results: Dict) -> List[str]:
        """Validate relationships between fields"""
        warnings = []
        fields = results['fields']

        # Example: Validate date relationships
        if 'start_date' in fields and 'end_date' in fields:
            try:
                start = datetime.strptime(fields['start_date'], '%Y-%m-%d')
                end = datetime.strptime(fields['end_date'], '%Y-%m-%d')
                if end < start:
                    warnings.append("End date is before start date")
            except ValueError:
                pass

        # Example: Validate amount calculations
        if all(key in fields for key in ['subtotal', 'tax', 'total']):
            try:
                subtotal = float(fields['subtotal'].replace('$', '').replace(',', ''))
                tax = float(fields['tax'].replace('$', '').replace(',', ''))
                total = float(fields['total'].replace('$', '').replace(',', ''))
                
                if not np.isclose(subtotal + tax, total, rtol=0.01):
                    warnings.append("Total amount doesn't match subtotal + tax")
            except ValueError:
                pass

        return warnings

    def validate_table_data(self, table_data: List[Dict]) -> Tuple[bool, List[str]]:
            """Validate extracted table data"""
            warnings = []
            is_valid = True

            if not table_data:
                return False, ["No table data found"]

            # Check for required columns
            required_columns = set(['description', 'amount', 'date'])  # Customize based on document type
            
            # Get actual columns from the first row
            if table_data:
                actual_columns = set(table_data[0].keys())
                missing_columns = required_columns - actual_columns
                if missing_columns:
                    warnings.append(f"Missing required columns: {', '.join(missing_columns)}")
                    is_valid = False

            # Validate each row
            for idx, row in enumerate(table_data, 1):
                row_warnings = self._validate_table_row(row, idx)
                if row_warnings:
                    warnings.extend(row_warnings)
                    is_valid = False

            # Validate table totals if present
            total_warnings = self._validate_table_totals(table_data)
            if total_warnings:
                warnings.extend(total_warnings)
                is_valid = False

            return is_valid, warnings

    def _validate_table_row(self, row: Dict, row_number: int) -> List[str]:
        """Validate individual table row"""
        warnings = []

        # Validate amount format
        if 'amount' in row:
            try:
                amount = self._parse_amount(row['amount'])
                if amount <= 0:
                    warnings.append(f"Row {row_number}: Invalid amount value")
            except ValueError:
                warnings.append(f"Row {row_number}: Invalid amount format")

        # Validate date format
        if 'date' in row:
            is_valid, message = self._validate_date(row['date'], {})
            if not is_valid:
                warnings.append(f"Row {row_number}: {message}")

        # Validate description
        if 'description' in row:
            if not row['description'] or len(row['description'].strip()) < 2:
                warnings.append(f"Row {row_number}: Missing or invalid description")

        # Validate quantity if present
        if 'quantity' in row:
            try:
                qty = float(row['quantity'])
                if qty <= 0:
                    warnings.append(f"Row {row_number}: Invalid quantity")
            except ValueError:
                warnings.append(f"Row {row_number}: Invalid quantity format")

        return warnings

    def _validate_table_totals(self, table_data: List[Dict]) -> List[str]:
        """Validate table totals and calculations"""
        warnings = []

        try:
            # Calculate sum of amounts
            total_amount = sum(self._parse_amount(row['amount']) 
                             for row in table_data 
                             if 'amount' in row)

            # If there's a total row, validate it
            total_rows = [row for row in table_data 
                         if 'description' in row and 
                         row['description'].lower().strip() in ['total', 'grand total']]

            if total_rows:
                total_row = total_rows[0]
                stated_total = self._parse_amount(total_row['amount'])
                
                # Compare calculated total with stated total
                if not np.isclose(total_amount, stated_total, rtol=0.01):
                    warnings.append(
                        f"Table total mismatch: calculated={total_amount:.2f}, "
                        f"stated={stated_total:.2f}"
                    )

            # Validate subtotals if present
            subtotal_warnings = self._validate_subtotals(table_data)
            warnings.extend(subtotal_warnings)

        except Exception as e:
            warnings.append(f"Error validating table totals: {str(e)}")

        return warnings

    def _validate_subtotals(self, table_data: List[Dict]) -> List[str]:
        """Validate subtotals within table data"""
        warnings = []
        current_subtotal = 0
        
        for row in table_data:
            description = row.get('description', '').lower().strip()
            
            if 'subtotal' in description:
                stated_subtotal = self._parse_amount(row['amount'])
                if not np.isclose(current_subtotal, stated_subtotal, rtol=0.01):
                    warnings.append(
                        f"Subtotal mismatch: calculated={current_subtotal:.2f}, "
                        f"stated={stated_subtotal:.2f}"
                    )
                current_subtotal = 0
            else:
                current_subtotal += self._parse_amount(row.get('amount', '0'))

        return warnings

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float"""
        if isinstance(amount_str, (int, float)):
            return float(amount_str)
            
        # Remove currency symbols and special characters
        clean_amount = re.sub(r'[^\d.-]', '', str(amount_str))
        
        try:
            return float(clean_amount)
        except ValueError:
            raise ValueError(f"Invalid amount format: {amount_str}")

    def validate_line_items(self, line_items: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate line items in an invoice or similar document"""
        warnings = []
        is_valid = True

        if not line_items:
            return False, ["No line items found"]

        # Track totals for validation
        calculated_total = 0
        calculated_tax = 0

        for idx, item in enumerate(line_items, 1):
            # Validate required fields
            required_fields = ['description', 'quantity', 'unit_price', 'amount']
            missing_fields = [field for field in required_fields if field not in item]
            
            if missing_fields:
                warnings.append(f"Line item {idx}: Missing fields: {', '.join(missing_fields)}")
                is_valid = False
                continue

            try:
                # Parse numeric values
                quantity = float(item['quantity'])
                unit_price = self._parse_amount(item['unit_price'])
                amount = self._parse_amount(item['amount'])

                # Validate calculations
                expected_amount = quantity * unit_price
                if not np.isclose(expected_amount, amount, rtol=0.01):
                    warnings.append(
                        f"Line item {idx}: Amount mismatch - expected {expected_amount:.2f}, "
                        f"got {amount:.2f}"
                    )
                    is_valid = False

                calculated_total += amount

                # Track tax if present
                if 'tax_amount' in item:
                    tax_amount = self._parse_amount(item['tax_amount'])
                    calculated_tax += tax_amount

            except ValueError as e:
                warnings.append(f"Line item {idx}: {str(e)}")
                is_valid = False

        return is_valid, warnings