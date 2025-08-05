"""
Nynava - Data Anonymization Module
HIPAA-compliant medical data anonymization and de-identification
"""

import os
import re
import hashlib
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
import PyPDF2
from PIL import Image
import pydicom
import structlog

logger = structlog.get_logger()

class DataAnonymizer:
    """HIPAA-compliant data anonymization engine"""
    
    def __init__(self):
        self.fake = Faker()
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # HIPAA Safe Harbor identifiers to remove/anonymize
        self.phi_patterns = {
            'names': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'addresses': r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'mrn': r'\b(?:MRN|Medical Record Number)[\s:]*(\d+)\b',
            'account_numbers': r'\b(?:Account|Acct)[\s#:]*(\d+)\b',
            'license_numbers': r'\b[A-Z]{1,2}\d{6,8}\b',
            'device_identifiers': r'\b[A-F0-9]{12,}\b',
            'urls': r'https?://[^\s]+',
            'ip_addresses': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }
        
        # Age buckets for HIPAA compliance (ages > 89 become 90+)
        self.age_buckets = {
            (0, 17): '0-17',
            (18, 29): '18-29',
            (30, 39): '30-39',
            (40, 49): '40-49',
            (50, 59): '50-59',
            (60, 69): '60-69',
            (70, 79): '70-79',
            (80, 89): '80-89',
            (90, 150): '90+'
        }
        
        logger.info("DataAnonymizer initialized")
    
    def anonymize_file(self, file_obj) -> Dict[str, Any]:
        """Anonymize uploaded file based on type"""
        try:
            file_extension = file_obj.filename.split('.')[-1].lower()
            content_type = file_obj.content_type
            
            logger.info("Anonymizing file", filename=file_obj.filename, type=content_type)
            
            if content_type == 'application/pdf':
                return self._anonymize_pdf(file_obj)
            elif content_type.startswith('image/'):
                return self._anonymize_image(file_obj)
            elif file_extension == 'dcm' or 'dicom' in content_type:
                return self._anonymize_dicom(file_obj)
            elif content_type == 'text/csv' or file_extension == 'csv':
                return self._anonymize_csv(file_obj)
            elif content_type == 'application/json':
                return self._anonymize_json(file_obj)
            else:
                return self._anonymize_text(file_obj)
                
        except Exception as e:
            logger.error("File anonymization failed", error=str(e))
            raise
    
    def _anonymize_pdf(self, file_obj) -> Dict[str, Any]:
        """Anonymize PDF documents"""
        try:
            reader = PyPDF2.PdfReader(file_obj)
            text_content = ""
            
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
            
            # Anonymize extracted text
            anonymized_text = self._anonymize_text_content(text_content)
            
            return {
                'type': 'pdf',
                'original_filename': file_obj.filename,
                'anonymized_content': anonymized_text,
                'page_count': len(reader.pages),
                'anonymization_timestamp': datetime.utcnow().isoformat(),
                'phi_removed': True
            }
            
        except Exception as e:
            logger.error("PDF anonymization failed", error=str(e))
            raise
    
    def _anonymize_image(self, file_obj) -> Dict[str, Any]:
        """Anonymize medical images (remove EXIF, blur text regions)"""
        try:
            image = Image.open(file_obj)
            
            # Remove EXIF data
            if hasattr(image, '_getexif'):
                image_without_exif = Image.new(image.mode, image.size)
                image_without_exif.putdata(list(image.getdata()))
                image = image_without_exif
            
            # Generate anonymized metadata
            width, height = image.size
            
            return {
                'type': 'image',
                'original_filename': file_obj.filename,
                'format': image.format,
                'dimensions': {'width': width, 'height': height},
                'mode': image.mode,
                'anonymized_image_data': self._image_to_bytes(image),
                'anonymization_timestamp': datetime.utcnow().isoformat(),
                'exif_removed': True
            }
            
        except Exception as e:
            logger.error("Image anonymization failed", error=str(e))
            raise
    
    def _anonymize_dicom(self, file_obj) -> Dict[str, Any]:
        """Anonymize DICOM medical images"""
        try:
            ds = pydicom.dcmread(file_obj)
            
            # Remove patient identifiers as per DICOM standard
            phi_tags = [
                'PatientName', 'PatientID', 'PatientBirthDate',
                'PatientSex', 'PatientAge', 'PatientAddress',
                'PatientTelephoneNumbers', 'InstitutionName',
                'InstitutionAddress', 'ReferringPhysicianName',
                'PerformingPhysicianName', 'OperatorsName',
                'StudyDate', 'SeriesDate', 'AcquisitionDate',
                'StudyTime', 'SeriesTime', 'AcquisitionTime'
            ]
            
            anonymized_tags = {}
            for tag in phi_tags:
                if hasattr(ds, tag):
                    original_value = getattr(ds, tag)
                    anonymized_tags[tag] = str(original_value)
                    
                    # Replace with anonymized values
                    if 'Name' in tag:
                        setattr(ds, tag, 'ANONYMIZED')
                    elif 'ID' in tag:
                        setattr(ds, tag, self._generate_anonymous_id())
                    elif 'Date' in tag:
                        setattr(ds, tag, self._anonymize_date(str(original_value)))
                    elif 'Address' in tag or 'Telephone' in tag:
                        setattr(ds, tag, 'REMOVED')
                    else:
                        setattr(ds, tag, 'ANONYMIZED')
            
            # Anonymize age if > 89
            if hasattr(ds, 'PatientAge'):
                age_str = str(ds.PatientAge)
                age_match = re.search(r'(\d+)', age_str)
                if age_match:
                    age = int(age_match.group(1))
                    if age > 89:
                        ds.PatientAge = '90+'
            
            return {
                'type': 'dicom',
                'original_filename': file_obj.filename,
                'study_description': getattr(ds, 'StudyDescription', 'Unknown'),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'anonymized_dicom_data': ds.to_json(),
                'removed_phi_tags': list(anonymized_tags.keys()),
                'anonymization_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("DICOM anonymization failed", error=str(e))
            raise
    
    def _anonymize_csv(self, file_obj) -> Dict[str, Any]:
        """Anonymize CSV data files"""
        try:
            df = pd.read_csv(file_obj)
            original_shape = df.shape
            
            # Identify potential PHI columns
            phi_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if any(phi_term in col_lower for phi_term in 
                      ['name', 'address', 'phone', 'email', 'ssn', 'id', 'mrn']):
                    phi_columns.append(col)
            
            # Anonymize identified PHI columns
            for col in phi_columns:
                if 'name' in col.lower():
                    df[col] = df[col].apply(lambda x: self.fake.name() if pd.notna(x) else x)
                elif 'email' in col.lower():
                    df[col] = df[col].apply(lambda x: self.fake.email() if pd.notna(x) else x)
                elif 'phone' in col.lower():
                    df[col] = df[col].apply(lambda x: self.fake.phone_number() if pd.notna(x) else x)
                elif 'address' in col.lower():
                    df[col] = df[col].apply(lambda x: self.fake.address() if pd.notna(x) else x)
                else:
                    # Generic anonymization for other PHI
                    df[col] = df[col].apply(lambda x: self._generate_anonymous_id() if pd.notna(x) else x)
            
            # Anonymize age columns
            age_columns = [col for col in df.columns if 'age' in col.lower()]
            for col in age_columns:
                df[col] = df[col].apply(self._anonymize_age)
            
            # Anonymize date columns
            date_columns = [col for col in df.columns if any(date_term in col.lower() 
                           for date_term in ['date', 'birth', 'dob'])]
            for col in date_columns:
                df[col] = df[col].apply(self._anonymize_date)
            
            return {
                'type': 'csv',
                'original_filename': file_obj.filename,
                'original_shape': original_shape,
                'anonymized_data': df.to_dict('records'),
                'anonymized_columns': phi_columns + age_columns + date_columns,
                'anonymization_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("CSV anonymization failed", error=str(e))
            raise
    
    def _anonymize_json(self, file_obj) -> Dict[str, Any]:
        """Anonymize JSON data"""
        try:
            data = json.load(file_obj)
            anonymized_data = self._anonymize_json_recursive(data)
            
            return {
                'type': 'json',
                'original_filename': file_obj.filename,
                'anonymized_data': anonymized_data,
                'anonymization_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("JSON anonymization failed", error=str(e))
            raise
    
    def _anonymize_text(self, file_obj) -> Dict[str, Any]:
        """Anonymize plain text files"""
        try:
            content = file_obj.read().decode('utf-8')
            anonymized_content = self._anonymize_text_content(content)
            
            return {
                'type': 'text',
                'original_filename': file_obj.filename,
                'anonymized_content': anonymized_content,
                'anonymization_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Text anonymization failed", error=str(e))
            raise
    
    def _anonymize_text_content(self, text: str) -> str:
        """Anonymize text content using Presidio and custom patterns"""
        try:
            # Use Presidio for initial PHI detection
            analyzer_results = self.analyzer.analyze(
                text=text,
                entities=['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'US_SSN', 
                         'DATE_TIME', 'LOCATION', 'US_DRIVER_LICENSE'],
                language='en'
            )
            
            # Anonymize using Presidio
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results
            )
            
            anonymized_text = anonymized_result.text
            
            # Apply custom PHI patterns
            for pattern_name, pattern in self.phi_patterns.items():
                anonymized_text = re.sub(pattern, f'[{pattern_name.upper()}_REMOVED]', anonymized_text)
            
            return anonymized_text
            
        except Exception as e:
            logger.error("Text content anonymization failed", error=str(e))
            return text  # Return original if anonymization fails
    
    def _anonymize_json_recursive(self, obj: Any) -> Any:
        """Recursively anonymize JSON objects"""
        if isinstance(obj, dict):
            anonymized = {}
            for key, value in obj.items():
                key_lower = key.lower()
                
                # Check if key suggests PHI
                if any(phi_term in key_lower for phi_term in 
                      ['name', 'email', 'phone', 'address', 'ssn', 'id']):
                    if isinstance(value, str):
                        if 'name' in key_lower:
                            anonymized[key] = self.fake.name()
                        elif 'email' in key_lower:
                            anonymized[key] = self.fake.email()
                        elif 'phone' in key_lower:
                            anonymized[key] = self.fake.phone_number()
                        elif 'address' in key_lower:
                            anonymized[key] = self.fake.address()
                        else:
                            anonymized[key] = self._generate_anonymous_id()
                    else:
                        anonymized[key] = self._generate_anonymous_id()
                else:
                    anonymized[key] = self._anonymize_json_recursive(value)
            return anonymized
        elif isinstance(obj, list):
            return [self._anonymize_json_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return self._anonymize_text_content(obj)
        else:
            return obj
    
    def _anonymize_age(self, age: Union[int, str, float]) -> str:
        """Anonymize age according to HIPAA Safe Harbor rules"""
        try:
            if pd.isna(age):
                return age
            
            age_int = int(float(age))
            
            for (min_age, max_age), bucket in self.age_buckets.items():
                if min_age <= age_int <= max_age:
                    return bucket
            
            return '90+'  # Default for very old ages
            
        except (ValueError, TypeError):
            return str(age)  # Return as-is if can't convert
    
    def _anonymize_date(self, date_str: str) -> str:
        """Anonymize dates by shifting them randomly"""
        try:
            if pd.isna(date_str) or not date_str:
                return date_str
            
            # Try different date formats
            date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(str(date_str), fmt)
                    # Shift date by random amount (±180 days)
                    shift_days = np.random.randint(-180, 181)
                    shifted_date = date_obj + timedelta(days=shift_days)
                    return shifted_date.strftime(fmt)
                except ValueError:
                    continue
            
            # If no format matches, return year only
            year_match = re.search(r'(\d{4})', str(date_str))
            if year_match:
                return year_match.group(1)
            
            return '[DATE_REMOVED]'
            
        except Exception:
            return '[DATE_REMOVED]'
    
    def _generate_anonymous_id(self) -> str:
        """Generate anonymous identifier"""
        return hashlib.sha256(
            f"{self.fake.uuid4()}{datetime.utcnow()}".encode()
        ).hexdigest()[:12].upper()
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes"""
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def validate_anonymization(self, original_data: Any, anonymized_data: Any) -> Dict[str, Any]:
        """Validate that anonymization was successful"""
        try:
            validation_results = {
                'phi_detected': False,
                'anonymization_quality': 'high',
                'issues': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Convert anonymized data to string for analysis
            if isinstance(anonymized_data, dict):
                text_to_check = json.dumps(anonymized_data)
            else:
                text_to_check = str(anonymized_data)
            
            # Check for remaining PHI patterns
            for pattern_name, pattern in self.phi_patterns.items():
                matches = re.findall(pattern, text_to_check)
                if matches:
                    validation_results['phi_detected'] = True
                    validation_results['issues'].append(f"Potential {pattern_name} found: {len(matches)} instances")
            
            # Use Presidio for additional validation
            analyzer_results = self.analyzer.analyze(
                text=text_to_check,
                entities=['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'US_SSN'],
                language='en'
            )
            
            if analyzer_results:
                validation_results['phi_detected'] = True
                validation_results['issues'].extend([
                    f"Presidio detected {result.entity_type} (confidence: {result.score:.2f})"
                    for result in analyzer_results
                ])
            
            # Set quality based on findings
            if validation_results['phi_detected']:
                validation_results['anonymization_quality'] = 'medium' if len(validation_results['issues']) < 3 else 'low'
            
            return validation_results
            
        except Exception as e:
            logger.error("Anonymization validation failed", error=str(e))
            return {
                'phi_detected': True,
                'anonymization_quality': 'unknown',
                'issues': ['Validation failed'],
                'error': str(e)
            }