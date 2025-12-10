import logging
import os
from typing import Dict, Any
from . import ocr
from . import pdf_processing
from . import text_preprocessing
from ..cv_refinement.keyword_extraction import extract_keywords_from_resume

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF by first converting to images and then using OCR."""
    try:
        # Create directories for images and OCR output
        img_dir = "pages_img"
        ocr_dir = "ocr_output"
        os.makedirs(ocr_dir, exist_ok=True)
        
        # Convert PDF to images and apply OCR
        pdf_processing.convert_pdf_to_img(pdf_path, img_dir=img_dir)
        return ocr.applyOCR(img_dir, ocr_dir)
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

def process_resume(file_path: str) -> tuple[str, dict]:
    """
    Main pipeline function to process a resume file.
    
    Args:
        file_path: Path to the resume file (PDF or image)
        
    Returns:
        Tuple of (processed_text, parsed_output) where:
        - processed_text: The cleaned text from the resume
        - parsed_output: Dictionary containing extracted information
    """
    try:
        # Step 1: Extract text from the file
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            # Assume it's an image and use OCR
            # Create a temporary directory for the image
            temp_img_dir = "temp_img"
            temp_ocr_dir = "temp_ocr"
            os.makedirs(temp_img_dir, exist_ok=True)
            os.makedirs(temp_ocr_dir, exist_ok=True)
            
            # Copy the image to the temp directory
            import shutil
            import uuid
            temp_img_path = os.path.join(temp_img_dir, f"{str(uuid.uuid4())}.png")
            shutil.copy2(file_path, temp_img_path)
            
            # Apply OCR and clean up
            text = ocr.applyOCR(temp_img_dir, temp_ocr_dir)
            shutil.rmtree(temp_img_dir, ignore_errors=True)
            shutil.rmtree(temp_ocr_dir, ignore_errors=True)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the file")
        
        # Step 2: Preprocess the extracted text
        cleaned_text = text_preprocessing.text_processing(text)
        
        # Step 3: Extract keywords and other information
        parsed_output = extract_keywords_from_resume(cleaned_text)
        
        # Return tuple matching the expected format in main.py
        return cleaned_text, parsed_output
        
    except Exception as e:
        logger.error(f"Error processing resume {file_path}: {str(e)}")
        raise Exception(f"Pipeline error: {str(e)}")

# For backward compatibility
pipeline = process_resume
