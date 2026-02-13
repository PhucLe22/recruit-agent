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
        page_count = pdf_processing.convert_pdf_to_img(pdf_path, img_dir=img_dir)
        logger.info(f"Successfully converted {page_count} pages from PDF")
        
        extracted_text = ocr.applyOCR(img_dir, ocr_dir)
        
        if not extracted_text.strip():
            raise ValueError("OCR extracted empty text from PDF")
            
        logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF")
        return extracted_text
        
    except ValueError as e:
        # Re-raise ValueError with more context
        logger.error(f"PDF validation/processing error for {pdf_path}: {str(e)}")
        raise ValueError(f"Invalid PDF file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error extracting text from PDF {pdf_path}: {str(e)}")
        raise Exception(f"PDF processing failed: {str(e)}")

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
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        if os.path.getsize(file_path) == 0:
            raise ValueError("File is empty")
        
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
            
            try:
                # Copy the image to the temp directory
                import shutil
                import uuid
                temp_img_path = os.path.join(temp_img_dir, f"{str(uuid.uuid4())}.png")
                shutil.copy2(file_path, temp_img_path)
                
                # Apply OCR
                text = ocr.applyOCR(temp_img_dir, temp_ocr_dir)
                
            finally:
                # Clean up temporary directories
                shutil.rmtree(temp_img_dir, ignore_errors=True)
                shutil.rmtree(temp_ocr_dir, ignore_errors=True)
        
        if not text or not text.strip():
            raise ValueError("No text could be extracted from the file. The file may be corrupted, password-protected, or contain only images.")
        
        # Step 2: Preprocess the extracted text
        cleaned_text = text_preprocessing.text_processing(text)
        
        if not cleaned_text or not cleaned_text.strip():
            raise ValueError("Text preprocessing resulted in empty content")
        
        # Step 3: Extract keywords and other information
        parsed_output = extract_keywords_from_resume(cleaned_text)
        
        # Return tuple matching the expected format in main.py
        return cleaned_text, parsed_output
        
    except (FileNotFoundError, ValueError) as e:
        # Re-raise known errors with context
        logger.error(f"Resume processing error for {file_path}: {str(e)}")
        raise Exception(f"Resume processing failed: {str(e)}")
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error processing resume {file_path}: {str(e)}")
        raise Exception(f"Pipeline error: {str(e)}")

# For backward compatibility
pipeline = process_resume
