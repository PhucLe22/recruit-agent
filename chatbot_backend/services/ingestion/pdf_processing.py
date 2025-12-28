from pdf2image import convert_from_path
import shutil
import os
import logging

logger = logging.getLogger(__name__)

poppler_path = r'C:\Users\Daryn Bang\PycharmProjects\poppler-24.08.0\Library\bin'

def validate_pdf(pdf_path: str) -> bool:
    """Validate if the file is a valid PDF by checking its header and structure"""
    try:
        # Check file exists and is not empty
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
            return False
            
        # Check PDF header (first 4 bytes should be %PDF)
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
        
        # Try to read a small portion to verify it's not completely corrupted
        try:
            # Attempt to get page count without full conversion
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                # If we can get the number of pages, it's likely a valid PDF
                _ = len(pdf_reader.pages)
                return True
        except Exception as e:
            logger.warning(f"PyPDF2 validation failed for {pdf_path}: {str(e)}")
            # Fall back to basic header check
            return True
            
    except Exception as e:
        logger.error(f"PDF validation failed for {pdf_path}: {str(e)}")
        return False

def convert_pdf_to_img(pdf_path, dpi=100, img_dir="pages_img", poppler_path=None):
    """Convert PDF to images with validation and error handling"""
    # Validate PDF first
    if not validate_pdf(pdf_path):
        raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")
    
    # Ensure img_dir exists
    os.makedirs(img_dir, exist_ok=True)

    # Empty img_dir before saving new images
    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # remove file/symlink
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # remove folder

    try:
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        if not pages:
            raise ValueError("No pages found in PDF")
            
        for idx, page in enumerate(pages, start=1):
            img_path = os.path.join(img_dir, f"page_{idx:02d}.png")
            page.save(img_path, "PNG")
            logger.info(f"Saved {img_path}")
            
        return len(pages)  # Return number of pages processed
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise ValueError(f"Failed to convert PDF to images: {str(e)}")

