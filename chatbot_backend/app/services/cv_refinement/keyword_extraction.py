"""
Resume Processing Pipeline - Extract and analyze resume content
"""

import os
import re
import logging
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Tuple, List
from pathlib import Path

import pytesseract
from PIL import Image
import PyPDF2
from pdf2image import convert_from_path
from docx import Document

from app.services.ingestion import text_preprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
IMAGE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = {'.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}

# Skill and title patterns (compiled for performance)
SKILL_PATTERN = re.compile(
    r'\b(?:python|java|javascript|typescript|react|vue|angular|node\.?js|django|flask|'
    r'spring|sql|mongodb|postgresql|aws|azure|docker|kubernetes|git|machine[ -]?learning|'
    r'data[ -]?science|ai|devops|api|microservices)\b',
    re.IGNORECASE
)

TITLE_PATTERN = re.compile(
    r'\b(?:software\s+(?:engineer|developer|architect)|front[ -]?end|back[ -]?end|'
    r'full[ -]?stack|devops|data\s+(?:scientist|engineer|analyst)|'
    r'machine[ -]?learning\s+engineer|cloud\s+engineer|product\s+manager)\b',
    re.IGNORECASE
)

# Industry keywords (simplified)
INDUSTRIES = {
    'technology': ['tech', 'software', 'it', 'saas', 'cloud', 'ai'],
    'finance': ['finance', 'banking', 'fintech', 'trading'],
    'healthcare': ['healthcare', 'medical', 'hospital', 'clinical'],
    'ecommerce': ['ecommerce', 'retail', 'marketplace']
}

# ==================== TEXT EXTRACTION ====================
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2, fallback to OCR"""
    try:
        # Try direct extraction first
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            if text.strip():
                return text
    except Exception as e:
        logger.debug(f"Direct PDF extraction failed: {e}")
    
    # Fallback to OCR
    temp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    try:
        images = convert_from_path(pdf_path)
        text_parts = []
        
        for i, img in enumerate(images):
            img_path = os.path.join(temp_dir, f"page_{i}.png")
            img.save(img_path, 'PNG')
            text_parts.append(_ocr_image(img_path))
        
        return "\n\n".join(filter(None, text_parts))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = Document(docx_path)
        return '\n'.join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to read DOCX: {e}")


def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using OCR"""
    if os.path.getsize(image_path) > IMAGE_MAX_SIZE:
        raise ValueError(f"Image too large. Max: 10MB")
    
    if os.path.getsize(image_path) == 0:
        raise ValueError("Image file is empty")
    
    temp_dir = tempfile.mkdtemp(prefix="img_ocr_")
    try:
        temp_img = os.path.join(temp_dir, "temp.png")
        
        # Normalize image
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(temp_img, 'PNG')
        
        return _ocr_image(temp_img)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _ocr_image(image_path: str) -> str:
    """Apply OCR to single image with multiple configs"""
    configs = [
        '--oem 3 --psm 6 -l eng+vie',  # Single block
        '--oem 3 --psm 3 -l eng+vie',  # Auto segmentation
        '--oem 3 --psm 6 -l eng',      # English only
    ]
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(
                Image.open(image_path),
                config=config,
                timeout=30
            )
            if text.strip():
                return text.strip()
        except Exception as e:
            logger.debug(f"OCR with config '{config}' failed: {e}")
    
    raise ValueError("No text could be extracted from image")


# ==================== KEYWORD EXTRACTION ====================
def extract_keywords_from_resume(text: str) -> Dict:
    """Extract skills, titles, industries, and experience from resume"""
    text_lower = text.lower()
    
    # Extract skills and titles
    skills = list(set(SKILL_PATTERN.findall(text_lower)))
    titles = list(set(TITLE_PATTERN.findall(text_lower)))
    
    # Detect industries
    industries = [
        industry for industry, keywords in INDUSTRIES.items()
        if any(kw in text_lower for kw in keywords)
    ]
    if not industries and (skills or titles):
        industries = ['technology']
    
    # Calculate experience
    experience_info = _calculate_experience(text)
    
    # Determine level
    level = _determine_experience_level(text, experience_info['years'])
    
    return {
        'technical_skills': skills,
        'job_titles': titles,
        'industries': industries,
        'experience_years': experience_info['years'],
        'experience_str': experience_info['str'],
        'level': level
    }


def _calculate_experience(text: str) -> Dict:
    """Calculate total work experience from date ranges"""
    # Pattern: MM/YYYY - MM/YYYY or YYYY - YYYY or present
    date_pattern = r'(\d{1,2}/\d{4}|\d{4})\s*[-–]\s*(\d{1,2}/\d{4}|\d{4}|nay|present|hiện\s*tại)'
    
    periods = []
    for match in re.finditer(date_pattern, text, re.IGNORECASE):
        start_str, end_str = match.groups()
        try:
            # Parse start date
            start = datetime.strptime(start_str, '%m/%Y') if '/' in start_str else datetime(int(start_str), 1, 1)
            
            # Parse end date
            if end_str.lower() in ['nay', 'present', 'hiện tại']:
                end = datetime.now()
            elif '/' in end_str:
                end = datetime.strptime(end_str, '%m/%Y')
            else:
                end = datetime(int(end_str), 1, 1)
            
            if start < end:
                periods.append((start, end))
        except (ValueError, TypeError):
            continue
    
    if not periods:
        return {'years': 0, 'months': 0, 'str': 'Chưa có kinh nghiệm'}
    
    # Merge overlapping periods
    periods.sort()
    merged = [list(periods[0])]
    
    for start, end in periods[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    
    # Calculate total months
    total_months = sum((end.year - start.year) * 12 + (end.month - start.month) for start, end in merged)
    years = total_months // 12
    months = total_months % 12
    
    # Format string
    parts = []
    if years > 0:
        parts.append(f"{years} năm")
    if months > 0 or years == 0:
        parts.append(f"{months} tháng")
    
    return {
        'years': round(total_months / 12, 1),
        'months': total_months,
        'str': ' '.join(parts)
    }


def _determine_experience_level(text: str, years: float) -> str:
    """Determine experience level based on years and keywords"""
    text_lower = text.lower()
    
    # Level keywords
    level_keywords = {
        'intern': ['intern', 'thực tập', 'fresher'],
        'junior': ['junior', 'develop', 'implement'],
        'mid': ['mid', 'lead', 'manage', 'design'],
        'senior': ['senior', 'architect', 'strategy', 'mentor']
    }
    
    # Count keyword matches
    scores = {level: sum(kw in text_lower for kw in keywords) 
              for level, keywords in level_keywords.items()}
    
    # Determine level
    if years >= 5:
        return 'senior'
    elif years >= 3:
        return 'mid' if scores['mid'] > 0 or scores['senior'] > 0 else 'junior'
    elif years >= 1:
        return 'junior' if scores['junior'] > 0 else 'entry'
    else:
        return 'intern' if scores['intern'] > 0 else 'entry'


# ==================== MAIN PIPELINE ====================
def process_resume(file_path: str) -> Tuple[str, Dict]:
    """
    Process resume file and extract information.
    
    Args:
        file_path: Path to resume file
        
    Returns:
        (processed_text, parsed_output)
        
    Raises:
        ValueError: If file cannot be processed
    """
    start = datetime.now()
    logger.info(f"Processing: {file_path}")
    
    # Validate file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("File is empty")
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Max: 50MB")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format. Supported: {SUPPORTED_FORMATS}")
    
    # Extract text
    try:
        if file_ext == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            text = extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:  # Image
            text = extract_text_from_image(file_path)
        
        if not text or not text.strip():
            raise ValueError("No text could be extracted")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise ValueError(f"Failed to extract text: {e}")
    
    # Preprocess text
    try:
        cleaned_text = text_preprocessing.text_processing(text)
        if not cleaned_text.strip():
            raise ValueError("No meaningful text after preprocessing")
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")
    
    # Extract keywords
    try:
        parsed = extract_keywords_from_resume(cleaned_text)
        
        # Add metadata
        parsed.update({
            'file_type': file_ext.lstrip('.'),
            'file_size_kb': round(file_size / 1024, 2),
            'processing_time': round((datetime.now() - start).total_seconds(), 2),
            'text_length': len(cleaned_text)
        })
        
        # Detect language
        en_words = sum(1 for w in ['the', 'and', 'experience', 'skills'] if w in cleaned_text.lower())
        vi_words = sum(1 for w in ['và', 'của', 'kinh nghiệm', 'kỹ năng'] if w in cleaned_text.lower())
        parsed['language'] = 'vi' if vi_words > en_words else 'en' if en_words > 0 else 'unknown'
        
        logger.info(f"Processed in {parsed['processing_time']}s")
        return cleaned_text, parsed
        
    except Exception as e:
        raise ValueError(f"Analysis failed: {e}")


# Backward compatibility
pipeline = process_resume


# ==================== SELF-TEST ====================
if __name__ == '__main__':
    # Test with a sample
    test_resume = """
    John Doe - Software Engineer
    Email: john@example.com
    
    EXPERIENCE:
    Senior Software Developer at TechCorp (01/2020 - Present)
    - Developed microservices using Python and Docker
    - Led team of 3 developers
    
    Full Stack Developer at StartupXYZ (06/2018 - 12/2019)
    - Built React frontend and Node.js backend
    - Implemented CI/CD with Jenkins
    
    SKILLS:
    Python, JavaScript, React, Docker, AWS, Kubernetes
    
    EDUCATION:
    BS Computer Science (2018)
    """
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_resume)
        temp_path = f.name
    
    try:
        text, info = process_resume(temp_path)
        print(f"\n{'='*60}")
        print(f"Resume Analysis Results")
        print(f"{'='*60}\n")
        print(f"Skills: {info['technical_skills']}")
        print(f"Titles: {info['job_titles']}")
        print(f"Industries: {info['industries']}")
        print(f"Experience: {info['experience_str']} ({info['experience_years']} years)")
        print(f"Level: {info['level']}")
        print(f"Language: {info['language']}")
        print(f"Processing time: {info['processing_time']}s")
    finally:
        os.unlink(temp_path)