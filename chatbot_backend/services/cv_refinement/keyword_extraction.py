import os
import re
import logging
import shutil
import uuid
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import mimetypes

# Third-party imports
import pytesseract
from PIL import Image, UnidentifiedImageError
import PyPDF2
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
import docx
from docx import Document

# Local imports
from services.ingestion import text_preprocessing

logger = logging.getLogger(__name__)

def extract_keywords_from_resume(text: str) -> Dict[str, Union[List[str], str]]:
    """Extract keywords from resume text using regex patterns and context analysis."""
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Common industry keywords and their variations
    industry_keywords = {
        'technology': ['technology', 'tech', 'software', 'it', 'information technology', 'computer', 'internet', 'saas', 'cloud', 'ai', 'artificial intelligence'],
        'finance': ['finance', 'financial', 'banking', 'investment', 'accounting', 'fintech', 'wealth management', 'trading', 'stock market'],
        'healthcare': ['healthcare', 'medical', 'pharmaceutical', 'hospital', 'health', 'biotech', 'clinical', 'nursing', 'doctor', 'physician'],
        'education': ['education', 'academic', 'university', 'school', 'learning', 'teaching', 'edtech', 'e-learning', 'training'],
        'ecommerce': ['ecommerce', 'e-commerce', 'retail', 'online shopping', 'marketplace', 'dropshipping'],
        'manufacturing': ['manufacturing', 'production', 'factory', 'industrial', 'assembly', 'supply chain', 'logistics'],
        'telecommunications': ['telecom', 'telecommunication', '5g', 'networking', 'isp', 'wireless', 'mobile'],
        'energy': ['energy', 'renewable', 'solar', 'wind', 'oil', 'gas', 'utilities', 'power', 'electricity'],
        'marketing': ['marketing', 'advertising', 'digital marketing', 'seo', 'sem', 'social media', 'content marketing', 'branding'],
        'consulting': ['consulting', 'consultant', 'advisory', 'professional services', 'business consulting']
    }
    
    # Extract technical skills
    technical_skills = list(set(re.findall(
        r'\b(?:python|java|c\+\+|javascript|typescript|react|angular|vue|node\.?js|django|flask|fastapi|spring|sql|nosql|mongodb|postgresql|mysql|aws|azure|gcp|docker|kubernetes|terraform|ansible|jenkins|git|linux|bash|machine[ -]?learning|data[ -]?science|ai|artificial[ -]?intelligence|nlp|computer[ -]?vision|big[ -]?data|spark|hadoop|tableau|power[ -]?bi|excel|agile|scrum|devops|ci/cd|rest|graphql|api|microservices|serverless)\b',
        text_lower
    )))
    
    # Extract job titles
    job_titles = list(set(re.findall(
        r'\b(?:software\s+(?:engineer|developer|architect)|front[ -]?end|back[ -]?end|full[ -]?stack|devops|data\s+(?:scientist|engineer|analyst)|machine[ -]?learning\s+engineer|ai\s+engineer|data\s+scientist|cloud\s+engineer|solutions?\s+architect|system\s+admin|network\s+engineer|cyber(?:security)?\s+engineer|ux/ui\s+designer|product\s+manager|project\s+manager|technical\s+lead|cto|ceo|cio|it\s+manager|scrum\s+master)\b',
        text_lower
    )))
    
    # Detect industries by analyzing the entire text
    detected_industries = []
    for industry, keywords in industry_keywords.items():
        # Check if any of the industry keywords appear in the text
        if any(f' {kw} ' in f' {text_lower} ' for kw in keywords):
            detected_industries.append(industry)
    
    # If no industries detected, try to infer from job titles and skills
    if not detected_industries:
        if any('data' in skill or 'ai' in skill or 'machine' in skill for skill in technical_skills):
            detected_industries.append('technology')
        if any('health' in skill or 'medical' in skill for skill in technical_skills):
            detected_industries.append('healthcare')
    
    # If still no industries, add a default
    if not detected_industries and (technical_skills or job_titles):
        detected_industries.append('technology')  # Default to technology if other tech skills are present
    
    # Extract work experience from companies (exclude school projects)
    total_months = 0
    work_experience = []
    work_periods = []  # To store all work periods for overlap checking
    
    # Look for work experience sections (case insensitive)
    work_exp_pattern = r'(work\s*experience|kinh\s*nghi\u1ec7m\s*làm\s*vi\u1ec7c|kinh\s*nghi\u1ec7m|experience)(.*?)(?=education|h\u1ecdc v\u1ea5n|skills|k\u1ef9 n\u0103ng|$)'
    logger.info(f"Searching for work experience section with pattern: {work_exp_pattern}")
    work_exp_section = re.search(work_exp_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if not work_exp_section:
        logger.warning("No work experience section found in the text")
    
    if work_exp_section:
        work_exp_text = work_exp_section.group(2)
        logger.info(f"Found work experience section. Text length: {len(work_exp_text)} characters")
        # Look for date patterns in work experience section (MM/YYYY - MM/YYYY or YYYY - YYYY)
        date_pattern = r'(?P<start>\d{1,2}/\d{4}|\d{4})\s*[-–]\s*(?P<end>\d{1,2}/\d{4}|\d{4}|nay|present|hiện\s*tại)'
        logger.info(f"Searching for date patterns: {date_pattern}")
        date_matches = list(re.finditer(date_pattern, work_exp_text, re.IGNORECASE))
        logger.info(f"Found {len(date_matches)} date patterns in work experience section")
        
        for match in date_matches:
            start_date = match.group('start')
            end_date = match.group('end')
            
            # Convert dates to datetime objects
            try:
                if '/' in start_date:
                    start = datetime.strptime(start_date, '%m/%Y')
                else:
                    start = datetime(start_date, 1, 1)  # Default to January 1st if only year is given
                
                if end_date.lower() in ['nay', 'present', 'hiện tại']:
                    end = datetime.now()
                elif '/' in end_date:
                    end = datetime.strptime(end_date, '%m/%Y')
                else:
                    end = datetime(int(end_date), 1, 1)  # Default to January 1st if only year is given
                
                # Calculate months of experience for this position
                delta_months = (end.year - start.year) * 12 + (end.month - start.month)
                if delta_months > 0:
                    # Store the period for overlap checking
                    work_periods.append((start, end))
                    
                    # Store each work period for reference
                    period_info = {
                        'start': start.strftime('%m/%Y'),
                        'end': 'Present' if end_date.lower() in ['nay', 'present', 'hiện tại'] else end.strftime('%m/%Y'),
                        'months': delta_months,
                        'years': round(delta_months / 12, 1)
                    }
                    work_experience.append(period_info)
                    logger.debug(f"Found work period: {period_info}")
            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing date range {start_date} - {end_date}: {e}")
                continue
    
    # Calculate total experience considering overlapping periods
    logger.info(f"Total work periods found: {len(work_periods)}")
    if work_periods:
        # Sort periods by start date
        work_periods.sort(key=lambda x: x[0])
        logger.info(f"Work periods after sorting: {work_periods}")
        
        # Merge overlapping periods
        merged_periods = []
        for period in work_periods:
            if not merged_periods:
                merged_periods.append(list(period))
            else:
                last_start, last_end = merged_periods[-1]
                current_start, current_end = period
                
                # If current period overlaps with or is adjacent to the last one
                if current_start <= last_end:
                    # Merge with the last period
                    merged_periods[-1][1] = max(last_end, current_end)
                else:
                    merged_periods.append([current_start, current_end])
        
        # Calculate total months from merged periods
        total_months = 0
        logger.info("Calculating total months from merged periods:")
        for i, (start, end) in enumerate(merged_periods, 1):
            delta_months = (end.year - start.year) * 12 + (end.month - start.month)
            total_months += delta_months
            logger.info(f"  Period {i}: {start.strftime('%m/%Y')} to {end.strftime('%m/%Y')} = {delta_months} months")
            
    # Calculate years and months
    total_years = total_months // 12
    remaining_months = int(round(total_months % 12, 0))
    
    total_experience = []
    if total_years > 0:
        total_experience.append(f"{int(total_years)} năm")
    if remaining_months > 0 or total_years == 0:  # Show months if >0 or if no years
        total_experience.append(f"{remaining_months} tháng")
    
    logger.info(f"Final calculation: {total_months} total months = {total_years} years and {remaining_months} months")
    
    total_experience_str = ' '.join(total_experience) if total_experience else "Chưa có kinh nghiệm"
    experience_years = round(total_months / 12, 1)  # For backward compatibility
    
    logger.info(f"Total work experience calculated: {total_months} months ({experience_years} years) - {total_experience_str}")
    
    def analyze_responsibilities(text: str) -> dict:
        """Analyze job responsibilities to determine experience level."""
        level_indicators = {
            'intern': ['internship', 'thực tập', 'hỗ trợ', 'assist', 'support', 'học hỏi'],
            'junior': ['phát triển', 'develop', 'thực hiện', 'implement', 'tham gia', 'participate'],
            'mid': ['quản lý', 'manage', 'dẫn dắt', 'lead', 'thiết kế', 'design', 'tối ưu', 'optimize'],
            'senior': ['kiến trúc', 'architect', 'chiến lược', 'strategy', 'định hướng', 'mentor', 'coach', 'hướng dẫn']
        }
        
        score = {'intern': 0, 'junior': 0, 'mid': 0, 'senior': 0}
        text_lower = text.lower()
        
        for level, keywords in level_indicators.items():
            score[level] = sum(keyword in text_lower for keyword in keywords)
            
        return score
    
    # Default level
    level = 'intern/fresher'
    
    # Check for experience-related keywords
    experience_keywords = ['experience', 'kinh nghiệm', 'work history', 'quá trình làm việc']
    has_experience_section = any(keyword in text_lower for keyword in experience_keywords)
    
    if has_experience_section:
        # Analyze job descriptions for level indicators
        responsibility_scores = analyze_responsibilities(text)
        
        # Get the highest scoring level
        max_level = max(responsibility_scores, key=responsibility_scores.get)
        
        # If we have actual experience years, use that as base
        if experience_years > 0:
            if experience_years >= 5:
                level = 'senior'
            elif experience_years >= 3:
                level = 'mid-level' if max_level in ['mid', 'senior'] else 'junior'
            elif experience_years > 1:
                level = 'junior' if max_level != 'intern' else 'intern/fresher'
            else:
                level = 'entry-level'
        # If no experience years but has responsibilities
        elif any(score > 0 for score in responsibility_scores.values()):
            level_map = {
                'intern': 'intern/fresher',
                'junior': 'junior',
                'mid': 'mid-level',
                'senior': 'senior'
            }
            level = level_map.get(max_level, 'intern/fresher')
    
    # Check for intern/fresher keywords if no experience found
    if level == 'intern/fresher' and any(term in text_lower for term in ['intern', 'internship', 'thực tập', 'fresher', 'mới ra trường', 'sinh viên', 'student']):
        level = 'intern/fresher'
    
    # Log the level determination
    logger.info(f"Determined experience level: {level} based on {experience_years} years of experience and responsibility analysis")
    
    # Prepare the result dictionary with basic fields
    result = {
        'technical_skills': technical_skills,
        'job_titles': job_titles,
        'industries': list(set(detected_industries))  # Remove duplicates
    }
    
    # Only add experience_years if we found a valid value
    if experience_years > 0:
        result['experience_years'] = experience_years
    
    # Set level in result
    result['level'] = level if level is not None else 'intern/fresher'
    
    return result

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using multiple methods for better accuracy."""
    text = ""
    
    # First try: Extract text directly from PDF
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            if text.strip():  # If we got text, return it
                return text
    except Exception as e:
        logger.warning(f"Direct PDF text extraction failed, falling back to OCR: {e}")

    # Second try: OCR the PDF pages
    img_dir = "temp_pdf_images"
    ocr_dir = "temp_ocr_output"
    try:
        # Create temporary directories
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ocr_dir, exist_ok=True)
        
        # Convert PDF to images
        try:
            images = convert_from_path(pdf_path)
            if not images:
                raise ValueError("No pages found in PDF")
                
            for i, image in enumerate(images):
                img_path = os.path.join(img_dir, f"page_{i+1}.png")
                try:
                    image.save(img_path, 'PNG')
                except Exception as img_error:
                    logger.warning(f"Failed to save image {i+1}: {img_error}")
                    continue
            
            # Apply OCR to each image
            text = apply_ocr_to_directory(img_dir, ocr_dir)
            if not text.strip():
                raise ValueError("No text could be extracted using OCR")
                
            return text
            
        except Exception as ocr_error:
            logger.error(f"PDF OCR extraction failed: {ocr_error}")
            raise ValueError(f"Could not extract text from PDF: {ocr_error}")
            
    except Exception as e:
        logger.error(f"Unexpected error during PDF processing: {e}")
        raise ValueError(f"Failed to process PDF: {e}")
        
    finally:
        # Clean up temporary directories
        shutil.rmtree(img_dir, ignore_errors=True)
        shutil.rmtree(ocr_dir, ignore_errors=True)

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        Extracted text from the document
        
    Raises:
        ValueError: If the file cannot be processed
    """
    try:
        doc = Document(docx_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX file: {str(e)}")

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using OCR with robust error handling.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text from the image
        
    Raises:
        ValueError: If the image cannot be processed or no text is found
    """
    # Create a single temp directory for all operations
    temp_dir = tempfile.mkdtemp(prefix="cv_ocr_")
    temp_img_path = os.path.join(temp_dir, "temp_image.png")
    
    try:
        # Validate input file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            raise ValueError("The image file is empty")
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Image file is too large. Maximum size is 10MB.")
        
        # Process the image
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (Tesseract works best with RGB)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save the processed image to a known location
                img.save(temp_img_path, 'PNG')
                
        except (IOError, OSError, UnidentifiedImageError) as e:
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")
        
        # Verify the saved image exists and is readable
        if not os.path.exists(temp_img_path):
            raise ValueError("Failed to process the image file. Please try again.")
            
        # Try different OCR configurations
        psm_modes = [
            ('6', 'eng+vie'),  # Assume a single uniform block of text
            ('3', 'eng+vie'),  # Fully automatic page segmentation
            ('4', 'eng+vie'),  # Assume a single column of text
            ('6', 'eng'),      # Try English only
            ('6', 'vie')       # Try Vietnamese only
        ]
        
        last_error = None
        
        for psm, lang in psm_modes:
            try:
                custom_config = f'--oem 3 --psm {psm} -l {lang}'
                text = pytesseract.image_to_string(
                    Image.open(temp_img_path),
                    config=custom_config,
                    timeout=30
                )
                
                if text and text.strip():
                    return text.strip()
                    
            except Exception as e:
                last_error = e
                logger.debug(f"OCR attempt with PSM {psm} and lang {lang} failed: {str(e)}")
                continue
        
        # If we get here, all OCR attempts failed
        if last_error:
            if "not found" in str(last_error).lower() or "no such file" in str(last_error).lower():
                raise ValueError("Failed to process the image. The file might be corrupted or in an unsupported format.")
            raise ValueError(f"OCR processing failed: {str(last_error)}")
        else:
            raise ValueError("No text could be extracted from the image. The image might not contain any readable text.")
            
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}", exc_info=True)
        if "cannot identify image file" in str(e).lower():
            raise ValueError("The file format is not a supported image type. Please upload a PNG, JPG, or JPEG file.")
        if "Tesseract not found" in str(e):
            raise ValueError("OCR engine is not properly installed. Please contact support.")
        raise ValueError(f"Error processing image: {str(e)}")
        
    finally:
        # Clean up the temp directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

def apply_ocr_to_directory(img_dir: str, ocr_dir: str) -> str:
    """Apply OCR to all images in a directory and return combined text.
    
    Args:
        img_dir: Directory containing images to process
        ocr_dir: Directory to store OCR output (unused, kept for backward compatibility)
        
    Returns:
        Combined text from all processed images
    """
    text_parts = []
    
    # Get list of image files, sorted by name
    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or '0')
    )
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        try:
            # Skip non-files and empty files
            if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                continue
                
            # Try different PSM modes for better text extraction
            psm_modes = [
                '6',  # Assume a single uniform block of text
                '3',  # Fully automatic page segmentation, but no OSD
                '4',  # Assume a single column of text of variable sizes
                '1'   # Automatic page segmentation with OSD
            ]
            
            page_text = ""
            
            for psm in psm_modes:
                try:
                    custom_config = f'--oem 3 --psm {psm} -l eng+vie'
                    page_text = pytesseract.image_to_string(
                        Image.open(img_path),
                        config=custom_config,
                        timeout=30  # 30 seconds timeout per page
                    )
                    if page_text.strip():
                        break  # If we got text, no need to try other modes
                except Exception as e:
                    logger.warning(f"OCR with PSM {psm} failed for {img_file}: {e}")
            
            if page_text.strip():
                text_parts.append(page_text.strip())
                
        except Exception as e:
            logger.error(f"Error processing image {img_file}: {e}")
            continue
            
    return "\n\n".join(text_parts)

def process_resume(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Main pipeline function to process a resume file.
    
    Args:
        file_path: Path to the resume file (PDF, DOCX, TXT, or image)
        
    Returns:
        Tuple of (processed_text, parsed_output) where:
        - processed_text: The cleaned text from the resume
        - parsed_output: Dictionary containing extracted information
        
    Raises:
        ValueError: If the file format is not supported or cannot be processed
    """
    start_time = datetime.now()
    logger.info(f"Starting resume processing for: {file_path}")
    
    try:
        # Validate file path
        if not file_path or not isinstance(file_path, str):
            raise ValueError("Invalid file path provided")
            
        file_path = os.path.abspath(file_path)
        
        # Check if file exists and is not empty
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("The file is empty")
            
        # Set maximum file size (50MB)
        if file_size > 50 * 1024 * 1024:
            raise ValueError("File is too large. Maximum size is 50MB.")
        
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        mime_type, _ = mimetypes.guess_type(file_path)
        
        logger.info(f"Processing {file_ext} file with mime type: {mime_type}")
        
        # Step 1: Extract text based on file type
        text = ""
        try:
            if file_ext == '.pdf':
                text = extract_text_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                text = extract_text_from_docx(file_path)
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            elif file_ext == '.svg':
                raise ValueError("SVG files are not supported. Please upload a PDF, DOCX, DOC, or image file.")
            elif mime_type and mime_type.startswith('image/'):
                text = extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: PDF, DOCX, DOC, TXT, PNG, JPG, JPEG")
                
            if not text or not text.strip():
                raise ValueError("The file appears to be empty or doesn't contain any extractable text")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            if 'cannot identify image file' in str(e).lower():
                raise ValueError("The file could not be processed as an image. Please ensure it's a valid image file (PNG, JPG, JPEG).")
            if 'cannot open' in str(e).lower() or 'not a supported' in str(e).lower():
                raise ValueError(f"The file format is not supported or the file is corrupted. Error: {str(e)}")
            raise
        
        # Step 2: Preprocess the extracted text
        try:
            cleaned_text = text_preprocessing.text_processing(text)
            if not cleaned_text.strip():
                raise ValueError("No meaningful text could be extracted after preprocessing")
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            raise ValueError(f"Error processing the document text: {str(e)}")
        
        # Step 3: Extract keywords and other information
        try:
            parsed_output = extract_keywords_from_resume(cleaned_text)
            
            # Add metadata
            parsed_output.update({
                'file_type': file_ext.lstrip('.'),
                'file_size_kb': round(file_size / 1024, 2),
                'processing_time_seconds': round((datetime.now() - start_time).total_seconds(), 2),
                'text_length': len(cleaned_text),
                'text_preview': cleaned_text[:500] + ('...' if len(cleaned_text) > 500 else '')
            })
            
            # Simple language detection
            en_words = ['the', 'and', 'for', 'with', 'experience', 'education', 'skills', 'work']
            vi_words = ['và', 'của', 'cho', 'với', 'kinh nghiệm', 'giáo dục', 'kỹ năng', 'công việc']
            
            en_count = sum(1 for word in en_words if word in cleaned_text.lower())
            vi_count = sum(1 for word in vi_words if word in cleaned_text.lower())
            
            if vi_count > en_count:
                parsed_output['detected_language'] = 'vi'
            elif en_count > 0:
                parsed_output['detected_language'] = 'en'
            else:
                parsed_output['detected_language'] = 'unknown'
            
            logger.info(f"Successfully processed resume in {parsed_output['processing_time_seconds']} seconds")
            return cleaned_text, parsed_output
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            raise ValueError(f"Error analyzing the resume content: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing resume {file_path}: {str(e)}", exc_info=True)
        if not str(e).startswith(('File not found', 'The file is empty', 'Unsupported file format', 'The file appears to be empty')):
            logger.error(f"Unexpected error: {str(e)}")
        raise ValueError(f"Failed to process resume: {str(e)}")
    
    finally:
        # Clean up any temporary files if needed
        pass

# For backward compatibility
pipeline = process_resume