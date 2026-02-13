"""
Resume Improvement Suggestion Module - AI-powered resume analysis
"""

import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from app.utils.llm_utils import get_hf_model, init_gemini, get_ollama_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()

# Optimized prompt template
RESUME_REVIEW_PROMPT = """You are an expert resume reviewer. Analyze this resume and provide:

1. **Rating** (1-10): Rate content quality, clarity, and impact (ignore formatting)
   - 10: Exceptional, industry-leading
   - 7-9: Strong, competitive
   - 5-6: Average, needs work
   - <5: Significant improvements needed

2. **Key Improvements** (3-5 actionable points):
   - Focus on: missing keywords, quantifiable achievements, clarity
   - Be specific: "Add metrics to project X" not "improve descriptions"

3. **Project Recommendations** (1-3 projects):
   - Align with existing skills
   - Teach complementary technologies
   - Must be practical and portfolio-worthy

Resume:
{resume_text}

Format your response clearly with sections: RATING, IMPROVEMENTS, PROJECTS."""


def suggest_resume_improvements(
    resume_text: str,
    model_name: str = "qwen2.5:3b",
    username: str = "user"
) -> Dict[str, str]:
    """
    Analyze resume and suggest improvements using AI.
    
    Args:
        resume_text: Extracted resume text
        model_name: "ollama" (default), "gemini", or HF model path
        username: Username for tracking
    
    Returns:
        {username, model, improvements}
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return {
            "username": username,
            "model": model_name,
            "improvements": "Resume text is too short or empty. Please upload a complete resume."
        }
    
    logger.info(f"Analyzing resume for {username} with {model_name}")
    
    try:
        # Generate prompt
        prompt = RESUME_REVIEW_PROMPT.format(resume_text=resume_text[:2000])  # Limit length
        
        # Route to appropriate model
        if "qwen" in model_name.lower() or "llama" in model_name.lower() or "ollama" in model_name.lower():
            response = _get_ollama_response(prompt, model_name)
        elif "gemini" in model_name.lower():
            response = _get_gemini_response(prompt, model_name)
        else:
            response = _get_hf_response(prompt, model_name)
        
        return {
            "username": username,
            "model": model_name,
            "improvements": response.strip()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        return {
            "username": username,
            "model": model_name,
            "improvements": (
                f"⚠️ Analysis temporarily unavailable.\n\n"
                f"General tips while we resolve this:\n"
                f"• Add quantifiable achievements (e.g., 'Improved performance by 40%')\n"
                f"• Use action verbs (e.g., 'Developed', 'Implemented', 'Led')\n"
                f"• Include relevant keywords for your target role\n"
                f"• Keep descriptions concise and impactful\n\n"
                f"Error: {str(e)}"
            )
        }


def _get_ollama_response(prompt: str, model_name: str) -> str:
    """Get response from Ollama"""
    return get_ollama_response(
        prompt=prompt,
        system_prompt="You are a professional resume reviewer. Be critical but constructive.",
        model=model_name
    )


def _get_gemini_response(prompt: str, model_name: str) -> str:
    """Get response from Gemini"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    
    llm = init_gemini(model_name, api_key)
    return llm.generate_content(prompt).text


def _get_hf_response(prompt: str, model_name: str) -> str:
    """Get response from HuggingFace model"""
    tokenizer, model = get_hf_model(model_name)
    
    messages = [
        {"role": "system", "content": "You are a professional resume reviewer."},
        {"role": "user", "content": prompt}
    ]
    
    # Generate response
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat_prompt], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# Quick self-test
if __name__ == '__main__':
    sample_resume = """
    John Doe | Software Engineer
    Email: john@example.com | Phone: 123-456-7890
    
    EXPERIENCE:
    Software Developer at TechCorp (2022-Present)
    - Worked on backend systems
    - Helped team with code reviews
    - Fixed bugs
    
    SKILLS:
    Python, JavaScript, Git, Teamwork
    
    EDUCATION:
    BS Computer Science, State University (2022)
    """
    
    result = suggest_resume_improvements(sample_resume, model_name="qwen2.5:3b")
    print(f"\n{'='*60}")
    print(f"Resume Analysis for {result['username']}")
    print(f"Model: {result['model']}")
    print(f"{'='*60}\n")
    print(result['improvements'])