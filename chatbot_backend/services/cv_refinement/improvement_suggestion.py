import textwrap
import os
from typing import Optional
from dotenv import load_dotenv
from utils.llm_utils import get_hf_model, init_gemini, get_ollama_response

# Configure logging
import logging

# ---------- LOGGER SETUP ----------
logging.basicConfig(
    level=logging.INFO,  # can be DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def suggest_resume_improvements(
    resume_text: str,
    model_name: str = "ollama",  # Default to Ollama
    username: str = "user"
) -> dict:
    """
    Analyze and suggest improvements to a resume, using either Gemini or HF.

    Args:
        resume_text: The cleaned text extracted from OCR or PDF.
        model_name:  Either a Gemini model name containing "gemini" (e.g. "gemini-pro")
                     or a HF model path (e.g. "Qwen/Qwen2.5-3B-Instruct").
        api_key:    Your Google API key (required for Gemini).

    Returns:
        A formatted string containing rating, detailed suggestions, and 1-3 project recommendations.
    """
    logger.info("Suggesting Improvements For Resume")

    system_prompt = textwrap.dedent(f"""
        You are an expert resume reviewer and career development assistant. 
        Your task is to **evaluate a given resume strictly based on its content, 
        clarity, conciseness, and impact.** 

        ❗Do not consider formatting, visual design, or layout in your assessment.❗

        **Provide a rating out of 10.** Be critical and unbiased in your assessment. 
        A rating of 10 should signify a resume with exceptional content and clarity. 
        A rating of 5 suggests an average resume; below 5 indicates significant deficiencies.

        After the rating, **suggest detailed changes and ideas for improvement.** 
        Focus on specific sections (summary, experience, skills, education) and offer actionable advice. 
        Consider:
        * **Content:** missing keywords, relevance, quantifiable data?
        * **Clarity & Conciseness:** clear, direct, no unnecessary jargon?
        * **Impact:** are achievements and contributions emphasized effectively?

        Additionally, based on the user's skills and experience, **recommend 1–3 new projects** that:
        * Leverage existing skills.
        * Teach complementary technologies.
        * Are practical and actionable.

        **Do NOT re-provide any resume sections or a full rewrite.** 
        Only output the rating, suggestions, and project recommendations.

        ---
        Resume Content:
        {resume_text}
    """).strip()

    try:
        if "ollama" in model_name.lower():
            # Use Ollama
            response = get_ollama_response(
                prompt="Please review and suggest improvements for the above resume.",
                system_prompt=system_prompt
            )
        elif "gemini" in model_name.lower():
            # Use Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
            llm = init_gemini(model_name, api_key)
            response = llm.generate_content(system_prompt).text
        else:
            # Use Hugging Face model
            tokenizer, model = get_hf_model(model_name)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please review and suggest improvements for the above resume."}
            ]
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([full], return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
            gen = out[0][inputs["input_ids"].shape[-1]:]
            response = tokenizer.decode(gen, skip_special_tokens=True).strip()
            
        # Format the response
        return {
            "username": username,
            "model": model_name,
            "improvements": response
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return {
            "username": username,
            "model": model_name,
            "improvements": (
                "We're currently experiencing issues with our AI service. "
                "Please try again later or contact support if the problem persists.\n\n"
                f"Error: {str(e)}"
            )
        }
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([full], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(gen, skip_special_tokens=True).strip()

    return response


if __name__ == '__main__':
    ocr_text = """
    John Doe
    Email: johndoe@example.com
    Skills: Python, Java, Teamwork
    Experience: Software intern at TechSoft, wrote some code and helped team.
    Projects: Personal website.
    """
    feedback = suggest_resume_improvements(ocr_text, model_name="qwen2.5:3b")
    print(feedback)

