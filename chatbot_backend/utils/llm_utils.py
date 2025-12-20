# llm_utils.py
import logging
import os
import json
import requests
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# ——————————————————————————————————————————————————————
# Global caches
# ——————————————————————————————————————————————————————
_HF_CACHE = {}
_GEMINI_INIT = set()

# Ollama configuration
OLLAMA_API_BASE = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"

def get_ollama_response(prompt: str, model: str = None, system_prompt: str = None) -> str:
    """
    Get a response from the local Ollama API.
    
    Args:
        prompt: The user's input prompt
        model: The Ollama model to use (default: OLLAMA_DEFAULT_MODEL)
        system_prompt: Optional system prompt to guide the model's behavior
        
    Returns:
        The generated text response
    """
    if model is None:
        model = OLLAMA_DEFAULT_MODEL
        
    url = f"{OLLAMA_API_BASE}/chat/completions"
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=600  # 5 minute timeout
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error calling Ollama API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        raise

def get_hf_model(model_name: str):
    """
    Returns (tokenizer, model), loading & caching them on first call.
    """
    if model_name not in _HF_CACHE:
        try:
            # Try to load with GPU first
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # For smaller models, we can use CPU if needed
            if device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            if device == "cpu":
                model = model.to(device)
                
            _HF_CACHE[model_name] = (tokenizer, model)
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
            
    return _HF_CACHE[model_name]

def init_gemini(model_name: str, api_key: str):
    """
    Configures google.generativeai once per api_key, then returns
    a GenerativeModel handle.
    """
    import google.generativeai as genai

    if api_key not in _GEMINI_INIT:
        genai.configure(api_key=api_key)
        _GEMINI_INIT.add(api_key)

    return genai.GenerativeModel(model_name=model_name)
