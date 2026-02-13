import re
import json

def text_processing(raw_text: str):
    print("Preprocessing OCR output...")
    text = re.sub(r"(?mi)^page\s*\d+\s*(?:of\s*\d+)?\s*$", "", raw_text)
    text = re.sub(r"[-_]{2,}", "", text)
    text = re.sub(r"^\s*[29e¢«•*▪➢➔\"=]\s+", "", text, flags=re.M)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    allowed_chars_pattern = r"[^a-zA-Z0-9.,?!'\":;()\-\[\]{}<>_+\=\/&%$#@*`~\|\\\s]"
    text = re.sub(allowed_chars_pattern, "-", text)
    text = text.replace("\t", " ")
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()

def extract_json(text):
    # Find the JSON part between the first '{' and last '}'
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)  # Convert to Python dict
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return None
    else:
        print("No JSON found in text")
        return None


