import os
import pytesseract
from PIL import Image

def applyOCR(img_dir, ocr_dir):
    os.makedirs(ocr_dir, exist_ok=True)
    final_text = ""
    for img_file in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_file)
        text = pytesseract.image_to_string(Image.open(img_path), lang="eng")
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(ocr_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        final_text += text + "\n"
    return final_text