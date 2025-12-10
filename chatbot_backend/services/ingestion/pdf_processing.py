from pdf2image import convert_from_path
import shutil
import os

poppler_path = r'C:\Users\Daryn Bang\PycharmProjects\poppler-24.08.0\Library\bin'

def convert_pdf_to_img(pdf_path, dpi=100, img_dir="pages_img", poppler_path=None):
    # Ensure img_dir exists
    os.makedirs(img_dir, exist_ok=True)

    # Empty img_dir before saving new images
    for filename in os.listdir(img_dir):
        file_path = os.path.join(img_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # remove file/symlink
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # remove folder

    # Convert PDF pages to images
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    for idx, page in enumerate(pages, start=1):
        img_path = os.path.join(img_dir, f"page_{idx:02d}.png")
        page.save(img_path, "PNG")
        print(f"Saved {img_path}")

