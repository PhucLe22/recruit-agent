#!/usr/bin/env python3

import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGO_ATLAS_URI = os.getenv("MONGO_ATLAS_URI")

# Connect to MongoDB
client = MongoClient(MONGO_ATLAS_URI)
db = client["CVProject"]
cvs_collection = db["cvs"]

# Read both OCR pages
with open("ocr_text/page_01.txt", "r") as f:
    page1 = f.read()

with open("ocr_text/page_02.txt", "r") as f:
    page2 = f.read()

# Combine both pages
full_text = page1 + "\n\n" + page2

# Update CV with full text
result = cvs_collection.update_one(
    {"username": "PhucLe"},
    {"$set": {"processed_text": full_text}}
)

print(f"Updated {result.modified_count} documents")
