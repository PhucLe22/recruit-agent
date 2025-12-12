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

# Read the OCR text
with open("ocr_text/page_01.txt", "r") as f:
    ocr_text = f.read()

# Update the CV with the actual text
result = cvs_collection.update_one(
    {"username": "PhucLe"},
    {"$set": {"processed_text": ocr_text}}
)

print(f"Updated {result.modified_count} documents")
print(f"Text length: {len(ocr_text)}")
print("First 200 characters:")
print(ocr_text[:200])
