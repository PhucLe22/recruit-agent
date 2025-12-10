import os
from dotenv import load_dotenv
from pymongo import MongoClient

def test_connection():
    # Load environment variables
    load_dotenv()
    
    # Get connection string
    mongo_uri = os.getenv("MONGO_ATLAS_URI")
    print(f"Using MongoDB URI: {mongo_uri}")
    
    if not mongo_uri:
        print("Error: MONGO_ATLAS_URI not found in environment variables")
        return
    
    try:
        # Try to connect with the connection string
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        
        # Test the connection
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
        
        # List databases
        print("\nAvailable databases:")
        for db in client.list_database_names():
            print(f"- {db}")
            
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")

if __name__ == "__main__":
    test_connection()
