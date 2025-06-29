#!/usr/bin/env python3
"""
Inspect Pinecone Data Structure
Examines the actual structure of vectors and metadata in Pinecone
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path for imports
sys.path.append('./backend')

def inspect_pinecone_data():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'cot-clustering-test')
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return
    
    print(f"🔍 Inspecting Pinecone index: {pinecone_index_name}")
    
    try:
        # Use new Pinecone API (version 7.x+)
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        
        # Query a few vectors to see their structure
        dummy_vector = [0.0] * 1024  # Use correct dimension
        query_response = index.query(
            vector=dummy_vector,
            top_k=5,  # Just get a few examples
            include_metadata=True,
            include_values=True,
            namespace=""  # Default namespace
        )
        
        print(f"\n📊 Retrieved {len(query_response.matches)} sample vectors")
        
        for i, match in enumerate(query_response.matches):
            print(f"\n--- Vector {i+1} ---")
            print(f"ID: {match.id}")
            print(f"Score: {match.score}")
            print(f"Values shape: {len(match.values) if match.values else 'None'}")
            print(f"Metadata: {match.metadata}")
            
            if i >= 2:  # Just show first 3 for brevity
                break
        
        # Try to get some specific IDs that might exist
        print(f"\n🔍 Trying to fetch specific vector IDs...")
        
        # Common ID patterns to try
        id_patterns = [
            "0", "1", "2",  # Simple numbers
            "test_0", "test_1", "test_2",  # test_ prefix
            "cot_0", "cot_1", "cot_2",  # cot_ prefix
            "inference_0", "inference_1", "inference_2",  # inference_ prefix
        ]
        
        for pattern in id_patterns:
            try:
                fetch_response = index.fetch(ids=[pattern], namespace="")
                if fetch_response.vectors:
                    print(f"✅ Found vector with ID: {pattern}")
                    vector = list(fetch_response.vectors.values())[0]
                    print(f"   Metadata: {vector.metadata}")
                    break
            except Exception as e:
                continue
        
    except Exception as e:
        print(f"❌ Error inspecting Pinecone: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_pinecone_data() 