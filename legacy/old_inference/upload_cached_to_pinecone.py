#!/usr/bin/env python3
"""
Upload cached CoTs and embeddings to Pinecone using proper client
"""

import json
import os
from pinecone import Pinecone, ServerlessSpec
import time

def main():
    # Load environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'cot-embeddings')
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return
    
    # Load cached data
    cache_file = "inference_cache/inference_cots_exact.json"
    if not os.path.exists(cache_file):
        print("❌ No cached data found. Run generate_inference_cots_exact.py first.")
        return
    
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    print(f"📂 Loaded cached data with {len(cache_data['pinecone_vectors'])} vectors")
    
    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        print("✅ Pinecone initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone: {e}")
        return
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if pinecone_index_name not in existing_indexes:
        print(f"Creating index {pinecone_index_name}...")
        pc.create_index(
            name=pinecone_index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        time.sleep(10)  # Wait for index to be ready
        print(f"✅ Created index: {pinecone_index_name}")
    else:
        print(f"✅ Index {pinecone_index_name} exists")
    
    # Get index
    index = pc.Index(pinecone_index_name)
    
    # Clear index
    try:
        index.delete(delete_all=True)
        print('✅ Pinecone index cleared successfully')
        time.sleep(2)  # Wait for deletion to complete
    except Exception as delete_error:
        print(f'Note: Could not clear index: {delete_error}')
    
    # Upload vectors
    vectors = cache_data['pinecone_vectors']
    print(f"📤 Uploading {len(vectors)} vectors to Pinecone...")
    
    try:
        # Upload in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"   Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            time.sleep(1)  # Small delay between batches
        
        print('✅ All CoT embeddings uploaded to Pinecone successfully!')
        
        # Verify upload
        stats = index.describe_index_stats()
        print(f"📊 Index stats: {stats['total_vector_count']} vectors in index")
        
    except Exception as e:
        print(f"❌ Failed to upload to Pinecone: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 