#!/usr/bin/env python3
"""
Check Pinecone Namespaces
Lists all available namespaces in the Pinecone index
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path for imports
sys.path.append('./backend')

def check_pinecone_namespaces():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'cot-clustering')
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        return
    
    print(f"🔍 Checking Pinecone index: {pinecone_index_name}")
    
    try:
        # Use new Pinecone API (version 7.x+)
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        
        print("\n📊 Index Statistics:")
        print(f"Total vectors: {stats.total_vector_count}")
        print(f"Index fullness: {stats.index_fullness}")
        
        if hasattr(stats, 'namespaces') and stats.namespaces:
            print(f"\n🏷️ Available Namespaces ({len(stats.namespaces)}):")
            for namespace_name, namespace_stats in stats.namespaces.items():
                print(f"  - '{namespace_name}': {namespace_stats.vector_count} vectors")
        else:
            print("\n🏷️ No namespaces found or all vectors are in the default namespace")
            
        # Check default namespace
        if not hasattr(stats, 'namespaces') or '' not in stats.namespaces:
            print("\n💡 Checking default namespace...")
            try:
                # Try to query the default namespace
                dummy_vector = [0.0] * 1536
                query_response = index.query(
                    vector=dummy_vector,
                    top_k=1,
                    include_metadata=True,
                    namespace=""  # Default namespace
                )
                if query_response.matches:
                    print(f"✅ Default namespace has vectors")
                else:
                    print("❌ No vectors found in default namespace")
            except Exception as e:
                print(f"❌ Error querying default namespace: {e}")
        
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_pinecone_namespaces() 