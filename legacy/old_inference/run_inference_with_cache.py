#!/usr/bin/env python3
"""
Run Inference Test with Caching Support
Demonstrates how to use the caching functionality to avoid losing expensive API calls
"""

import os
import sys
from inference_test_pipeline import InferenceTestPipeline

def main():
    print("🚀 Inference Test with Caching Support")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    if not os.getenv('PINECONE_API_KEY'):
        print("❌ PINECONE_API_KEY environment variable not set")
        return
    
    # Initialize pipeline
    pipeline = InferenceTestPipeline()
    
    print("\nOptions:")
    print("1. Run complete pipeline (with caching)")
    print("2. Retry Pinecone upload from cache")
    print("3. Check cache status")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🏃 Running complete inference pipeline...")
        try:
            results = pipeline.run_complete_pipeline()
            print(f"\n🎯 Final Accuracy: {results['accuracy']:.1%}")
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            print("\n💡 If OpenAI API calls succeeded, your data is cached!")
            print("   Check cache status with option 3")
    
    elif choice == "2":
        print("\n🔄 Attempting to upload cached data to Pinecone...")
        success = pipeline.upload_cached_data_to_pinecone()
        if success:
            print("✅ Upload successful!")
        else:
            print("❌ Upload failed. Check your Pinecone configuration.")
    
    elif choice == "3":
        print("\n📋 Cache Status:")
        
        # Check CoTs cache
        test_data, pos_cots, neg_cots = pipeline.load_cots_from_cache()
        if test_data is not None:
            print(f"✅ CoTs Cache: {len(test_data)} examples")
        else:
            print("❌ CoTs Cache: Empty")
        
        # Check embeddings cache
        pos_emb, neg_emb = pipeline.load_embeddings_from_cache()
        if pos_emb is not None:
            print(f"✅ Embeddings Cache: {pos_emb.shape}")
        else:
            print("❌ Embeddings Cache: Empty")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 