#!/usr/bin/env python3
"""
Generate CoTs and Embeddings for Inference Test Data
Uses the EXACT same logic as app/api/generate-dual-cots/route.js
"""

import json
import numpy as np
import os
import sys
import requests
import re
import time
from typing import List, Dict, Tuple

class ExactCoTGenerator:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'cot-embeddings')
        
        # Cache file paths
        self.cache_dir = "inference_cache"
        self.cots_cache_file = f"{self.cache_dir}/inference_cots_exact.json"
        self.embeddings_cache_file = f"{self.cache_dir}/inference_embeddings_exact.npz"
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
    
    def load_test_set(self) -> List[Dict]:
        """Load the 30 new test questions"""
        with open('inference_test_set.json', 'r') as f:
            test_data = json.load(f)
        
        # Convert to the format expected by the route.js logic
        qa_data = []
        for i, item in enumerate(test_data):
            qa_data.append({
                'id': i,
                'q': item['question'],
                'a': item['answer']
            })
        
        return qa_data
    
    def generate_dual_cots_exact(self, question: str, answer: str) -> Tuple[str, str]:
        """Generate dual CoTs using the EXACT same logic as route.js"""
        
        # EXACT system content from route.js
        system_content = """For each question and answer pair, generate two distinct Chain of Thought (CoT) reasoning paths:
• One assuming the answer is NOT HALLUCINATED (i.e., grounded and reliable).
• One assuming the answer is HALLUCINATED (i.e., flawed, misleading, or incomplete).

These CoTs should:
1. Reflect different reasoning perspectives, not just list pros vs. cons. Use diverse lenses like legal/ethical responsibility, technical accuracy, user helpfulness, safety implications, tone, and specificity.
2. Vary in structure — not all CoTs should follow the same formula (e.g., context → detail → conclusion). Some may begin with broader framing, others with technical critique, others with implications.
3. Introduce subtle reasoning differences across examples, even for the same label. This variation will help expose what makes certain CoTs persuasive while others fall short.
4. Be plausible and human-like in both directions. The "hallucinated" CoT should not be a caricature or obviously wrong. Instead, it should sound believable while still exhibiting subtle flaws in logic, helpfulness, or grounding.
5. Avoid directly copying language across examples. Each CoT should be independently reasoned."""
        
        # EXACT user content from route.js
        user_content = f"""You are labeling Q&A data using reasoning chains.

Given a question and answer pair, produce:
• A 5-step reasoning chain assuming the answer is NOT hallucinated
• A 5-step reasoning chain assuming the answer is hallucinated

IMPORTANT: Each reasoning should take a distinct perspective, vary in how it argues its position, and be written as if a human is trying to justify their judgment. Include nuance. Don't just say "it's vague" or "it's grounded" — explain why in human terms.

Format your response exactly as:
HALLUCINATED: [Your 5-step reasoning chain for why this answer might be hallucinated]
NOT_HALLUCINATED: [Your 5-step reasoning chain for why this answer is not hallucinated]

Question: {question}

Answer: {answer}"""
        
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        # EXACT parameters from route.js
        data = {
            'model': 'gpt-4.1',
            'messages': [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content}
            ],
            'temperature': 0.3,
            'top_p': 1.0,
            'max_tokens': 10000
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            cot_content = response.json()['choices'][0]['message']['content']
            
            # EXACT parsing logic from route.js
            hallucinated_match = re.search(r'HALLUCINATED:\s*(.+?)(?=NOT_HALLUCINATED:|$)', cot_content, re.DOTALL)
            not_hallucinated_match = re.search(r'NOT_HALLUCINATED:\s*(.+?)$', cot_content, re.DOTALL)
            
            cot_pos = hallucinated_match.group(1).strip() if hallucinated_match else "This answer appears to be fabricated or incorrect based on the context."
            cot_neg = not_hallucinated_match.group(1).strip() if not_hallucinated_match else "This answer appears to be factual and correct based on the context."
            
            return cot_pos, cot_neg
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    def generate_embeddings_exact(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the EXACT same parameters as route.js"""
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        # EXACT parameters from route.js
        data = {
            'model': 'text-embedding-3-small',
            'input': texts,
            'dimensions': 1024
        }
        
        response = requests.post(
            'https://api.openai.com/v1/embeddings',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            embeddings_data = response.json()['data']
            return [item['embedding'] for item in embeddings_data]
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    def upload_to_pinecone_exact(self, vectors: List[Dict]):
        """Upload to Pinecone using the EXACT same logic as route.js"""
        try:
            # Try new Pinecone API first
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists
            index_list = pc.list_indexes()
            index_exists = any(idx.name == self.pinecone_index_name for idx in index_list.indexes)
            
            if not index_exists:
                print(f"Index {self.pinecone_index_name} doesn't exist. Creating it...")
                
                # EXACT index creation parameters from route.js
                pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=1024,  # text-embedding-3-small with 1024 dimensions
                    metric='cosine',
                    spec={
                        'serverless': {
                            'cloud': 'aws',
                            'region': 'us-east-1'
                        }
                    }
                )
                
                print(f"✅ Created new Pinecone index: {self.pinecone_index_name}")
                
                # Wait a moment for the index to be ready
                time.sleep(2)
            else:
                print(f"Index {self.pinecone_index_name} exists. Clearing it...")
                
            index = pc.Index(self.pinecone_index_name)
            
            # Try to clear the index (same logic as route.js)
            try:
                index.delete(delete_all=True)
                print('✅ Pinecone index cleared successfully')
            except Exception as delete_error:
                print(f'Note: Could not clear index (it may already be empty): {delete_error}')
                # Continue anyway - the index exists and we can use it
            
            # Upload all vectors to Pinecone
            print(f"Uploading {len(vectors)} vectors to Pinecone...")
            if len(vectors) > 0:
                index.upsert(vectors=vectors)
                print('✅ All CoT embeddings uploaded to Pinecone')
                
        except ImportError:
            # Fall back to old Pinecone API
            import pinecone
            pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
            
            # Similar logic but with old API
            if self.pinecone_index_name not in pinecone.list_indexes():
                print(f"Creating index {self.pinecone_index_name}...")
                pinecone.create_index(
                    name=self.pinecone_index_name,
                    dimension=1024,
                    metric='cosine'
                )
                time.sleep(2)
            
            index = pinecone.Index(self.pinecone_index_name)
            
            try:
                index.delete(delete_all=True)
                print('✅ Pinecone index cleared successfully')
            except Exception as delete_error:
                print(f'Note: Could not clear index: {delete_error}')
            
            print(f"Uploading {len(vectors)} vectors to Pinecone...")
            if len(vectors) > 0:
                index.upsert(vectors=vectors)
                print('✅ All CoT embeddings uploaded to Pinecone')
    
    def save_to_cache(self, qa_data: List[Dict], data_with_cots: List[Dict], pinecone_vectors: List[Dict]):
        """Save generated data to cache"""
        cache_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'qa_data': qa_data,
            'data_with_cots': data_with_cots,
            'pinecone_vectors': pinecone_vectors
        }
        
        with open(self.cots_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"💾 Data saved to cache: {self.cots_cache_file}")
    
    def load_from_cache(self):
        """Load data from cache"""
        if not os.path.exists(self.cots_cache_file):
            return None
        
        with open(self.cots_cache_file, 'r') as f:
            cache_data = json.load(f)
        
        print(f"📂 Loaded data from cache: {self.cots_cache_file}")
        print(f"   Cache timestamp: {cache_data['timestamp']}")
        
        return cache_data
    
    def run_complete_generation(self):
        """Run the complete CoT generation process using EXACT route.js logic"""
        print("🚀 Starting CoT Generation with EXACT route.js Logic")
        print("=" * 60)
        
        # Check if we have cached data
        cached_data = self.load_from_cache()
        if cached_data:
            print("✅ Found cached data. Using cached CoTs and embeddings.")
            print("   If you want to regenerate, delete the cache file first.")
            
            # Try to upload cached data to Pinecone
            try:
                self.upload_to_pinecone_exact(cached_data['pinecone_vectors'])
                return cached_data['data_with_cots']
            except Exception as e:
                print(f"⚠️ Failed to upload cached data to Pinecone: {e}")
                return cached_data['data_with_cots']
        
        # Step 1: Load test set
        print("📂 Loading test set...")
        qa_data = self.load_test_set()
        print(f"✅ Loaded {len(qa_data)} Q&A pairs")
        
        # Step 2: Generate CoTs and embeddings (EXACT route.js logic)
        data_with_cots = []
        pinecone_vectors = []
        
        print('\n🧠 Starting CoT generation for Q&A pairs...')
        
        for qa in qa_data:
            try:
                print(f"Generating CoTs for Q&A {qa['id']}...")
                
                # Generate dual CoTs using EXACT route.js logic
                cot_pos, cot_neg = self.generate_dual_cots_exact(qa['q'], qa['a'])
                
                # Generate embeddings using EXACT route.js parameters
                embeddings = self.generate_embeddings_exact([cot_pos, cot_neg])
                pos_embedding = embeddings[0]
                neg_embedding = embeddings[1]
                
                # Prepare vectors for Pinecone with EXACT structure from route.js
                pinecone_vectors.extend([
                    {
                        'id': f"cot_{qa['id']}_pos",
                        'values': pos_embedding,
                        'metadata': {
                            'qa_id': qa['id'],
                            'type': 'pos',
                            'cot_text': cot_pos,
                            'question': qa['q'],
                            'answer': qa['a'],
                            'is_labeled': False
                        }
                    },
                    {
                        'id': f"cot_{qa['id']}_neg",
                        'values': neg_embedding,
                        'metadata': {
                            'qa_id': qa['id'],
                            'type': 'neg',
                            'cot_text': cot_neg,
                            'question': qa['q'],
                            'answer': qa['a'],
                            'is_labeled': False
                        }
                    }
                ])
                
                data_with_cots.append({
                    **qa,
                    'cot_pos': cot_pos,
                    'cot_neg': cot_neg
                })
                
                # EXACT delay from route.js
                time.sleep(0.2)
                
            except Exception as error:
                print(f"Error generating CoTs for Q&A {qa['id']}: {error}")
                
                # EXACT fallback logic from route.js
                fallback_pos = "Unable to generate reasoning trace for hallucinated scenario."
                fallback_neg = "Unable to generate reasoning trace for non-hallucinated scenario."
                
                try:
                    # Still try to generate embeddings for fallback CoTs
                    embeddings = self.generate_embeddings_exact([fallback_pos, fallback_neg])
                    pos_embedding = embeddings[0]
                    neg_embedding = embeddings[1]
                    
                    pinecone_vectors.extend([
                        {
                            'id': f"cot_{qa['id']}_pos",
                            'values': pos_embedding,
                            'metadata': {
                                'qa_id': qa['id'],
                                'type': 'pos',
                                'cot_text': fallback_pos,
                                'question': qa['q'],
                                'answer': qa['a'],
                                'is_labeled': False,
                                'is_fallback': True
                            }
                        },
                        {
                            'id': f"cot_{qa['id']}_neg",
                            'values': neg_embedding,
                            'metadata': {
                                'qa_id': qa['id'],
                                'type': 'neg',
                                'cot_text': fallback_neg,
                                'question': qa['q'],
                                'answer': qa['a'],
                                'is_labeled': False,
                                'is_fallback': True
                            }
                        }
                    ])
                except Exception as embedding_error:
                    print(f"Failed to generate embeddings for fallback CoTs for Q&A {qa['id']}: {embedding_error}")
                
                data_with_cots.append({
                    **qa,
                    'cot_pos': fallback_pos,
                    'cot_neg': fallback_neg
                })
        
        print(f"✅ Generated CoTs for {len(data_with_cots)} Q&A pairs")
        
        # Step 3: Save to cache before uploading to Pinecone
        self.save_to_cache(qa_data, data_with_cots, pinecone_vectors)
        
        # Step 4: Upload to Pinecone
        try:
            self.upload_to_pinecone_exact(pinecone_vectors)
        except Exception as e:
            print(f"⚠️ Failed to upload to Pinecone: {e}")
            print("💡 Don't worry! Your CoTs and embeddings are safely cached.")
            print("   You can retry the upload later.")
        
        print(f"\n✅ Process completed! Generated dual CoTs for {len(data_with_cots)} Q&A pairs")
        print(f"📊 Stored {len(pinecone_vectors)} embeddings")
        
        return data_with_cots

if __name__ == "__main__":
    generator = ExactCoTGenerator()
    results = generator.run_complete_generation() 