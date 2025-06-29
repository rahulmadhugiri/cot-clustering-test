#!/usr/bin/env python3
"""
Inference Test Pipeline for Advanced Binary Choice Classifier
Tests the model on 30 completely new, unseen Q&A pairs
WITH CACHING: Saves CoTs and embeddings locally before uploading to Pinecone
"""

import json
import numpy as np
import torch
import os
import sys
from typing import List, Dict, Tuple
import requests
from tqdm import tqdm
import time
from datetime import datetime

# Add backend to path
sys.path.append('./backend')

from original_binary_choice_classifier import BinaryChoiceClassifier

class InferenceTestPipeline:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'cot-embeddings')
        
        # Cache file paths
        self.cache_dir = "inference_cache"
        self.cots_cache_file = f"{self.cache_dir}/inference_cots.json"
        self.embeddings_cache_file = f"{self.cache_dir}/inference_embeddings.npz"
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
    
    def load_test_set(self) -> List[Dict]:
        """Load the 30 new test questions"""
        with open('inference_test_set.json', 'r') as f:
            return json.load(f)
    
    def save_cots_to_cache(self, test_data: List[Dict], positive_cots: List[str], negative_cots: List[str]):
        """Save generated CoTs to local cache"""
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'test_data': test_data,
            'positive_cots': positive_cots,
            'negative_cots': negative_cots
        }
        
        with open(self.cots_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"💾 CoTs saved to cache: {self.cots_cache_file}")
    
    def load_cots_from_cache(self) -> Tuple[List[Dict], List[str], List[str]]:
        """Load CoTs from local cache"""
        if not os.path.exists(self.cots_cache_file):
            return None, None, None
        
        with open(self.cots_cache_file, 'r') as f:
            cache_data = json.load(f)
        
        print(f"📂 Loaded CoTs from cache: {self.cots_cache_file}")
        print(f"   Cache timestamp: {cache_data['timestamp']}")
        
        return cache_data['test_data'], cache_data['positive_cots'], cache_data['negative_cots']
    
    def save_embeddings_to_cache(self, pos_embeddings: np.ndarray, neg_embeddings: np.ndarray):
        """Save generated embeddings to local cache"""
        np.savez(
            self.embeddings_cache_file,
            positive_embeddings=pos_embeddings,
            negative_embeddings=neg_embeddings,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"💾 Embeddings saved to cache: {self.embeddings_cache_file}")
    
    def load_embeddings_from_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings from local cache"""
        if not os.path.exists(self.embeddings_cache_file):
            return None, None
        
        cache_data = np.load(self.embeddings_cache_file, allow_pickle=True)
        
        print(f"📂 Loaded embeddings from cache: {self.embeddings_cache_file}")
        if 'timestamp' in cache_data:
            print(f"   Cache timestamp: {cache_data['timestamp'].item()}")
        
        return cache_data['positive_embeddings'], cache_data['negative_embeddings']
    
    def upload_cached_data_to_pinecone(self):
        """Upload cached CoTs and embeddings to Pinecone (retry functionality)"""
        print("🔄 Attempting to upload cached data to Pinecone...")
        
        # Load cached data
        test_data, positive_cots, negative_cots = self.load_cots_from_cache()
        pos_embeddings, neg_embeddings = self.load_embeddings_from_cache()
        
        if test_data is None or pos_embeddings is None:
            print("❌ No cached data found. Please run the pipeline first.")
            return False
        
        try:
            # Prepare all embeddings and metadata
            all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
            metadata = []
            for i, item in enumerate(test_data):
                metadata.extend([
                    {
                        'type': 'positive_cot',
                        'question': item['question'],
                        'answer': item['answer'],
                        'cot': positive_cots[i],
                        'test_id': i
                    },
                    {
                        'type': 'negative_cot', 
                        'question': item['question'],
                        'answer': item['answer'],
                        'cot': negative_cots[i],
                        'test_id': i
                    }
                ])
            
            self.upload_to_pinecone(all_embeddings, metadata)
            print("✅ Successfully uploaded cached data to Pinecone!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload to Pinecone: {e}")
            return False

    def generate_dual_cots(self, question: str, answer: str) -> Tuple[str, str]:
        """Generate positive and negative CoTs for a Q&A pair"""
        
        # Positive CoT prompt (logical, step-by-step reasoning)
        positive_prompt = f"""Given this question and answer, provide a clear, logical chain-of-thought reasoning that leads to the correct answer.

Question: {question}
Answer: {answer}

Provide step-by-step logical reasoning (2-3 sentences) that would lead someone to this answer:"""

        # Negative CoT prompt (flawed reasoning, hallucinations)
        negative_prompt = f"""Given this question and answer, provide a flawed chain-of-thought reasoning that contains logical errors, incorrect assumptions, or hallucinated information.

Question: {question}
Answer: {answer}

Provide flawed step-by-step reasoning (2-3 sentences) that contains errors or hallucinations:"""

        positive_cot = self._call_openai(positive_prompt)
        negative_cot = self._call_openai(negative_prompt)
        
        return positive_cot, negative_cot
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate CoT"""
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant that generates reasoning chains.'},
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 200,
            'temperature': 0.7
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        headers = {
            'Authorization': f'Bearer {self.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings"):
            data = {
                'model': 'text-embedding-3-large',
                'input': text
            }
            
            response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                embedding = response.json()['data'][0]['embedding']
                embeddings.append(embedding)
            else:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
            time.sleep(0.1)  # Rate limiting
        
        return np.array(embeddings)
    
    def upload_to_pinecone(self, embeddings: np.ndarray, metadata: List[Dict], namespace: str = "inference_test"):
        """Upload embeddings to Pinecone"""
        try:
            # Try new Pinecone API first
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_api_key)
            index = pc.Index(self.pinecone_index_name)
        except ImportError:
            # Fall back to old Pinecone API
            import pinecone
            pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
            index = pinecone.Index(self.pinecone_index_name)
        
        # Prepare vectors for upload
        vectors = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            vectors.append({
                'id': f"inference_test_{i}",
                'values': embedding.tolist(),
                'metadata': meta
            })
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
        
        print(f"Uploaded {len(vectors)} vectors to Pinecone namespace: {namespace}")
    
    def load_trained_model(self) -> BinaryChoiceClassifier:
        """Load the trained Binary Choice Classifier"""
        # Load cached embeddings to get embedding dimension
        try:
            dual_cache = np.load('backend/dual_embeddings_cache.npz')
            pos_embeddings = dual_cache['pos_embeddings']
            embedding_dim = pos_embeddings.shape[1]
        except FileNotFoundError:
            # Default embedding dimension for text-embedding-3-small
            embedding_dim = 1024
        
        # Initialize model
        model = BinaryChoiceClassifier(embedding_dim=embedding_dim)
        
        # Load trained weights
        try:
            model.load_state_dict(torch.load('backend/best_binary_choice_model.pth'))
            model.eval()
            print("✅ Loaded trained Binary Choice Classifier")
            return model
        except FileNotFoundError:
            raise FileNotFoundError("Trained model not found. Please train the model first.")
    
    def run_inference(self, pos_embeddings: np.ndarray, neg_embeddings: np.ndarray) -> np.ndarray:
        """Run inference on the test set"""
        model = self.load_trained_model()
        
        # Convert to tensors
        pos_tensor = torch.FloatTensor(pos_embeddings)
        neg_tensor = torch.FloatTensor(neg_embeddings)
        
        predictions = []
        
        with torch.no_grad():
            for i in range(len(pos_embeddings)):
                pos_emb = pos_tensor[i:i+1]
                neg_emb = neg_tensor[i:i+1]
                
                # Get model prediction
                outputs = model(pos_emb, neg_emb)
                choice_probs = torch.softmax(outputs['choice_logits'], dim=1)
                
                # Choice 0 = positive CoT, Choice 1 = negative CoT
                # If model chooses positive CoT (0), it predicts "Not Hallucinated"
                # If model chooses negative CoT (1), it predicts "Hallucinated"
                predicted_choice = torch.argmax(choice_probs, dim=1).item()
                
                predictions.append(predicted_choice)
        
        return np.array(predictions)
    
    def evaluate_results(self, predictions: np.ndarray, ground_truth_labels: List[str]) -> Dict:
        """Evaluate the model's performance"""
        # Convert ground truth to binary (0 = Not Hallucinated, 1 = Hallucinated)
        gt_binary = [1 if label == "Hallucinated" else 0 for label in ground_truth_labels]
        
        # Calculate metrics
        correct = sum(pred == gt for pred, gt in zip(predictions, gt_binary))
        accuracy = correct / len(predictions)
        
        # Detailed breakdown
        true_positives = sum(pred == 1 and gt == 1 for pred, gt in zip(predictions, gt_binary))
        false_positives = sum(pred == 1 and gt == 0 for pred, gt in zip(predictions, gt_binary))
        true_negatives = sum(pred == 0 and gt == 0 for pred, gt in zip(predictions, gt_binary))
        false_negatives = sum(pred == 0 and gt == 1 for pred, gt in zip(predictions, gt_binary))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(predictions),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'predictions': predictions.tolist(),
            'ground_truth': gt_binary
        }
    
    def run_complete_pipeline(self):
        """Run the complete inference test pipeline"""
        print("🚀 Starting Inference Test Pipeline for Advanced Binary Choice Classifier")
        print("=" * 80)
        
        # Step 1: Load test set
        print("📂 Loading test set...")
        test_data = self.load_test_set()
        print(f"✅ Loaded {len(test_data)} test examples")
        
        # Step 2: Load CoTs from cache
        print("\n🧠 Loading CoTs from cache...")
        cached_test_data, cached_positive_cots, cached_negative_cots = self.load_cots_from_cache()
        
        if cached_test_data is None:
            # Step 3: Generate dual CoTs
            print("\n🧠 Generating dual CoTs...")
            positive_cots = []
            negative_cots = []
            
            for i, item in enumerate(tqdm(test_data, desc="Generating CoTs")):
                pos_cot, neg_cot = self.generate_dual_cots(item['question'], item['answer'])
                positive_cots.append(pos_cot)
                negative_cots.append(neg_cot)
                time.sleep(0.5)  # Rate limiting
            
            print(f"✅ Generated {len(positive_cots)} positive and {len(negative_cots)} negative CoTs")
            
            # Step 4: Save CoTs to cache
            self.save_cots_to_cache(test_data, positive_cots, negative_cots)
        else:
            print(f"✅ Loaded {len(cached_test_data)} cached test examples")
            positive_cots = cached_positive_cots
            negative_cots = cached_negative_cots
        
        # Step 5: Load embeddings from cache or generate new ones
        print("\n🔢 Loading embeddings from cache...")
        cached_pos_embeddings, cached_neg_embeddings = self.load_embeddings_from_cache()
        
        if cached_pos_embeddings is None:
            print("\n🔢 Generating embeddings...")
            all_texts = positive_cots + negative_cots
            all_embeddings = self.generate_embeddings(all_texts)
            
            # Split embeddings
            pos_embeddings = all_embeddings[:len(positive_cots)]
            neg_embeddings = all_embeddings[len(positive_cots):]
            
            print(f"✅ Generated embeddings: {pos_embeddings.shape}")
            
            # Step 6: Save embeddings to cache
            self.save_embeddings_to_cache(pos_embeddings, neg_embeddings)
        else:
            print(f"✅ Loaded cached embeddings: {cached_pos_embeddings.shape}")
            pos_embeddings = cached_pos_embeddings
            neg_embeddings = cached_neg_embeddings
        
        # Step 7: Upload to Pinecone (optional, for record keeping)
        print("\n📤 Uploading to Pinecone...")
        try:
            all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
            metadata = []
            for i, item in enumerate(test_data):
                metadata.extend([
                    {
                        'type': 'positive_cot',
                        'question': item['question'],
                        'answer': item['answer'],
                        'cot': positive_cots[i],
                        'test_id': i
                    },
                    {
                        'type': 'negative_cot', 
                        'question': item['question'],
                        'answer': item['answer'],
                        'cot': negative_cots[i],
                        'test_id': i
                    }
                ])
            
            self.upload_to_pinecone(all_embeddings, metadata)
            print("✅ Successfully uploaded to Pinecone!")
            
        except Exception as e:
            print(f"⚠️ Failed to upload to Pinecone: {e}")
            print("💡 Don't worry! Your CoTs and embeddings are safely cached.")
            print("   You can retry the Pinecone upload later using:")
            print("   pipeline.upload_cached_data_to_pinecone()")
            print("   Continuing with inference...")
        
        # Step 8: Run inference (MODEL NEVER SEES LABELS!)
        print("\n🤖 Running inference with Advanced Binary Choice Classifier...")
        predictions = self.run_inference(pos_embeddings, neg_embeddings)
        
        # Step 9: Evaluate results
        print("\n📊 Evaluating results...")
        ground_truth_labels = [item['ground_truth_label'] for item in test_data]
        results = self.evaluate_results(predictions, ground_truth_labels)
        
        # Step 10: Display results
        print("\n" + "=" * 80)
        print("🎯 INFERENCE TEST RESULTS")
        print("=" * 80)
        print(f"Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
        print(f"Precision: {results['precision']:.3f}")
        print(f"Recall: {results['recall']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"True Positives (Correctly identified hallucinations): {results['true_positives']}")
        print(f"False Positives (Incorrectly flagged as hallucinations): {results['false_positives']}")
        print(f"True Negatives (Correctly identified non-hallucinations): {results['true_negatives']}")
        print(f"False Negatives (Missed hallucinations): {results['false_negatives']}")
        
        # Step 11: Save detailed results
        detailed_results = {
            'test_data': test_data,
            'positive_cots': positive_cots,
            'negative_cots': negative_cots,
            'predictions': results['predictions'],
            'ground_truth': results['ground_truth'],
            'metrics': results
        }
        
        with open('inference_test_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: inference_test_results.json")
        print("\n✅ Inference test pipeline completed!")
        
        return results

if __name__ == "__main__":
    pipeline = InferenceTestPipeline()
    results = pipeline.run_complete_pipeline() 