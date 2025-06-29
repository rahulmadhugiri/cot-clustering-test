#!/usr/bin/env python3
"""
Simplified Inference Test for Advanced Binary Choice Classifier
Tests the model on 30 completely new, unseen Q&A pairs
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append('./backend')

from original_binary_choice_classifier import BinaryChoiceClassifier

class SimpleInferenceTest:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    def load_test_set(self) -> List[Dict]:
        """Load the 30 new test questions"""
        with open('inference_test_set.json', 'r') as f:
            return json.load(f)
    
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
                'model': 'text-embedding-3-small',
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
    
    def load_trained_model(self) -> BinaryChoiceClassifier:
        """Load the trained Advanced Binary Choice Classifier"""
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
            model.load_state_dict(torch.load('backend/best_advanced_binary_choice_model.pth'))
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
    
    def run_test(self):
        """Run the complete inference test"""
        print("🚀 Starting Simplified Inference Test for Binary Choice Classifier")
        print("=" * 80)
        
        # Step 1: Load test set
        print("📂 Loading test set...")
        test_data = self.load_test_set()
        print(f"✅ Loaded {len(test_data)} test examples")
        
        # Step 2: Generate dual CoTs
        print("\n🧠 Generating dual CoTs...")
        positive_cots = []
        negative_cots = []
        
        for i, item in enumerate(tqdm(test_data, desc="Generating CoTs")):
            pos_cot, neg_cot = self.generate_dual_cots(item['question'], item['answer'])
            positive_cots.append(pos_cot)
            negative_cots.append(neg_cot)
            time.sleep(0.5)  # Rate limiting
        
        print(f"✅ Generated {len(positive_cots)} positive and {len(negative_cots)} negative CoTs")
        
        # Step 3: Generate embeddings
        print("\n🔢 Generating embeddings...")
        all_texts = positive_cots + negative_cots
        all_embeddings = self.generate_embeddings(all_texts)
        
        # Split embeddings
        pos_embeddings = all_embeddings[:len(positive_cots)]
        neg_embeddings = all_embeddings[len(positive_cots):]
        
        print(f"✅ Generated embeddings: {pos_embeddings.shape}")
        
        # Step 4: Run inference (MODEL NEVER SEES LABELS!)
        print("\n🤖 Running inference with Binary Choice Classifier...")
        predictions = self.run_inference(pos_embeddings, neg_embeddings)
        
        # Step 5: Evaluate results
        print("\n📊 Evaluating results...")
        ground_truth_labels = [item['ground_truth_label'] for item in test_data]
        results = self.evaluate_results(predictions, ground_truth_labels)
        
        # Step 6: Display results
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
        
        # Step 7: Save detailed results
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
        print("\n✅ Inference test completed!")
        
        return results

if __name__ == "__main__":
    test = SimpleInferenceTest()
    results = test.run_test() 