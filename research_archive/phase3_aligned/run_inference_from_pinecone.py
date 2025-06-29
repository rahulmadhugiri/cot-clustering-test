#!/usr/bin/env python3
"""
Run Inference from Pinecone Data
Retrieves the 30 new Q&A pairs (60 CoTs) from Pinecone and runs inference
on the trained Binary Choice Classifier without making any API calls.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.append('./backend')

from original_binary_choice_classifier import BinaryChoiceClassifier

class PineconeInferenceRunner:
    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'cot-clustering')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
    
    def retrieve_data_from_pinecone(self, namespace: str = "") -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Retrieve all inference test data from Pinecone"""
        namespace_display = "default" if namespace == "" else namespace
        print(f"🔍 Retrieving data from Pinecone namespace: {namespace_display}")
        
        # Use new Pinecone API (version 7.x+)
        from pinecone import Pinecone
        pc = Pinecone(api_key=self.pinecone_api_key)
        index = pc.Index(self.pinecone_index_name)
        
        # Get all vectors from the namespace
        # First, let's get the stats to see how many vectors we have
        stats = index.describe_index_stats()
        if namespace in stats.namespaces:
            vector_count = stats.namespaces[namespace].vector_count
            print(f"📊 Found {vector_count} vectors in namespace '{namespace_display}'")
        else:
            raise ValueError(f"Namespace '{namespace_display}' not found in Pinecone index")
        
        # Query all vectors using the correct dimension
        print("🔄 Querying all vectors...")
        try:
            dummy_vector = [0.0] * 1024  # Use correct dimension for the index
            query_response = index.query(
                vector=dummy_vector,
                top_k=vector_count,
                include_metadata=True,
                include_values=True,
                namespace=namespace
            )
            print(f"✅ Retrieved {len(query_response.matches)} vectors")
        except Exception as e:
            print(f"❌ Error querying Pinecone: {e}")
            raise Exception("Could not retrieve vectors from Pinecone")
        
        # Organize data by qa_id and type
        data_by_qa_id = defaultdict(dict)
        
        for match in query_response.matches:
            metadata = match.metadata
            qa_id = int(metadata['qa_id'])  # Convert to int for sorting
            cot_type = metadata['type']  # 'pos' or 'neg'
            
            # Map 'pos' to 'positive_cot' and 'neg' to 'negative_cot'
            cot_type_full = 'positive_cot' if cot_type == 'pos' else 'negative_cot'
            
            data_by_qa_id[qa_id][cot_type_full] = {
                'embedding': match.values,
                'question': metadata['question'],
                'answer': metadata['answer'],
                'cot': metadata['cot_text']
            }
        
        # Extract embeddings and metadata in order
        pos_embeddings = []
        neg_embeddings = []
        test_data = []
        
        for qa_id in sorted(data_by_qa_id.keys()):
            qa_item = data_by_qa_id[qa_id]
            
            if 'positive_cot' in qa_item and 'negative_cot' in qa_item:
                pos_embeddings.append(qa_item['positive_cot']['embedding'])
                neg_embeddings.append(qa_item['negative_cot']['embedding'])
                
                test_data.append({
                    'qa_id': qa_id,
                    'question': qa_item['positive_cot']['question'],
                    'answer': qa_item['positive_cot']['answer'],
                    'positive_cot': qa_item['positive_cot']['cot'],
                    'negative_cot': qa_item['negative_cot']['cot']
                })
        
        pos_embeddings = np.array(pos_embeddings)
        neg_embeddings = np.array(neg_embeddings)
        
        print(f"✅ Retrieved {len(test_data)} Q&A pairs with embeddings")
        print(f"   Positive embeddings shape: {pos_embeddings.shape}")
        print(f"   Negative embeddings shape: {neg_embeddings.shape}")
        
        return pos_embeddings, neg_embeddings, test_data
    
    def load_ground_truth_labels(self) -> List[str]:
        """Load ground truth labels from the original test set"""
        try:
            with open('inference_test_set.json', 'r') as f:
                test_set = json.load(f)
            return [item['ground_truth_label'] for item in test_set]
        except FileNotFoundError:
            print("⚠️ inference_test_set.json not found. Ground truth labels will not be available.")
            return None
    
    def load_trained_model(self) -> BinaryChoiceClassifier:
        """Load the trained Binary Choice Classifier"""
        # The original model uses 1024-dimensional embeddings
        embedding_dim = 1024
        
        # Initialize model
        model = BinaryChoiceClassifier(embedding_dim=embedding_dim)
        
        # Load trained weights
        try:
            model.load_state_dict(torch.load('backend/best_binary_choice_model.pth', map_location='cpu'))
            model.eval()
            print("✅ Loaded trained Binary Choice Classifier")
            return model
        except FileNotFoundError:
            raise FileNotFoundError("Trained model not found at 'backend/best_binary_choice_model.pth'")
    
    def run_inference(self, pos_embeddings: np.ndarray, neg_embeddings: np.ndarray) -> np.ndarray:
        """Run inference on the test set"""
        print("🤖 Running inference with Binary Choice Classifier...")
        
        model = self.load_trained_model()
        
        # Convert to tensors
        pos_tensor = torch.FloatTensor(pos_embeddings)
        neg_tensor = torch.FloatTensor(neg_embeddings)
        
        predictions = []
        confidences = []
        
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
                confidence = torch.max(choice_probs, dim=1).values.item()
                
                predictions.append(predicted_choice)
                confidences.append(confidence)
        
        print(f"✅ Completed inference on {len(predictions)} examples")
        return np.array(predictions), np.array(confidences)
    
    def evaluate_results(self, predictions: np.ndarray, ground_truth_labels: List[str]) -> Dict:
        """Evaluate the model's performance"""
        if ground_truth_labels is None:
            print("⚠️ No ground truth labels available for evaluation")
            return None
        
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
    
    def display_detailed_results(self, test_data: List[Dict], predictions: np.ndarray, 
                               confidences: np.ndarray, ground_truth_labels: List[str] = None):
        """Display detailed results for each prediction"""
        print("\n" + "=" * 100)
        print("📋 DETAILED INFERENCE RESULTS")
        print("=" * 100)
        
        for i, (item, pred, conf) in enumerate(zip(test_data, predictions, confidences)):
            prediction_label = "Hallucinated" if pred == 1 else "Not Hallucinated"
            
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {item['question']}")
            print(f"Answer: {item['answer']}")
            print(f"Positive CoT: {item['positive_cot']}")
            print(f"Negative CoT: {item['negative_cot']}")
            print(f"Model Prediction: {prediction_label} (confidence: {conf:.3f})")
            
            if ground_truth_labels:
                gt_label = ground_truth_labels[i]
                correct = "✅" if (pred == 1 and gt_label == "Hallucinated") or (pred == 0 and gt_label == "Not Hallucinated") else "❌"
                print(f"Ground Truth: {gt_label} {correct}")
    
    def run_inference_test(self):
        """Run the complete inference test using data from Pinecone"""
        print("🚀 Running Inference Test from Pinecone Data")
        print("=" * 80)
        
        try:
            # Step 1: Retrieve data from Pinecone
            pos_embeddings, neg_embeddings, test_data = self.retrieve_data_from_pinecone()
            
            # Step 2: Load ground truth labels (if available)
            ground_truth_labels = self.load_ground_truth_labels()
            
            # Step 3: Run inference
            predictions, confidences = self.run_inference(pos_embeddings, neg_embeddings)
            
            # Step 4: Evaluate results (if ground truth is available)
            if ground_truth_labels:
                results = self.evaluate_results(predictions, ground_truth_labels)
                
                print("\n" + "=" * 80)
                print("🎯 INFERENCE TEST RESULTS")
                print("=" * 80)
                print(f"Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
                print(f"Precision: {results['precision']:.3f}")
                print(f"Recall: {results['recall']:.3f}")
                print(f"F1 Score: {results['f1_score']:.3f}")
                print(f"True Positives: {results['true_positives']}")
                print(f"False Positives: {results['false_positives']}")
                print(f"True Negatives: {results['true_negatives']}")
                print(f"False Negatives: {results['false_negatives']}")
            
            # Step 5: Display detailed results
            show_details = input("\nShow detailed results for each example? (y/n): ").strip().lower()
            if show_details == 'y':
                self.display_detailed_results(test_data, predictions, confidences, ground_truth_labels)
            
            print("\n✅ Inference test completed successfully!")
            
            # Return results for programmatic use
            return {
                'predictions': predictions,
                'confidences': confidences,
                'test_data': test_data,
                'evaluation_results': results if ground_truth_labels else None
            }
            
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the inference test"""
    runner = PineconeInferenceRunner()
    results = runner.run_inference_test()
    
    if results:
        print(f"\n🎉 Successfully completed inference on {len(results['predictions'])} examples")
    else:
        print("\n💥 Inference test failed")

if __name__ == "__main__":
    main() 