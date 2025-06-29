#!/usr/bin/env python3
"""
Reasoning Pattern Clustering for Hallucination Detection
Python implementation of the CoT clustering experiment
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI
import pinecone
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CoTExample:
    id: str
    question: str
    answer: str
    cot: str
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    outlier_score: Optional[float] = None

@dataclass
class QAPair:
    question: str
    answer: str
    cots: List[CoTExample]
    clusters: List[int]
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None

class ReasoningClusterExperiment:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key)
        self.index = self.pinecone_client.Index(pinecone_index)
        self.cot_examples: List[CoTExample] = []
        self.qa_pairs: Dict[str, QAPair] = {}
        
    def load_pure_logic_dataset(self) -> List[CoTExample]:
        """Load the pure logic CoT dataset"""
        pure_logic_data = [
            # Deductive Reasoning Pattern
            {
                'id': 'cot-1',
                'question': 'What is the recommended viscosity?',
                'answer': 'The recommended viscosity and quality grades for engine oil vary based on the vehicle and climate. Check your owner\'s manual for specifics.',
                'cot': 'Given P1: All X have property Y. Given P2: Z is an X. Therefore C: Z has property Y. Since the relationship is universal and Z belongs to the category, the conclusion follows logically.',
            },
            # Add all 15 examples here...
            # (truncated for brevity, but would include all examples)
        ]
        
        self.cot_examples = [CoTExample(**data) for data in pure_logic_data]
        return self.cot_examples
    
    def generate_embeddings(self) -> None:
        """Generate embeddings for all CoT examples"""
        print("Generating embeddings...")
        for cot in self.cot_examples:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=cot.cot
            )
            cot.embedding = np.array(response.data[0].embedding)
        print(f"Generated embeddings for {len(self.cot_examples)} CoT examples")
    
    def upload_to_pinecone(self) -> None:
        """Upload embeddings to Pinecone"""
        print("Uploading to Pinecone...")
        vectors = []
        for cot in self.cot_examples:
            vectors.append({
                'id': cot.id,
                'values': cot.embedding.tolist(),
                'metadata': {
                    'question': cot.question,
                    'answer': cot.answer,
                    'cot': cot.cot
                }
            })
        
        self.index.upsert(vectors)
        print(f"Uploaded {len(vectors)} vectors to Pinecone")
    
    def fetch_from_pinecone(self) -> None:
        """Fetch embeddings from Pinecone"""
        print("Fetching from Pinecone...")
        query_response = self.index.query(
            vector=[0.0] * 1024,  # Dummy vector
            top_k=100,
            include_metadata=True,
            include_values=True
        )
        
        self.cot_examples = []
        for match in query_response.matches:
            cot = CoTExample(
                id=match.id,
                question=match.metadata['question'],
                answer=match.metadata['answer'],
                cot=match.metadata['cot'],
                embedding=np.array(match.values)
            )
            self.cot_examples.append(cot)
        
        print(f"Fetched {len(self.cot_examples)} CoT examples from Pinecone")
    
    def cluster_reasoning_patterns(self, min_cluster_size: int = 2) -> None:
        """Cluster CoT examples using HDBSCAN"""
        print("Clustering reasoning patterns...")
        
        # Extract embeddings matrix
        embeddings_matrix = np.vstack([cot.embedding for cot in self.cot_examples])
        
        # Apply HDBSCAN clustering
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
        cluster_labels = clusterer.fit_predict(embeddings_matrix)
        outlier_scores = clusterer.outlier_scores_
        
        # Assign cluster results to CoT examples
        for i, cot in enumerate(self.cot_examples):
            cot.cluster_id = int(cluster_labels[i])
            cot.outlier_score = float(outlier_scores[i])
        
        # Print clustering results
        unique_clusters = set(cluster_labels)
        outliers = sum(1 for label in cluster_labels if label == -1)
        print(f"Found {len(unique_clusters)} clusters ({len(unique_clusters) - (1 if -1 in unique_clusters else 0)} non-outlier clusters)")
        print(f"Outliers: {outliers}")
    
    def group_by_qa_pairs(self) -> None:
        """Group CoT examples by Q&A pairs"""
        self.qa_pairs = {}
        
        for cot in self.cot_examples:
            qa_key = f"{cot.question}|||{cot.answer}"
            if qa_key not in self.qa_pairs:
                self.qa_pairs[qa_key] = QAPair(
                    question=cot.question,
                    answer=cot.answer,
                    cots=[],
                    clusters=[]
                )
            
            self.qa_pairs[qa_key].cots.append(cot)
            if cot.cluster_id not in self.qa_pairs[qa_key].clusters and cot.cluster_id != -1:
                self.qa_pairs[qa_key].clusters.append(cot.cluster_id)
        
        print(f"Grouped into {len(self.qa_pairs)} unique Q&A pairs")
    
    def select_representatives(self, num_representatives: int = 2) -> List[str]:
        """Select Q&A pairs for human labeling based on cluster coverage"""
        qa_list = []
        
        for qa_key, qa_pair in self.qa_pairs.items():
            cluster_coverage = len(qa_pair.clusters)
            best_outlier_score = min(cot.outlier_score for cot in qa_pair.cots if cot.outlier_score is not None)
            
            qa_list.append({
                'qa_key': qa_key,
                'cluster_coverage': cluster_coverage,
                'best_outlier_score': best_outlier_score,
                'qa_pair': qa_pair
            })
        
        # Sort by cluster coverage (descending) then by outlier score (ascending)
        qa_list.sort(key=lambda x: (-x['cluster_coverage'], x['best_outlier_score']))
        
        selected = qa_list[:num_representatives]
        selected_keys = [item['qa_key'] for item in selected]
        
        print(f"Selected {len(selected_keys)} Q&A pairs for human labeling:")
        for item in selected:
            print(f"  - Clusters {item['qa_pair'].clusters}, Score: {item['best_outlier_score']:.3f}")
        
        return selected_keys
    
    def propagate_labels(self, human_labels: Dict[str, str]) -> None:
        """Propagate labels from human-labeled Q&A pairs to others"""
        print("Propagating labels...")
        
        for qa_key, qa_pair in self.qa_pairs.items():
            if qa_key in human_labels:
                # Direct human label
                qa_pair.predicted_label = human_labels[qa_key]
                qa_pair.confidence = 1.0
                qa_pair.source = 'HUMAN'
            else:
                # Find best propagation match
                best_match = None
                best_overlap = 0
                
                for labeled_key, label in human_labels.items():
                    labeled_clusters = set(self.qa_pairs[labeled_key].clusters)
                    this_clusters = set(qa_pair.clusters)
                    overlap = len(labeled_clusters.intersection(this_clusters))
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = {
                            'label': label,
                            'confidence': overlap / max(len(labeled_clusters), len(this_clusters))
                        }
                
                if best_match and best_match['confidence'] > 0:
                    qa_pair.predicted_label = best_match['label']
                    qa_pair.confidence = max(0.3, min(0.9, best_match['confidence']))
                    qa_pair.source = 'PROPAGATED'
                else:
                    qa_pair.predicted_label = 'uncertain'
                    qa_pair.confidence = 0.0
                    qa_pair.source = 'UNPROPAGATED'
    
    def evaluate_results(self, ground_truth: Dict[str, str]) -> Dict[str, float]:
        """Evaluate propagation results against ground truth"""
        results = {
            'total_pairs': len(self.qa_pairs),
            'human_labeled': 0,
            'propagated': 0,
            'unpropagated': 0,
            'correct_predictions': 0,
            'total_predictions': 0
        }
        
        for qa_key, qa_pair in self.qa_pairs.items():
            if qa_pair.source == 'HUMAN':
                results['human_labeled'] += 1
            elif qa_pair.source == 'PROPAGATED':
                results['propagated'] += 1
            else:
                results['unpropagated'] += 1
            
            if qa_key in ground_truth and qa_pair.predicted_label in ['correct', 'incorrect']:
                results['total_predictions'] += 1
                if qa_pair.predicted_label == ground_truth[qa_key]:
                    results['correct_predictions'] += 1
        
        if results['total_predictions'] > 0:
            results['accuracy'] = results['correct_predictions'] / results['total_predictions']
        else:
            results['accuracy'] = 0.0
        
        return results
    
    def visualize_results(self) -> None:
        """Create visualizations of the clustering and propagation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster distribution
        cluster_counts = {}
        for cot in self.cot_examples:
            cluster_id = cot.cluster_id if cot.cluster_id != -1 else 'Outlier'
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        axes[0, 0].bar(range(len(cluster_counts)), list(cluster_counts.values()))
        axes[0, 0].set_title('CoT Distribution by Cluster')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of CoTs')
        
        # 2. Outlier scores distribution
        outlier_scores = [cot.outlier_score for cot in self.cot_examples if cot.outlier_score is not None]
        axes[0, 1].hist(outlier_scores, bins=20, alpha=0.7)
        axes[0, 1].set_title('Outlier Score Distribution')
        axes[0, 1].set_xlabel('Outlier Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Propagation confidence by source
        sources = []
        confidences = []
        for qa_pair in self.qa_pairs.values():
            if qa_pair.confidence is not None:
                sources.append(qa_pair.source)
                confidences.append(qa_pair.confidence)
        
        df_conf = pd.DataFrame({'Source': sources, 'Confidence': confidences})
        sns.boxplot(data=df_conf, x='Source', y='Confidence', ax=axes[1, 0])
        axes[1, 0].set_title('Confidence by Source Type')
        
        # 4. Cluster coverage vs confidence
        coverages = []
        conf_scores = []
        for qa_pair in self.qa_pairs.values():
            if qa_pair.confidence is not None and qa_pair.source == 'PROPAGATED':
                coverages.append(len(qa_pair.clusters))
                conf_scores.append(qa_pair.confidence)
        
        if coverages and conf_scores:
            axes[1, 1].scatter(coverages, conf_scores, alpha=0.7)
            axes[1, 1].set_title('Cluster Coverage vs Propagation Confidence')
            axes[1, 1].set_xlabel('Number of Clusters')
            axes[1, 1].set_ylabel('Confidence')
        
        plt.tight_layout()
        plt.savefig('reasoning_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results(self, filename: str = 'experiment_results.json') -> None:
        """Export results to JSON for further analysis"""
        results = {
            'cot_examples': [],
            'qa_pairs': {},
            'clustering_summary': {}
        }
        
        # Export CoT examples
        for cot in self.cot_examples:
            results['cot_examples'].append({
                'id': cot.id,
                'question': cot.question,
                'answer': cot.answer,
                'cot': cot.cot,
                'cluster_id': cot.cluster_id,
                'outlier_score': cot.outlier_score
            })
        
        # Export Q&A pairs
        for qa_key, qa_pair in self.qa_pairs.items():
            results['qa_pairs'][qa_key] = {
                'question': qa_pair.question,
                'answer': qa_pair.answer,
                'clusters': qa_pair.clusters,
                'predicted_label': qa_pair.predicted_label,
                'confidence': qa_pair.confidence,
                'source': qa_pair.source,
                'num_cots': len(qa_pair.cots)
            }
        
        # Clustering summary
        cluster_counts = {}
        for cot in self.cot_examples:
            cluster_id = str(cot.cluster_id)
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        results['clustering_summary'] = {
            'total_cots': len(self.cot_examples),
            'total_qa_pairs': len(self.qa_pairs),
            'cluster_counts': cluster_counts,
            'outliers': cluster_counts.get('-1', 0)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filename}")

def main():
    """Run the complete experiment"""
    # Load environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index = os.getenv('PINECONE_INDEX_NAME', 'cot-clustering-test')
    
    if not openai_api_key or not pinecone_api_key:
        raise ValueError("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")
    
    # Initialize experiment
    experiment = ReasoningClusterExperiment(openai_api_key, pinecone_api_key, pinecone_index)
    
    # Load dataset and generate embeddings
    experiment.load_pure_logic_dataset()
    experiment.generate_embeddings()
    experiment.upload_to_pinecone()
    
    # Alternatively, fetch existing embeddings
    # experiment.fetch_from_pinecone()
    
    # Cluster reasoning patterns
    experiment.cluster_reasoning_patterns()
    experiment.group_by_qa_pairs()
    
    # Select representatives for human labeling
    selected_qa_keys = experiment.select_representatives(num_representatives=2)
    
    # Simulate human labels (in practice, this would be interactive)
    human_labels = {
        selected_qa_keys[0]: 'incorrect',  # Towing Q&A
        selected_qa_keys[1]: 'correct'     # Cold Engine Q&A
    }
    
    # Propagate labels
    experiment.propagate_labels(human_labels)
    
    # Ground truth for evaluation
    ground_truth = {
        # Add your ground truth labels here
        # qa_key: 'correct' or 'incorrect'
    }
    
    # Evaluate results
    if ground_truth:
        eval_results = experiment.evaluate_results(ground_truth)
        print("\nEvaluation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    
    # Visualize and export
    experiment.visualize_results()
    experiment.export_results()
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main() 