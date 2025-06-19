import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..models.schemas import CoTExample, QAPair

logger = logging.getLogger(__name__)

class ClusteringService:
    def __init__(self):
        self.clusterer: Optional[HDBSCAN] = None
        self.embeddings_matrix: Optional[np.ndarray] = None
        
    def cluster_cot_examples(
        self, 
        cot_examples: List[CoTExample], 
        min_cluster_size: int = 2,
        min_samples: int = 1
    ) -> Tuple[List[CoTExample], Dict[str, any]]:
        """
        Cluster CoT examples using HDBSCAN based on their embeddings
        """
        logger.info(f"Clustering {len(cot_examples)} CoT examples")
        
        # Extract embeddings (assuming they're already generated)
        embeddings = []
        for cot in cot_examples:
            if hasattr(cot, 'embedding') and cot.embedding is not None:
                embeddings.append(cot.embedding)
            else:
                raise ValueError(f"CoT {cot.id} missing embedding")
        
        self.embeddings_matrix = np.vstack(embeddings)
        
        # Apply HDBSCAN clustering
        self.clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'  # HDBSCAN works better with euclidean for embeddings
        )
        
        cluster_labels = self.clusterer.fit_predict(self.embeddings_matrix)
        
        # Calculate outlier scores
        outlier_scores = self.clusterer.outlier_scores_
        
        # Update CoT examples with cluster assignments
        clustered_examples = []
        for i, cot in enumerate(cot_examples):
            updated_cot = cot.model_copy()
            updated_cot.cluster_id = int(cluster_labels[i])
            updated_cot.outlier_score = float(outlier_scores[i])
            clustered_examples.append(updated_cot)
        
        # Generate clustering summary
        unique_clusters = set(cluster_labels)
        outliers = sum(1 for label in cluster_labels if label == -1)
        
        summary = {
            "total_cots": len(cot_examples),
            "num_clusters": len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            "outliers": outliers,
            "cluster_sizes": self._calculate_cluster_sizes(cluster_labels),
            "silhouette_score": self._calculate_silhouette_score() if len(unique_clusters) > 1 else None
        }
        
        logger.info(f"Clustering complete: {summary['num_clusters']} clusters, {outliers} outliers")
        
        return clustered_examples, summary
    
    def _calculate_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[str, int]:
        """Calculate the size of each cluster"""
        cluster_sizes = {}
        for label in cluster_labels:
            cluster_id = str(label)
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        return cluster_sizes
    
    def _calculate_silhouette_score(self) -> float:
        """Calculate silhouette score for clustering quality assessment"""
        try:
            from sklearn.metrics import silhouette_score
            if self.clusterer is not None and self.embeddings_matrix is not None:
                labels = self.clusterer.labels_
                if len(set(labels)) > 1:  # Need at least 2 clusters
                    return float(silhouette_score(self.embeddings_matrix, labels))
            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
            return 0.0
    
    def group_by_qa_pairs(self, cot_examples: List[CoTExample]) -> Dict[str, QAPair]:
        """Group CoT examples by their Q&A pairs"""
        qa_pairs = {}
        
        for cot in cot_examples:
            qa_key = f"{cot.question}|||{cot.answer}"
            
            if qa_key not in qa_pairs:
                qa_pairs[qa_key] = QAPair(
                    question=cot.question,
                    answer=cot.answer,
                    cots=[],
                    clusters=[]
                )
            
            qa_pairs[qa_key].cots.append(cot)
            
            # Track unique clusters for this Q&A pair
            if cot.cluster_id is not None and cot.cluster_id != -1:
                if cot.cluster_id not in qa_pairs[qa_key].clusters:
                    qa_pairs[qa_key].clusters.append(cot.cluster_id)
        
        logger.info(f"Grouped into {len(qa_pairs)} unique Q&A pairs")
        return qa_pairs
    
    def select_representatives(
        self, 
        qa_pairs: Dict[str, QAPair], 
        num_representatives: int = 2
    ) -> List[str]:
        """
        Select Q&A pairs for human labeling based on cluster coverage
        """
        qa_candidates = []
        
        for qa_key, qa_pair in qa_pairs.items():
            # Calculate cluster coverage (how many clusters this Q&A appears in)
            cluster_coverage = len(qa_pair.clusters)
            
            # Find the best (lowest) outlier score among CoTs for this Q&A
            outlier_scores = [cot.outlier_score for cot in qa_pair.cots if cot.outlier_score is not None]
            best_outlier_score = min(outlier_scores) if outlier_scores else 1.0
            
            qa_candidates.append({
                'qa_key': qa_key,
                'cluster_coverage': cluster_coverage,
                'best_outlier_score': best_outlier_score,
                'qa_pair': qa_pair
            })
        
        # Sort by cluster coverage (descending) then by outlier score (ascending)
        qa_candidates.sort(key=lambda x: (-x['cluster_coverage'], x['best_outlier_score']))
        
        # Select top candidates
        selected = qa_candidates[:num_representatives]
        selected_keys = [item['qa_key'] for item in selected]
        
        logger.info(f"Selected {len(selected_keys)} Q&A pairs for human labeling")
        for item in selected:
            logger.info(f"  - Coverage: {item['cluster_coverage']} clusters, Score: {item['best_outlier_score']:.3f}")
        
        return selected_keys
    
    def calculate_similarity_matrix(self, qa_pairs: Dict[str, QAPair]) -> pd.DataFrame:
        """
        Calculate similarity matrix between Q&A pairs based on cluster overlap
        """
        qa_keys = list(qa_pairs.keys())
        n = len(qa_keys)
        similarity_matrix = np.zeros((n, n))
        
        for i, qa_key1 in enumerate(qa_keys):
            for j, qa_key2 in enumerate(qa_keys):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    clusters1 = set(qa_pairs[qa_key1].clusters)
                    clusters2 = set(qa_pairs[qa_key2].clusters)
                    
                    if len(clusters1) == 0 or len(clusters2) == 0:
                        similarity_matrix[i, j] = 0.0
                    else:
                        # Jaccard similarity
                        intersection = len(clusters1.intersection(clusters2))
                        union = len(clusters1.union(clusters2))
                        similarity_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        return pd.DataFrame(
            similarity_matrix,
            index=qa_keys,
            columns=qa_keys
        ) 