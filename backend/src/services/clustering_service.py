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
            
            # Sanitize outlier score to handle NaN/inf values
            outlier_score = float(outlier_scores[i])
            if np.isnan(outlier_score) or np.isinf(outlier_score):
                outlier_score = 1.0  # Default value for invalid scores
            updated_cot.outlier_score = outlier_score
            
            clustered_examples.append(updated_cot)
        
        # Generate clustering summary
        unique_clusters = set(cluster_labels)
        outliers = sum(1 for label in cluster_labels if label == -1)
        
        # Calculate silhouette score and sanitize it
        silhouette_score = self._calculate_silhouette_score() if len(unique_clusters) > 1 else 0.0
        if np.isnan(silhouette_score) or np.isinf(silhouette_score):
            silhouette_score = 0.0
        
        summary = {
            "total_examples": len(cot_examples),
            "total_cots": len(cot_examples),
            "num_clusters": len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            "outliers": outliers,
            "cluster_sizes": self._calculate_cluster_sizes(cluster_labels),
            "silhouette_score": float(silhouette_score)
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
                    score = float(silhouette_score(self.embeddings_matrix, labels))
                    # Sanitize the score
                    if np.isnan(score) or np.isinf(score):
                        return 0.0
                    return score
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
        
        # Sort clusters for each Q&A pair for consistency
        for qa_pair in qa_pairs.values():
            qa_pair.clusters.sort()
        
        logger.info(f"Grouped into {len(qa_pairs)} unique Q&A pairs")
        
        # Log cluster distribution
        cluster_counts = {}
        for qa_pair in qa_pairs.values():
            cluster_count = len(qa_pair.clusters)
            cluster_counts[cluster_count] = cluster_counts.get(cluster_count, 0) + 1
        
        logger.info(f"Cluster distribution per Q&A pair: {dict(sorted(cluster_counts.items()))}")
        
        return qa_pairs
    
    def select_diverse_representatives(
        self, 
        qa_pairs: Dict[str, QAPair], 
        target_count: int = 30
    ) -> List[str]:
        """
        Select ~30 representative Q&A pairs based on semantic diversity and cluster coverage.
        
        Selection criteria:
        - Pick 1-2 high-trust CoTs from each cluster (closest to centroid)
        - Prioritize well-written, logically sound, distinct reasoning patterns
        - Avoid extremely similar or templated pairs unless they're core exemplars
        - Ensure coverage across all clusters
        """
        logger.info(f"Selecting {target_count} diverse representatives from {len(qa_pairs)} Q&A pairs")
        
        # Step 1: Group Q&A pairs by their cluster membership
        cluster_qa_map = {}
        
        for qa_key, qa_pair in qa_pairs.items():
            # Get the best (lowest outlier score) CoT for this Q&A pair
            best_cot = min(qa_pair.cots, key=lambda cot: cot.outlier_score or 1.0)
            
            # Skip if no valid clusters (all outliers)
            valid_clusters = [c for c in qa_pair.clusters if c != -1]
            if not valid_clusters:
                continue
            
            # Primary cluster is the one with the lowest outlier score CoT
            primary_cluster = best_cot.cluster_id
            if primary_cluster == -1:
                continue
                
            if primary_cluster not in cluster_qa_map:
                cluster_qa_map[primary_cluster] = []
            
            # Calculate quality metrics for this Q&A pair
            quality_score = self._calculate_qa_quality_score(qa_pair, best_cot)
            
            cluster_qa_map[primary_cluster].append({
                'qa_key': qa_key,
                'qa_pair': qa_pair,
                'best_cot': best_cot,
                'outlier_score': best_cot.outlier_score or 1.0,
                'quality_score': quality_score,
                'cluster_coverage': len(valid_clusters),
                'cot_length': len(best_cot.cot),
                'reasoning_complexity': self._assess_reasoning_complexity(best_cot.cot)
            })
        
        # Step 2: Select representatives from each cluster
        selected_representatives = []
        total_clusters = len(cluster_qa_map)
        
        if total_clusters == 0:
            logger.warning("No valid clusters found for representative selection")
            return []
        
        # Calculate how many representatives per cluster
        base_per_cluster = max(1, target_count // total_clusters)
        remaining_slots = target_count - (base_per_cluster * total_clusters)
        
        logger.info(f"Selecting from {total_clusters} clusters: {base_per_cluster} per cluster + {remaining_slots} bonus")
        
        cluster_selections = {}
        
        for cluster_id, candidates in cluster_qa_map.items():
            # Sort candidates by combined quality and trust score
            candidates.sort(key=lambda x: (
                -x['quality_score'],          # Higher quality first
                x['outlier_score'],           # Lower outlier score (closer to centroid)
                -x['reasoning_complexity'],   # More complex reasoning preferred
                -x['cluster_coverage']        # Better cluster coverage
            ))
            
            # Select base representatives for this cluster
            cluster_selections[cluster_id] = candidates[:base_per_cluster]
            selected_representatives.extend(cluster_selections[cluster_id])
        
        # Step 3: Fill remaining slots with highest quality candidates
        if remaining_slots > 0:
            all_remaining = []
            for cluster_id, candidates in cluster_qa_map.items():
                remaining_candidates = candidates[base_per_cluster:]
                all_remaining.extend(remaining_candidates)
            
            # Sort all remaining by quality and select best
            all_remaining.sort(key=lambda x: (
                -x['quality_score'],
                x['outlier_score'],
                -x['reasoning_complexity']
            ))
            
            bonus_selections = all_remaining[:remaining_slots]
            selected_representatives.extend(bonus_selections)
        
        # Step 4: Apply diversity filter to avoid extremely similar pairs
        final_representatives = self._apply_diversity_filter(selected_representatives, target_count)
        
        # Extract just the qa_keys
        selected_keys = [rep['qa_key'] for rep in final_representatives]
        
        # Log selection summary
        self._log_selection_summary(final_representatives, cluster_qa_map)
        
        return selected_keys
    
    def _calculate_qa_quality_score(self, qa_pair: QAPair, best_cot) -> float:
        """Calculate a quality score for a Q&A pair based on multiple factors"""
        score = 0.0
        
        # Factor 1: Reasoning length (moderate length preferred)
        cot_length = len(best_cot.cot)
        if 100 <= cot_length <= 500:  # Sweet spot for reasoning length
            score += 0.3
        elif 50 <= cot_length <= 800:  # Acceptable range
            score += 0.2
        
        # Factor 2: Question quality (avoid very short or templated questions)
        question_length = len(qa_pair.question)
        if 20 <= question_length <= 150:
            score += 0.2
        
        # Factor 3: Answer specificity (avoid overly generic answers)
        answer_length = len(qa_pair.answer)
        if 30 <= answer_length <= 300:
            score += 0.2
        
        # Factor 4: Reasoning pattern indicators
        cot_text = best_cot.cot.lower()
        
        # Prefer reasoning with logical structure
        logical_indicators = ['because', 'therefore', 'since', 'given', 'if', 'then', 'thus', 'hence']
        logic_score = sum(1 for indicator in logical_indicators if indicator in cot_text)
        score += min(0.15, logic_score * 0.03)
        
        # Prefer step-by-step reasoning
        step_indicators = ['first', 'second', 'next', 'then', 'finally', 'step']
        step_score = sum(1 for indicator in step_indicators if indicator in cot_text)
        score += min(0.15, step_score * 0.03)
        
        return min(1.0, score)  # Cap at 1.0
    
    def _assess_reasoning_complexity(self, cot_text: str) -> float:
        """Assess the complexity/sophistication of reasoning"""
        complexity_score = 0.0
        cot_lower = cot_text.lower()
        
        # Complex reasoning indicators
        complex_terms = [
            'analysis', 'consider', 'examine', 'evaluate', 'determine', 'compare',
            'contrast', 'implication', 'consequence', 'relationship', 'pattern',
            'assumption', 'hypothesis', 'inference', 'deduction', 'induction'
        ]
        
        for term in complex_terms:
            if term in cot_lower:
                complexity_score += 0.1
        
        # Multi-step reasoning
        sentences = cot_text.split('.')
        if len(sentences) > 3:
            complexity_score += 0.2
        
        return min(1.0, complexity_score)
    
    def _apply_diversity_filter(self, candidates: List[Dict], target_count: int) -> List[Dict]:
        """Apply diversity filtering to avoid extremely similar Q&A pairs"""
        if len(candidates) <= target_count:
            return candidates
        
        if not candidates:
            return []
        
        # Use a simple diversity heuristic based on text similarity
        selected = []
        remaining = candidates.copy()
        
        # Always include the highest quality candidate
        if remaining:
            selected.append(remaining.pop(0))
        
        while len(selected) < target_count and remaining:
            best_candidate = None
            best_diversity_score = -1
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # Calculate diversity score (average dissimilarity to selected items)
                diversity_score = 0.0
                
                for selected_item in selected:
                    # Simple text-based diversity heuristic
                    similarity = self._calculate_text_similarity(
                        candidate['best_cot'].cot,
                        selected_item['best_cot'].cot
                    )
                    diversity_score += (1.0 - similarity)
                
                avg_diversity = diversity_score / len(selected) if selected else 0.0
                combined_score = 0.7 * candidate['quality_score'] + 0.3 * avg_diversity
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _log_selection_summary(self, selected: List[Dict], cluster_map: Dict) -> None:
        """Log a summary of the selection results"""
        if not selected:
            logger.warning("No representatives selected")
            return
            
        cluster_counts = {}
        total_quality = 0.0
        total_trust = 0.0
        
        for rep in selected:
            cluster_id = rep['best_cot'].cluster_id
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            total_quality += rep['quality_score']
            total_trust += (1.0 - rep['outlier_score'])
        
        avg_quality = total_quality / len(selected) if selected else 0.0
        avg_trust = total_trust / len(selected) if selected else 0.0
        
        logger.info(f"Selected {len(selected)} representatives:")
        logger.info(f"  Average quality score: {avg_quality:.3f}")
        logger.info(f"  Average trust score: {avg_trust:.3f}")
        logger.info(f"  Cluster coverage: {len(cluster_counts)}/{len(cluster_map)} clusters")
        logger.info(f"  Per-cluster distribution: {dict(sorted(cluster_counts.items()))}")
    
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
    
    def experiment_with_parameters(
        self, 
        cot_examples: List[CoTExample]
    ) -> Dict[str, Dict]:
        """
        Experiment with different HDBSCAN parameters to find optimal settings
        """
        logger.info("Experimenting with different HDBSCAN parameters...")
        
        # Extract embeddings
        embeddings = []
        for cot in cot_examples:
            if hasattr(cot, 'embedding') and cot.embedding is not None:
                embeddings.append(cot.embedding)
            else:
                raise ValueError(f"CoT {cot.id} missing embedding")
        
        embeddings_matrix = np.vstack(embeddings)
        
        # Parameter combinations to test
        param_combinations = [
            # More lenient settings for subtle differences
            {'min_cluster_size': 3, 'min_samples': 2, 'description': 'Slightly larger clusters, low density'},
            {'min_cluster_size': 4, 'min_samples': 2, 'description': 'Medium clusters, low density'},
            {'min_cluster_size': 5, 'min_samples': 3, 'description': 'Larger clusters, medium density'},
            
            # Alternative approaches
            {'min_cluster_size': 2, 'min_samples': 2, 'description': 'Small clusters, higher density'},
            {'min_cluster_size': 3, 'min_samples': 3, 'description': 'Balanced approach'},
            
            # More aggressive clustering for subtle patterns
            {'min_cluster_size': 6, 'min_samples': 2, 'description': 'Large clusters, low density (captures subtle patterns)'},
            {'min_cluster_size': 8, 'min_samples': 3, 'description': 'Very large clusters, medium density'},
        ]
        
        results = {}
        
        for params in param_combinations:
            try:
                clusterer = HDBSCAN(
                    min_cluster_size=params['min_cluster_size'],
                    min_samples=params['min_samples'],
                    metric='euclidean',
                    cluster_selection_method='eom'  # Can also try 'leaf' for more granular clusters
                )
                
                cluster_labels = clusterer.fit_predict(embeddings_matrix)
                
                # Calculate metrics
                unique_clusters = set(cluster_labels)
                num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                outliers = sum(1 for label in cluster_labels if label == -1)
                outlier_percentage = (outliers / len(cluster_labels)) * 100
                
                # Calculate cluster sizes
                cluster_sizes = []
                for cluster_id in unique_clusters:
                    if cluster_id != -1:
                        size = sum(1 for label in cluster_labels if label == cluster_id)
                        cluster_sizes.append(size)
                
                param_key = f"min_cluster_size_{params['min_cluster_size']}_min_samples_{params['min_samples']}"
                
                results[param_key] = {
                    'parameters': params,
                    'num_clusters': num_clusters,
                    'outliers': outliers,
                    'outlier_percentage': round(outlier_percentage, 1),
                    'cluster_sizes': cluster_sizes,
                    'avg_cluster_size': round(np.mean(cluster_sizes), 1) if cluster_sizes else 0,
                    'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                    'min_cluster_size_actual': min(cluster_sizes) if cluster_sizes else 0
                }
                
                logger.info(f"{params['description']}: {num_clusters} clusters, {outlier_percentage:.1f}% outliers")
                
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")
                continue
        
        return results 
    
    def experiment_alternative_methods(
        self, 
        cot_examples: List[CoTExample]
    ) -> Dict[str, Dict]:
        """
        Experiment with alternative clustering methods beyond HDBSCAN
        """
        logger.info("Experimenting with alternative clustering methods...")
        
        # Extract embeddings
        embeddings = []
        for cot in cot_examples:
            if hasattr(cot, 'embedding') and cot.embedding is not None:
                embeddings.append(cot.embedding)
            else:
                raise ValueError(f"CoT {cot.id} missing embedding")
        
        embeddings_matrix = np.vstack(embeddings)
        results = {}
        
        # 1. K-Means with different k values
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        for k in [10, 15, 20, 25, 30]:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings_matrix)
                
                silhouette = silhouette_score(embeddings_matrix, cluster_labels)
                cluster_sizes = [sum(1 for label in cluster_labels if label == i) for i in range(k)]
                
                results[f"kmeans_k_{k}"] = {
                    'method': 'K-Means',
                    'parameters': {'k': k},
                    'num_clusters': k,
                    'outliers': 0,  # K-means doesn't have outliers
                    'outlier_percentage': 0.0,
                    'cluster_sizes': cluster_sizes,
                    'avg_cluster_size': round(np.mean(cluster_sizes), 1),
                    'silhouette_score': round(silhouette, 3),
                    'min_cluster_size': min(cluster_sizes),
                    'max_cluster_size': max(cluster_sizes)
                }
            except Exception as e:
                logger.error(f"Error with K-means k={k}: {e}")
        
        # 2. Agglomerative Clustering with different linkages
        from sklearn.cluster import AgglomerativeClustering
        
        for n_clusters in [15, 20, 25]:
            for linkage in ['ward', 'complete', 'average']:
                try:
                    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                    cluster_labels = agg.fit_predict(embeddings_matrix)
                    
                    silhouette = silhouette_score(embeddings_matrix, cluster_labels)
                    cluster_sizes = [sum(1 for label in cluster_labels if label == i) for i in range(n_clusters)]
                    
                    results[f"agglomerative_{linkage}_{n_clusters}"] = {
                        'method': 'Agglomerative',
                        'parameters': {'n_clusters': n_clusters, 'linkage': linkage},
                        'num_clusters': n_clusters,
                        'outliers': 0,
                        'outlier_percentage': 0.0,
                        'cluster_sizes': cluster_sizes,
                        'avg_cluster_size': round(np.mean(cluster_sizes), 1),
                        'silhouette_score': round(silhouette, 3),
                        'min_cluster_size': min(cluster_sizes),
                        'max_cluster_size': max(cluster_sizes)
                    }
                except Exception as e:
                    logger.error(f"Error with Agglomerative {linkage} n_clusters={n_clusters}: {e}")
        
        # 3. HDBSCAN with different metrics
        for metric in ['cosine', 'manhattan']:
            for min_cluster_size in [3, 5, 8]:
                try:
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=2,
                        metric=metric
                    )
                    cluster_labels = clusterer.fit_predict(embeddings_matrix)
                    
                    unique_clusters = set(cluster_labels)
                    num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
                    outliers = sum(1 for label in cluster_labels if label == -1)
                    outlier_percentage = (outliers / len(cluster_labels)) * 100
                    
                    cluster_sizes = []
                    for cluster_id in unique_clusters:
                        if cluster_id != -1:
                            size = sum(1 for label in cluster_labels if label == cluster_id)
                            cluster_sizes.append(size)
                    
                    results[f"hdbscan_{metric}_{min_cluster_size}"] = {
                        'method': 'HDBSCAN',
                        'parameters': {'min_cluster_size': min_cluster_size, 'metric': metric},
                        'num_clusters': num_clusters,
                        'outliers': outliers,
                        'outlier_percentage': round(outlier_percentage, 1),
                        'cluster_sizes': cluster_sizes,
                        'avg_cluster_size': round(np.mean(cluster_sizes), 1) if cluster_sizes else 0,
                        'silhouette_score': 0.0,  # Calculate separately if needed
                        'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                        'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0
                    }
                except Exception as e:
                    logger.error(f"Error with HDBSCAN {metric} min_cluster_size={min_cluster_size}: {e}")
        
        return results 
    
    def cluster_cot_examples_kmeans(
        self, 
        cot_examples: List[CoTExample], 
        n_clusters: int = 20
    ) -> Tuple[List[CoTExample], Dict[str, any]]:
        """
        Cluster CoT examples using K-Means (recommended for CoT data)
        """
        logger.info(f"Clustering {len(cot_examples)} CoT examples using K-Means")
        
        # Extract embeddings
        embeddings = []
        for cot in cot_examples:
            if hasattr(cot, 'embedding') and cot.embedding is not None:
                embeddings.append(cot.embedding)
            else:
                raise ValueError(f"CoT {cot.id} missing embedding")
        
        self.embeddings_matrix = np.vstack(embeddings)
        
        # Apply K-Means clustering
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.clusterer.fit_predict(self.embeddings_matrix)
        
        # Calculate distances to centroids for outlier-like scoring
        distances = np.min(self.clusterer.transform(self.embeddings_matrix), axis=1)
        # Normalize distances to 0-1 range for outlier scores
        max_distance = np.max(distances)
        outlier_scores = distances / max_distance if max_distance > 0 else np.zeros_like(distances)
        
        # Update CoT examples with cluster assignments
        clustered_examples = []
        for i, cot in enumerate(cot_examples):
            updated_cot = cot.model_copy()
            updated_cot.cluster_id = int(cluster_labels[i])
            updated_cot.outlier_score = float(outlier_scores[i])
            clustered_examples.append(updated_cot)
        
        # Generate clustering summary
        unique_clusters = set(cluster_labels)
        silhouette_score_val = silhouette_score(self.embeddings_matrix, cluster_labels)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for label in cluster_labels:
            cluster_id = str(label)
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        summary = {
            "total_examples": len(cot_examples),
            "total_cots": len(cot_examples),
            "num_clusters": len(unique_clusters),
            "outliers": 0,  # K-means doesn't produce outliers
            "cluster_sizes": cluster_sizes,
            "silhouette_score": float(silhouette_score_val),
            "avg_cluster_size": len(cot_examples) / n_clusters,
            "method": "K-Means"
        }
        
        logger.info(f"K-Means clustering complete: {n_clusters} clusters, silhouette score: {silhouette_score_val:.3f}")
        
        return clustered_examples, summary 
    
    def select_representatives(
        self, 
        qa_pairs: Dict[str, QAPair], 
        num_representatives: int = 2
    ) -> List[str]:
        """
        Select Q&A pairs for human labeling based on cluster coverage (original method)
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