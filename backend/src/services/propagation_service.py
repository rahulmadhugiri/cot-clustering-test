import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..models.schemas import QAPair

logger = logging.getLogger(__name__)

class PropagationService:
    def __init__(self):
        pass
    
    def propagate_labels(
        self,
        qa_pairs: Dict[str, QAPair],
        human_labels: Dict[str, str],
        representatives: List[str]
    ) -> Tuple[Dict[str, QAPair], Dict[str, any]]:
        """
        Propagate human labels to unlabeled Q&A pairs based on reasoning pattern similarity
        """
        logger.info(f"Propagating labels from {len(human_labels)} human-labeled pairs")
        logger.info(f"Using {len(representatives)} representatives for propagation")
        logger.info(f"Total Q&A pairs to process: {len(qa_pairs)}")
        
        # Debug: Check representative coverage
        valid_representatives = [rep for rep in representatives if rep in qa_pairs and rep in human_labels]
        logger.info(f"Valid representatives with both qa_pairs and human_labels: {len(valid_representatives)}")
        
        updated_qa_pairs = {}
        propagation_stats = {
            "human_labeled": 0,
            "propagated": 0,
            "unpropagated": 0,
            "outliers": 0
        }
        
        for qa_key, qa_pair in qa_pairs.items():
            updated_pair = qa_pair.model_copy(deep=True)
            
            if qa_key in human_labels:
                # Direct human label
                updated_pair.predicted_label = human_labels[qa_key]
                updated_pair.confidence = 1.0
                updated_pair.source = "HUMAN"
                propagation_stats["human_labeled"] += 1
                
            else:
                # Try to propagate from human-labeled pairs
                propagation_result = self._find_best_propagation_match(
                    qa_pair, qa_pairs, human_labels, representatives
                )
                
                if propagation_result["success"]:
                    updated_pair.predicted_label = propagation_result["label"]
                    updated_pair.confidence = propagation_result["confidence"]
                    updated_pair.source = "PROPAGATED"
                    updated_pair.propagation_source = propagation_result["source_qa"]
                    updated_pair.shared_reasoning = propagation_result["shared_clusters"]
                    propagation_stats["propagated"] += 1
                    
                else:
                    # For K-Means clustering, items aren't true outliers
                    # They just don't have sufficient overlap with labeled representatives
                    updated_pair.predicted_label = "uncertain"
                    updated_pair.confidence = 0.0
                    
                    # Check if this has any valid clusters (not -1)
                    if self._has_only_outlier_clusters(qa_pair):
                        updated_pair.source = "OUTLIER"
                        propagation_stats["outliers"] += 1
                    else:
                        # Has valid clusters but no good propagation match
                        updated_pair.source = "UNPROPAGATED"
                        propagation_stats["unpropagated"] += 1
            
            updated_qa_pairs[qa_key] = updated_pair
        
        summary = {
            **propagation_stats,
            "total_pairs": len(qa_pairs),
            "coverage_rate": (propagation_stats["human_labeled"] + propagation_stats["propagated"]) / len(qa_pairs),
            "automation_rate": propagation_stats["propagated"] / len(qa_pairs) if len(qa_pairs) > 0 else 0.0
        }
        
        logger.info(f"Propagation complete: {summary}")
        return updated_qa_pairs, summary
    
    def _find_best_propagation_match(
        self,
        target_qa: QAPair,
        all_qa_pairs: Dict[str, QAPair],
        human_labels: Dict[str, str],
        representatives: List[str]
    ) -> Dict[str, any]:
        """
        Find the best human-labeled Q&A pair to propagate from based on cluster overlap
        """
        target_clusters = set(target_qa.clusters)
        
        if not target_clusters or (len(target_clusters) == 1 and -1 in target_clusters):
            return {"success": False, "reason": "No valid clusters"}
        
        best_match = None
        best_overlap = 0
        best_confidence = 0.0
        
        matches_found = 0
        representatives_checked = 0
        
        for rep_qa_key in representatives:
            representatives_checked += 1
            
            if rep_qa_key in human_labels and rep_qa_key in all_qa_pairs:
                rep_qa = all_qa_pairs[rep_qa_key]
                rep_clusters = set(rep_qa.clusters)
                
                # Calculate cluster overlap
                intersection = target_clusters.intersection(rep_clusters)
                overlap_count = len(intersection)
                
                if overlap_count > 0:  # Any overlap is good for propagation
                    matches_found += 1
                    
                    # Calculate confidence based on Jaccard similarity
                    union = target_clusters.union(rep_clusters)
                    jaccard_similarity = len(intersection) / len(union) if union else 0.0
                    
                    # More lenient confidence calculation for K-Means results
                    # Since K-Means doesn't have outliers, we can be more confident in propagation
                    base_confidence = 0.5 + (jaccard_similarity * 0.4)  # Range: 0.5 to 0.9
                    
                    # Boost confidence if there's significant overlap
                    if overlap_count >= 2:
                        base_confidence += 0.1
                    if len(intersection) == len(target_clusters):  # Target is subset of representative
                        base_confidence += 0.1
                    
                    confidence = min(0.95, max(0.3, base_confidence))
                    
                    # Sanitize confidence value
                    if np.isnan(confidence) or np.isinf(confidence):
                        confidence = 0.5  # Default confidence
                    
                    # Update best match if this is better
                    if overlap_count > best_overlap or (overlap_count == best_overlap and confidence > best_confidence):
                        best_overlap = overlap_count
                        best_confidence = confidence
                        best_match = {
                            "label": human_labels[rep_qa_key],
                            "confidence": float(confidence),
                            "source_qa": rep_qa_key,
                            "shared_clusters": ", ".join([f"Cluster {c}" for c in sorted(intersection)]),
                            "overlap_count": overlap_count,
                            "jaccard_similarity": jaccard_similarity
                        }
        
        # Debug logging for failed matches
        if not best_match:
            logger.debug(f"No propagation match found: {representatives_checked} representatives checked, "
                        f"{matches_found} had cluster overlap, target clusters: {sorted(target_clusters)}")
        
        # Lower the threshold for propagation success
        if best_match and best_match["confidence"] >= 0.3:  # Reduced from previous threshold
            return {"success": True, **best_match}
        else:
            reason = "No matches found" if not best_match else f"Best confidence {best_match['confidence']:.3f} below threshold"
            return {"success": False, "reason": reason}
    
    def _has_only_outlier_clusters(self, qa_pair: QAPair) -> bool:
        """Check if Q&A pair has only outlier clusters (-1)"""
        valid_clusters = [c for c in qa_pair.clusters if c != -1]
        return len(valid_clusters) == 0
    
    def evaluate_propagation(
        self,
        qa_pairs: Dict[str, QAPair],
        ground_truth: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Evaluate propagation results against ground truth labels
        """
        if not ground_truth:
            return {}
        
        metrics = {
            "total_pairs": len(qa_pairs),
            "ground_truth_available": len(ground_truth),
            "human_labeled": 0,
            "propagated": 0,
            "correct_human": 0,
            "correct_propagated": 0,
            "total_predictions": 0,
            "correct_predictions": 0
        }
        
        for qa_key, qa_pair in qa_pairs.items():
            if qa_key in ground_truth and qa_pair.predicted_label in ["correct", "incorrect"]:
                metrics["total_predictions"] += 1
                
                if qa_pair.source == "HUMAN":
                    metrics["human_labeled"] += 1
                    if qa_pair.predicted_label == ground_truth[qa_key]:
                        metrics["correct_human"] += 1
                        metrics["correct_predictions"] += 1
                        
                elif qa_pair.source == "PROPAGATED":
                    metrics["propagated"] += 1
                    if qa_pair.predicted_label == ground_truth[qa_key]:
                        metrics["correct_propagated"] += 1
                        metrics["correct_predictions"] += 1
        
        # Calculate accuracy metrics
        if metrics["total_predictions"] > 0:
            metrics["overall_accuracy"] = metrics["correct_predictions"] / metrics["total_predictions"]
        else:
            metrics["overall_accuracy"] = 0.0
            
        if metrics["human_labeled"] > 0:
            metrics["human_accuracy"] = metrics["correct_human"] / metrics["human_labeled"]
        else:
            metrics["human_accuracy"] = 0.0
            
        if metrics["propagated"] > 0:
            metrics["propagation_accuracy"] = metrics["correct_propagated"] / metrics["propagated"]
        else:
            metrics["propagation_accuracy"] = 0.0
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def calculate_confidence_statistics(self, qa_pairs: Dict[str, QAPair]) -> Dict[str, float]:
        """Calculate statistics about confidence scores"""
        confidences_by_source = {
            "HUMAN": [],
            "PROPAGATED": [],
            "OUTLIER": [],
            "UNPROPAGATED": []
        }
        
        for qa_pair in qa_pairs.values():
            if qa_pair.confidence is not None and qa_pair.source:
                confidences_by_source[qa_pair.source].append(qa_pair.confidence)
        
        stats = {}
        for source, confidences in confidences_by_source.items():
            if confidences:
                stats[f"{source.lower()}_mean_confidence"] = np.mean(confidences)
                stats[f"{source.lower()}_std_confidence"] = np.std(confidences)
                stats[f"{source.lower()}_count"] = len(confidences)
            else:
                stats[f"{source.lower()}_mean_confidence"] = 0.0
                stats[f"{source.lower()}_std_confidence"] = 0.0
                stats[f"{source.lower()}_count"] = 0
        
        return stats 