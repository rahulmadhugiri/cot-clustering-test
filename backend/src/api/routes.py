from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
import numpy as np

from ..models.schemas import (
    CoTExample, QAPair, ClusteringRequest, ClusteringResponse,
    PropagationRequest, PropagationResponse, ExperimentResults
)
from ..services.clustering_service import ClusteringService
from ..services.propagation_service import PropagationService
from ..services.embedding_service import EmbeddingService
from ..utils.data_loader import DataLoader

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
def get_clustering_service():
    return ClusteringService()

def get_propagation_service():
    return PropagationService()

def get_embedding_service():
    return EmbeddingService()

def get_data_loader():
    return DataLoader()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "reasoning-clustering-backend"}

@router.get("/datasets", response_model=List[str])
async def list_datasets(data_loader: DataLoader = Depends(get_data_loader)):
    """List available datasets"""
    try:
        datasets = data_loader.list_available_datasets()
        return datasets
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/{dataset_name}", response_model=List[CoTExample])
async def load_dataset(
    dataset_name: str,
    data_loader: DataLoader = Depends(get_data_loader)
):
    """Load a specific dataset"""
    try:
        cot_examples = data_loader.load_dataset(dataset_name)
        return cot_examples
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings/generate")
async def generate_embeddings(
    cot_examples: List[CoTExample],
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Generate embeddings for CoT examples"""
    try:
        embedded_examples = await embedding_service.generate_embeddings(cot_examples)
        return {"success": True, "count": len(embedded_examples)}
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster", response_model=ClusteringResponse)
async def cluster_cots(
    request: ClusteringRequest,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    data_loader: DataLoader = Depends(get_data_loader)
):
    """Cluster CoT examples using HDBSCAN"""
    try:
        # First, try to fetch embeddings from vector database
        cot_examples = await embedding_service.fetch_embeddings()
        
        # If no embeddings found in Pinecone, load from local data and generate embeddings
        if not cot_examples:
            logger.info("No embeddings found in Pinecone, loading from local data...")
            
            # Load CoT data from local files
            cot_examples = data_loader.load_dataset('cots')
            
            if not cot_examples:
                raise HTTPException(status_code=404, detail="No CoT data found in local files.")
            
            # Generate embeddings for the loaded data
            logger.info(f"Generating embeddings for {len(cot_examples)} CoT examples...")
            cot_examples = await embedding_service.generate_embeddings(cot_examples)
        
        # Perform clustering
        clustered_examples, summary = clustering_service.cluster_cot_examples(
            cot_examples,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples
        )
        
        return ClusteringResponse(
            success=True,
            data=clustered_examples,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Error during clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster-kmeans", response_model=ClusteringResponse)
async def cluster_cots_kmeans(
    n_clusters: int = 20,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    data_loader: DataLoader = Depends(get_data_loader)
):
    """Cluster CoT examples using K-Means (recommended for CoT data)"""
    try:
        # Fetch embeddings
        cot_examples = await embedding_service.fetch_embeddings()
        
        if not cot_examples:
            logger.info("No embeddings found in Pinecone, loading from local data...")
            cot_examples = data_loader.load_dataset('cots')
            
            if not cot_examples:
                raise HTTPException(status_code=404, detail="No CoT data found in local files.")
            
            logger.info(f"Generating embeddings for {len(cot_examples)} CoT examples...")
            cot_examples = await embedding_service.generate_embeddings(cot_examples)
        
        # Perform K-Means clustering
        clustered_examples, summary = clustering_service.cluster_cot_examples_kmeans(
            cot_examples,
            n_clusters=n_clusters
        )
        
        return ClusteringResponse(
            success=True,
            data=clustered_examples,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Error during K-Means clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/qa-pairs", response_model=Dict[str, QAPair])
async def get_qa_pairs(
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Get Q&A pairs grouped from clustered CoT examples"""
    try:
        # Fetch clustered examples
        cot_examples = await embedding_service.fetch_embeddings()
        
        if not cot_examples or not any(cot.cluster_id is not None for cot in cot_examples):
            raise HTTPException(status_code=404, detail="No clustered data found. Run clustering first.")
        
        # Group by Q&A pairs
        qa_pairs = clustering_service.group_by_qa_pairs(cot_examples)
        
        return qa_pairs
    except Exception as e:
        logger.error(f"Error getting Q&A pairs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/representatives/{num_representatives}", response_model=List[str])
async def select_representatives(
    num_representatives: int = 2,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Select representative Q&A pairs for human labeling"""
    try:
        # Get Q&A pairs
        cot_examples = await embedding_service.fetch_embeddings()
        qa_pairs = clustering_service.group_by_qa_pairs(cot_examples)
        
        # Select representatives
        representatives = clustering_service.select_representatives(
            qa_pairs, num_representatives
        )
        
        return representatives
    except Exception as e:
        logger.error(f"Error selecting representatives: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/representatives-diverse/{target_count}", response_model=List[str])
async def select_diverse_representatives(
    target_count: int = 30,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Select ~30 diverse, high-quality Q&A pairs for human labeling based on semantic diversity and cluster coverage"""
    try:
        # First ensure we have K-means clustered data
        cot_examples = await embedding_service.fetch_embeddings()
        
        if not cot_examples:
            raise HTTPException(status_code=404, detail="No CoT data found. Generate embeddings first.")
        
        # Ensure clustering has been done with K-means
        clustered_examples, _ = clustering_service.cluster_cot_examples_kmeans(cot_examples, n_clusters=20)
        
        # Group into Q&A pairs
        qa_pairs = clustering_service.group_by_qa_pairs(clustered_examples)
        
        # Use the new sophisticated selection method
        representatives = clustering_service.select_diverse_representatives(
            qa_pairs, target_count
        )
        
        return representatives
    except Exception as e:
        logger.error(f"Error selecting diverse representatives: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/representatives-preview/{target_count}")
async def preview_diverse_representatives(
    target_count: int = 30,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Preview the diverse representative selection with detailed info about selected Q&A pairs"""
    try:
        # First ensure we have K-means clustered data
        cot_examples = await embedding_service.fetch_embeddings()
        
        if not cot_examples:
            raise HTTPException(status_code=404, detail="No CoT data found. Generate embeddings first.")
        
        # Ensure clustering has been done with K-means
        clustered_examples, clustering_summary = clustering_service.cluster_cot_examples_kmeans(cot_examples, n_clusters=20)
        
        # Group into Q&A pairs
        qa_pairs = clustering_service.group_by_qa_pairs(clustered_examples)
        
        # Get selected representatives
        selected_keys = clustering_service.select_diverse_representatives(
            qa_pairs, target_count
        )
        
        # Build detailed preview
        preview_data = []
        cluster_stats = {}
        
        for qa_key in selected_keys:
            qa_pair = qa_pairs[qa_key]
            best_cot = min(qa_pair.cots, key=lambda cot: cot.outlier_score or 1.0)
            
            # Track cluster distribution
            cluster_id = best_cot.cluster_id
            cluster_stats[cluster_id] = cluster_stats.get(cluster_id, 0) + 1
            
            preview_data.append({
                "qa_key": qa_key,
                "question": qa_pair.question[:100] + "..." if len(qa_pair.question) > 100 else qa_pair.question,
                "answer": qa_pair.answer[:150] + "..." if len(qa_pair.answer) > 150 else qa_pair.answer,
                "cot_preview": best_cot.cot[:200] + "..." if len(best_cot.cot) > 200 else best_cot.cot,
                "cluster_id": cluster_id,
                "outlier_score": round(best_cot.outlier_score or 1.0, 3),
                "trust_level": "High" if (best_cot.outlier_score or 1.0) < 0.3 else "Medium" if (best_cot.outlier_score or 1.0) < 0.6 else "Low",
                "cot_length": len(best_cot.cot),
                "reasoning_type": _classify_reasoning_type(best_cot.cot)
            })
        
        return {
            "success": True,
            "selected_count": len(selected_keys),
            "target_count": target_count,
            "clustering_info": {
                "method": clustering_summary.get("method", "K-Means"),
                "total_clusters": clustering_summary.get("num_clusters", 0),
                "silhouette_score": clustering_summary.get("silhouette_score", 0.0)
            },
            "cluster_coverage": {
                "total_clusters": clustering_summary.get("num_clusters", 0),
                "covered_clusters": len(cluster_stats),
                "cluster_distribution": dict(sorted(cluster_stats.items()))
            },
            "selected_pairs": preview_data,
            "quality_stats": {
                "avg_trust_score": round(sum(1.0 - item["outlier_score"] for item in preview_data) / len(preview_data), 3) if preview_data else 0.0,
                "avg_cot_length": round(sum(item["cot_length"] for item in preview_data) / len(preview_data), 1) if preview_data else 0.0,
                "reasoning_types": {rtype: sum(1 for item in preview_data if item["reasoning_type"] == rtype) 
                                   for rtype in set(item["reasoning_type"] for item in preview_data)} if preview_data else {}
            }
        }
    except Exception as e:
        logger.error(f"Error previewing diverse representatives: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _classify_reasoning_type(cot_text: str) -> str:
    """Classify the type of reasoning based on text analysis"""
    cot_lower = cot_text.lower()
    
    # Pattern matching for different reasoning types
    if any(word in cot_lower for word in ['step', 'first', 'second', 'then', 'next', 'finally']):
        return "Step-by-step"
    elif any(word in cot_lower for word in ['because', 'since', 'therefore', 'thus', 'hence']):
        return "Causal"
    elif any(word in cot_lower for word in ['compare', 'contrast', 'similar', 'different', 'versus']):
        return "Comparative"
    elif any(word in cot_lower for word in ['if', 'assume', 'suppose', 'given']):
        return "Conditional"
    elif any(word in cot_lower for word in ['analysis', 'examine', 'evaluate', 'consider']):
        return "Analytical"
    else:
        return "General"

@router.post("/propagate", response_model=PropagationResponse)
async def propagate_labels(
    request: PropagationRequest,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    propagation_service: PropagationService = Depends(get_propagation_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Propagate human labels to other Q&A pairs"""
    try:
        # Get Q&A pairs - ensure we have the same clustering as used for selection
        cot_examples = await embedding_service.fetch_embeddings()
        
        if not cot_examples:
            raise HTTPException(status_code=404, detail="No CoT data found.")
        
        # Re-cluster with K-means to ensure consistency
        clustered_examples, _ = clustering_service.cluster_cot_examples_kmeans(cot_examples, n_clusters=20)
        qa_pairs = clustering_service.group_by_qa_pairs(clustered_examples)
        
        # Use the SAME diverse selection method as the frontend used
        # This ensures representative keys match between selection and propagation
        representatives = clustering_service.select_diverse_representatives(
            qa_pairs, request.num_representatives or 30
        )
        
        logger.info(f"Using {len(representatives)} diverse representatives for propagation")
        logger.info(f"Human labels provided for {len(request.human_labels)} Q&A pairs")
        
        # Propagate labels
        updated_qa_pairs, summary = propagation_service.propagate_labels(
            qa_pairs, request.human_labels, representatives
        )
        
        # Calculate confidence statistics
        confidence_stats = propagation_service.calculate_confidence_statistics(updated_qa_pairs)
        summary.update(confidence_stats)
        
        return PropagationResponse(
            success=True,
            qa_pairs=list(updated_qa_pairs.values()),
            summary=summary
        )
    except Exception as e:
        logger.error(f"Error during label propagation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_results(
    ground_truth: Dict[str, str],
    clustering_service: ClusteringService = Depends(get_clustering_service),
    propagation_service: PropagationService = Depends(get_propagation_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Evaluate propagation results against ground truth"""
    try:
        # Get current Q&A pairs with predictions
        cot_examples = await embedding_service.fetch_embeddings()
        qa_pairs = clustering_service.group_by_qa_pairs(cot_examples)
        
        # Note: This assumes propagation has already been run
        # In a real implementation, you'd want to store the propagation results
        
        evaluation_metrics = propagation_service.evaluate_propagation(qa_pairs, ground_truth)
        
        return {
            "success": True,
            "evaluation": evaluation_metrics
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiment/results", response_model=ExperimentResults)
async def get_experiment_results(
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Get complete experiment results"""
    try:
        # Fetch all data
        cot_examples = await embedding_service.fetch_embeddings()
        qa_pairs = clustering_service.group_by_qa_pairs(cot_examples)
        
        # Generate summaries
        clustering_summary = {
            "total_cots": len(cot_examples),
            "total_qa_pairs": len(qa_pairs),
            "clusters": len(set(cot.cluster_id for cot in cot_examples if cot.cluster_id != -1)),
            "outliers": sum(1 for cot in cot_examples if cot.cluster_id == -1)
        }
        
        propagation_summary = {
            "human_labeled": sum(1 for qa in qa_pairs.values() if qa.source == "HUMAN"),
            "propagated": sum(1 for qa in qa_pairs.values() if qa.source == "PROPAGATED"),
            "unpropagated": sum(1 for qa in qa_pairs.values() if qa.source == "UNPROPAGATED")
        }
        
        return ExperimentResults(
            cot_examples=cot_examples,
            qa_pairs=qa_pairs,
            clustering_summary=clustering_summary,
            propagation_summary=propagation_summary,
            evaluation_metrics={}  # Would be populated after evaluation
        )
    except Exception as e:
        logger.error(f"Error getting experiment results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/similarity-matrix")
async def get_similarity_matrix(
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Get similarity matrix between Q&A pairs"""
    try:
        cot_examples = await embedding_service.fetch_embeddings()
        qa_pairs = clustering_service.group_by_qa_pairs(cot_examples)
        
        similarity_matrix = clustering_service.calculate_similarity_matrix(qa_pairs)
        
        return {
            "success": True,
            "similarity_matrix": similarity_matrix.to_dict(),
            "qa_pairs": list(qa_pairs.keys())
        }
    except Exception as e:
        logger.error(f"Error calculating similarity matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiment/parameters", response_model=Dict[str, Any])
async def experiment_clustering_parameters(
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Experiment with different HDBSCAN parameters to find optimal settings"""
    try:
        # Fetch embeddings
        cot_examples = await embedding_service.fetch_embeddings()
        
        if not cot_examples:
            raise HTTPException(status_code=404, detail="No CoT data found. Generate embeddings first.")
        
        # Run parameter experiments
        results = clustering_service.experiment_with_parameters(cot_examples)
        
        return {
            "success": True,
            "total_examples": len(cot_examples),
            "current_settings": {
                "min_cluster_size": 2,
                "min_samples": 1,
                "outlier_percentage": 45.0  # Your current result
            },
            "experiments": results,
            "recommendations": _generate_parameter_recommendations(results)
        }
    except Exception as e:
        logger.error(f"Error during parameter experimentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_parameter_recommendations(results: Dict[str, Dict]) -> List[str]:
    """Generate recommendations based on experimental results"""
    recommendations = []
    
    # Find settings with good outlier reduction
    low_outlier_results = [(k, v) for k, v in results.items() if v['outlier_percentage'] < 30]
    
    if low_outlier_results:
        best_outlier = min(low_outlier_results, key=lambda x: x[1]['outlier_percentage'])
        recommendations.append(
            f"For lowest outlier rate ({best_outlier[1]['outlier_percentage']}%): "
            f"min_cluster_size={best_outlier[1]['parameters']['min_cluster_size']}, "
            f"min_samples={best_outlier[1]['parameters']['min_samples']}"
        )
    
    # Find settings with good cluster count
    good_cluster_results = [(k, v) for k, v in results.items() if 10 <= v['num_clusters'] <= 25]
    
    if good_cluster_results:
        best_clusters = max(good_cluster_results, key=lambda x: x[1]['num_clusters'])
        recommendations.append(
            f"For good cluster count ({best_clusters[1]['num_clusters']} clusters): "
            f"min_cluster_size={best_clusters[1]['parameters']['min_cluster_size']}, "
            f"min_samples={best_clusters[1]['parameters']['min_samples']}"
        )
    
    # Balanced approach
    balanced_results = [(k, v) for k, v in results.items() 
                       if v['outlier_percentage'] < 35 and v['num_clusters'] >= 8]
    
    if balanced_results:
        balanced = min(balanced_results, key=lambda x: abs(x[1]['outlier_percentage'] - 25))
        recommendations.append(
            f"For balanced approach ({balanced[1]['num_clusters']} clusters, {balanced[1]['outlier_percentage']}% outliers): "
            f"min_cluster_size={balanced[1]['parameters']['min_cluster_size']}, "
            f"min_samples={balanced[1]['parameters']['min_samples']}"
        )
    
    if not recommendations:
        recommendations.append("Try min_cluster_size=5, min_samples=3 as a starting point for better results.")
    
    return recommendations

@router.get("/experiment/methods", response_model=Dict[str, Any])
async def experiment_clustering_methods(
    clustering_service: ClusteringService = Depends(get_clustering_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Experiment with different clustering methods (K-means, Agglomerative, HDBSCAN variants)"""
    try:
        # Fetch embeddings
        cot_examples = await embedding_service.fetch_embeddings()
        
        if not cot_examples:
            raise HTTPException(status_code=404, detail="No CoT data found. Generate embeddings first.")
        
        # Run method experiments
        results = clustering_service.experiment_alternative_methods(cot_examples)
        
        return {
            "success": True,
            "total_examples": len(cot_examples),
            "current_hdbscan": {
                "method": "HDBSCAN",
                "outliers": 135,
                "outlier_percentage": 45.0,
                "num_clusters": 48
            },
            "alternative_methods": results,
            "analysis": _analyze_clustering_methods(results)
        }
    except Exception as e:
        logger.error(f"Error during method experimentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _analyze_clustering_methods(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Analyze results from different clustering methods"""
    analysis = {
        "best_silhouette": {"score": -1, "method": None, "config": None},
        "most_balanced_clusters": {"variance": float('inf'), "method": None, "config": None},
        "methods_summary": {}
    }
    
    for method_key, result in results.items():
        method = result['method']
        
        # Track best silhouette score
        if 'silhouette_score' in result and result['silhouette_score'] > analysis["best_silhouette"]["score"]:
            analysis["best_silhouette"] = {
                "score": result['silhouette_score'],
                "method": method,
                "config": method_key
            }
        
        # Track most balanced clusters (lowest variance in cluster sizes)
        if result['cluster_sizes'] and len(result['cluster_sizes']) > 1:
            variance = np.var(result['cluster_sizes'])
            if variance < analysis["most_balanced_clusters"]["variance"]:
                analysis["most_balanced_clusters"] = {
                    "variance": round(variance, 2),
                    "method": method,
                    "config": method_key,
                    "avg_size": result['avg_cluster_size']
                }
        
        # Summarize by method
        if method not in analysis["methods_summary"]:
            analysis["methods_summary"][method] = {
                "count": 0,
                "avg_clusters": 0,
                "avg_outlier_rate": 0,
                "configs": []
            }
        
        summary = analysis["methods_summary"][method]
        summary["count"] += 1
        summary["avg_clusters"] += result['num_clusters']
        summary["avg_outlier_rate"] += result['outlier_percentage']
        summary["configs"].append({
            "config": method_key,
            "clusters": result['num_clusters'],
            "outlier_rate": result['outlier_percentage']
        })
    
    # Calculate averages
    for method, summary in analysis["methods_summary"].items():
        if summary["count"] > 0:
            summary["avg_clusters"] = round(summary["avg_clusters"] / summary["count"], 1)
            summary["avg_outlier_rate"] = round(summary["avg_outlier_rate"] / summary["count"], 1)
    
    return analysis 