from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

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

@router.post("/propagate", response_model=PropagationResponse)
async def propagate_labels(
    request: PropagationRequest,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    propagation_service: PropagationService = Depends(get_propagation_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """Propagate human labels to other Q&A pairs"""
    try:
        # Get Q&A pairs and representatives
        cot_examples = await embedding_service.fetch_embeddings()
        qa_pairs = clustering_service.group_by_qa_pairs(cot_examples)
        representatives = clustering_service.select_representatives(
            qa_pairs, request.num_representatives
        )
        
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