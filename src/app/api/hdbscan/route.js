// Updated to use K-Means clustering via Python FastAPI backend (better than HDBSCAN for CoT data)
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://localhost:8000/api/v1';

export async function GET() {
  try {
    console.log('Starting K-Means clustering via Python backend (recommended for CoT data)...');
    
    // Call Python FastAPI backend for K-Means clustering (much better than HDBSCAN)
    const response = await fetch(`${PYTHON_API_BASE}/cluster-kmeans?n_clusters=20`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Python API error: ${response.status}`);
    }
    
    const clusteringResult = await response.json();
    
    if (!clusteringResult.success) {
      throw new Error('Clustering failed in Python backend');
    }
    
    // Transform Python backend response to match your existing frontend format
    const results = clusteringResult.data.map(item => ({
      id: item.id,
      question: item.question,
      answer: item.answer,
      cot: item.cot,
      cluster_id: item.cluster_id,
      cluster_size: clusteringResult.summary.cluster_sizes[item.cluster_id] || 0,
      outlier_score: item.outlier_score,
      is_outlier: false  // K-means doesn't produce outliers
    }));
    
    // Sort by cluster ID, then by outlier score (distance to centroid)
    results.sort((a, b) => {
      if (a.cluster_id !== b.cluster_id) {
        return a.cluster_id - b.cluster_id;
      }
      return a.outlier_score - b.outlier_score;  // Lower score = closer to centroid
    });
    
    console.log(`Returning ${results.length} K-Means clustered results from Python backend`);
    console.log(`K-Means results: ${clusteringResult.summary.num_clusters} clusters, 0 outliers, silhouette score: ${clusteringResult.summary.silhouette_score?.toFixed(3) || 'N/A'}`);
    
    // Return in the same format your frontend expects
    return Response.json({
      success: true,
      data: results,
      summary: {
        total_vectors: results.length,
        clusters: clusteringResult.summary.num_clusters,
        outliers: clusteringResult.summary.outliers,
        method: clusteringResult.summary.method,
        silhouette_score: clusteringResult.summary.silhouette_score,
        avg_cluster_size: clusteringResult.summary.avg_cluster_size
      }
    });
    
  } catch (error) {
    console.error('Error calling Python K-Means API:', error);
    
    // Fallback: if Python backend is not available, return a helpful error
    if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch')) {
      return Response.json({
        error: 'Python backend not available. Please start the Python FastAPI server with: cd backend && python main.py',
        details: error.message
      }, { status: 503 });
    }
    
    return Response.json({
      error: 'Failed to perform K-Means clustering',
      details: error.message
    }, { status: 500 });
  }
} 