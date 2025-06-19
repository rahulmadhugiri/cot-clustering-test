// No longer using JavaScript HDBSCAN - now calls Python FastAPI backend

// Updated to use Python FastAPI backend for HDBSCAN clustering
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://localhost:8000/api/v1';

export async function GET() {
  try {
    console.log('Starting HDBSCAN clustering via Python backend...');
    
    // Call Python FastAPI backend for clustering
    const response = await fetch(`${PYTHON_API_BASE}/cluster`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        min_cluster_size: 2,
        min_samples: 1
      })
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
      is_outlier: item.cluster_id === -1
    }));
    
    // Sort by cluster ID, then by outlier score (same as before)
    results.sort((a, b) => {
      if (a.cluster_id !== b.cluster_id) {
        return a.cluster_id - b.cluster_id;
      }
      return b.outlier_score - a.outlier_score;
    });
    
    console.log(`Returning ${results.length} clustered results from Python backend`);
    
    // Return in the same format your frontend expects
    return Response.json({
      success: true,
      data: results,
      summary: {
        total_vectors: results.length,
        clusters: clusteringResult.summary.num_clusters,
        outliers: clusteringResult.summary.outliers
      }
    });
    
  } catch (error) {
    console.error('Error calling Python HDBSCAN API:', error);
    
    // Fallback: if Python backend is not available, return a helpful error
    if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch')) {
      return Response.json({
        error: 'Python backend not available. Please start the Python FastAPI server with: cd backend && python main.py',
        details: error.message
      }, { status: 503 });
    }
    
    return Response.json({
      error: 'Failed to perform clustering',
      details: error.message
    }, { status: 500 });
  }
} 