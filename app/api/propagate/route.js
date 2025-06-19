// Label propagation API route - calls Python FastAPI backend
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://localhost:8000/api/v1';

export async function POST(req) {
  try {
    const { human_labels, num_representatives = 2 } = await req.json();
    
    console.log('Calling Python backend for label propagation...');
    
    // Call Python FastAPI backend for propagation
    const response = await fetch(`${PYTHON_API_BASE}/propagate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        human_labels,
        num_representatives
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Python API error: ${response.status}`);
    }
    
    const propagationResult = await response.json();
    
    if (!propagationResult.success) {
      throw new Error('Label propagation failed in Python backend');
    }
    
    // Transform Python backend response to match your existing frontend format
    const results = propagationResult.qa_pairs.map(qaPair => ({
      qa_pair_key: `${qaPair.question}|||${qaPair.answer}`,
      question: qaPair.question,
      answer: qaPair.answer,
      predicted_label: qaPair.predicted_label,
      confidence: qaPair.confidence,
      source: qaPair.source,
      propagation_source: qaPair.propagation_source,
      shared_reasoning: qaPair.shared_reasoning,
      reasoning_cots: qaPair.cots,
      cluster_info: qaPair.clusters.map(id => `Cluster ${id}`).join(', ')
    }));
    
    console.log(`Returning ${results.length} propagation results from Python backend`);
    
    return Response.json({
      success: true,
      results: results,
      summary: propagationResult.summary
    });
    
  } catch (error) {
    console.error('Error calling Python propagation API:', error);
    
    // Fallback: if Python backend is not available, return a helpful error
    if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch')) {
      return Response.json({
        error: 'Python backend not available. Please start the Python FastAPI server with: cd backend && python main.py',
        details: error.message
      }, { status: 503 });
    }
    
    return Response.json({
      error: 'Failed to perform label propagation',
      details: error.message
    }, { status: 500 });
  }
} 