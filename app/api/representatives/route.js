// Proxy to Python FastAPI backend for diverse representative selection
const PYTHON_API_BASE = process.env.PYTHON_API_BASE || 'http://localhost:8000/api/v1';

export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const targetCount = searchParams.get('count') || '30';
    
    console.log(`Fetching ${targetCount} diverse representatives from Python backend...`);
    
    // Call Python FastAPI backend
    const response = await fetch(`${PYTHON_API_BASE}/representatives-preview/${targetCount}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Python API error: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error('Failed to select diverse representatives');
    }
    
    console.log(`Successfully selected ${data.selected_count} diverse representatives`);
    
    return Response.json({
      success: true,
      data: data
    });
    
  } catch (error) {
    console.error('Error in representatives API:', error);
    return Response.json(
      { 
        success: false, 
        error: error.message 
      },
      { status: 500 }
    );
  }
} 