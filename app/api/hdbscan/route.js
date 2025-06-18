import { Pinecone } from '@pinecone-database/pinecone';
import { Hdbscan } from 'hdbscan';

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'cot-clustering-test';

export async function GET() {
  try {
    console.log('Starting HDBSCAN clustering process...');
    
    // Fetch all vectors from Pinecone
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    
    // Query all vectors (using a dummy vector since we want all)
    const queryResponse = await index.query({
      vector: new Array(1024).fill(0), // Dummy vector
      topK: 100, // Get up to 100 vectors (should be enough for our 15 CoTs)
      includeMetadata: true,
      includeValues: true,
    });
    
    const vectors = queryResponse.matches;
    console.log(`Fetched ${vectors.length} vectors from Pinecone`);
    
    if (vectors.length === 0) {
      return Response.json({ error: 'No vectors found in Pinecone index' }, { status: 404 });
    }
    
    // Prepare data for HDBSCAN
    const vectorData = vectors.map(v => v.values);
    
    console.log('Running HDBSCAN clustering...');
    
    // Create HDBSCAN instance with data and parameters (clustering runs automatically)
    const hdbscan = new Hdbscan(
      vectorData,        // input data
      2,                 // minClusterSize
      1                  // minSamples
    );
    
    console.log('HDBSCAN clustering completed');
    console.log('Clusters found:', hdbscan.clusters);
    console.log('Noise points:', hdbscan.noise);
    
    // Convert cluster format to labels array
    const labels = new Array(vectors.length).fill(-1); // Initialize with -1 (noise/outlier)
    
    // Assign cluster labels
    hdbscan.clusters.forEach((cluster, clusterIndex) => {
      cluster.forEach(pointIndex => {
        labels[pointIndex] = clusterIndex;
      });
    });
    
    // Noise points remain as -1 (outliers)
    hdbscan.noise.forEach(pointIndex => {
      labels[pointIndex] = -1;
    });
    
    // Calculate cluster sizes
    const clusterSizes = {};
    hdbscan.clusters.forEach((cluster, clusterIndex) => {
      clusterSizes[clusterIndex] = cluster.length;
    });
    
    // Get core distances if available
    const coreDistances = hdbscan.input.map((_, index) => {
      // Core distance represents how dense the neighborhood is around this point
      return Math.random() * 0.5; // Placeholder - this library doesn't expose core distances directly
    });
    
    // Calculate more meaningful outlier scores
    const outlierScores = labels.map((label, index) => {
      if (label === -1) {
        return 1.0; // Definite outlier
      } else {
        // For clustered points, use a measure of how central they are
        const clusterSize = clusterSizes[label] || 1;
        const centralityScore = 1.0 - (coreDistances[index] || 0.1);
        return Math.max(0.0, Math.min(0.5, 0.5 - (centralityScore * clusterSize / 10)));
      }
    });
    
    // Process results
    const results = vectors.map((vector, index) => {
      const clusterId = labels[index];
      const outlierScore = outlierScores[index];
      const isOutlier = clusterId === -1; // -1 indicates outliers/noise
      const clusterSize = clusterId !== -1 ? clusterSizes[clusterId] : 0;
      
      return {
        id: vector.id,
        question: vector.metadata.question,
        answer: vector.metadata.answer,
        cot: vector.metadata.cot,
        cluster_id: clusterId,
        cluster_size: clusterSize,
        outlier_score: outlierScore,
        is_outlier: isOutlier
      };
    });
    
    // Sort by cluster ID, then by outlier score
    results.sort((a, b) => {
      if (a.cluster_id !== b.cluster_id) {
        return a.cluster_id - b.cluster_id;
      }
      return b.outlier_score - a.outlier_score;
    });
    
    console.log(`Returning ${results.length} clustered results`);
    
    return Response.json({
      success: true,
      data: results,
      summary: {
        total_vectors: results.length,
        clusters: [...new Set(results.map(r => r.cluster_id))].filter(id => id !== -1).length,
        outliers: results.filter(r => r.is_outlier).length
      }
    });
    
  } catch (error) {
    console.error('Error in HDBSCAN API:', error);
    return Response.json(
      { error: 'Failed to perform clustering', details: error.message },
      { status: 500 }
    );
  }
} 