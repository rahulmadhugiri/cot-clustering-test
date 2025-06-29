import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'cot-clustering-test';

// Helper function to calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// Helper function to calculate betweenness centrality (simplified)
function calculateBetweennessCentrality(adjacencyMatrix) {
  const n = adjacencyMatrix.length;
  const centrality = new Array(n).fill(0);
  
  // Simplified betweenness centrality calculation
  for (let s = 0; s < n; s++) {
    for (let t = 0; t < n; t++) {
      if (s !== t) {
        // Find shortest paths from s to t
        const distances = new Array(n).fill(Infinity);
        const previous = new Array(n).fill(-1);
        const visited = new Array(n).fill(false);
        distances[s] = 0;
        
        for (let i = 0; i < n; i++) {
          let u = -1;
          for (let v = 0; v < n; v++) {
            if (!visited[v] && (u === -1 || distances[v] < distances[u])) {
              u = v;
            }
          }
          
          if (distances[u] === Infinity) break;
          visited[u] = true;
          
          for (let v = 0; v < n; v++) {
            if (adjacencyMatrix[u][v] > 0) {
              const alt = distances[u] + (1 - adjacencyMatrix[u][v]);
              if (alt < distances[v]) {
                distances[v] = alt;
                previous[v] = u;
              }
            }
          }
        }
        
        // Count paths through each node
        let current = t;
        while (previous[current] !== -1) {
          if (current !== s && current !== t) {
            centrality[current] += 1;
          }
          current = previous[current];
        }
      }
    }
  }
  
  return centrality;
}

export async function POST(request) {
  try {
    const { qaDataWithCoTs } = await request.json();
    
    if (!qaDataWithCoTs || !Array.isArray(qaDataWithCoTs)) {
      return Response.json({ success: false, error: 'Invalid Q&A data with CoTs provided' });
    }

    // Step 1: Fetch all CoT vectors from Pinecone
    console.log('Fetching CoT embeddings from Pinecone...');
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    
    // First, let's check what vectors exist in the index
    try {
      const stats = await index.describeIndexStats();
      console.log('Pinecone index stats:', stats);
    } catch (statsError) {
      console.log('Could not get index stats:', statsError.message);
    }
    
    // Get all vectors from Pinecone
    const allVectorIds = qaDataWithCoTs.flatMap(qa => [`cot_${qa.id}_pos`, `cot_${qa.id}_neg`]);
    console.log('Looking for vector IDs:', allVectorIds);
    console.log('Total expected vectors:', allVectorIds.length);
    
    const fetchResponse = await index.fetch(allVectorIds);
    console.log('Pinecone fetch response:', fetchResponse);
    console.log('Pinecone fetch response type:', typeof fetchResponse);
    
    // Check if fetchResponse and vectors exist
    if (!fetchResponse) {
      return Response.json({ 
        success: false, 
        error: 'No response from Pinecone fetch operation',
        debug: { expectedVectorIds: allVectorIds }
      });
    }
    
    // Check if we have records (Pinecone's actual response structure)
    if (!fetchResponse.records) {
      return Response.json({ 
        success: false, 
        error: 'No records property in Pinecone response',
        debug: { 
          expectedVectorIds: allVectorIds,
          responseKeys: Object.keys(fetchResponse),
          response: fetchResponse
        }
      });
    }
    
    console.log('Pinecone fetch response keys:', Object.keys(fetchResponse.records));
    
    const vectors = [];
    const vectorMetadata = [];
    
    qaDataWithCoTs.forEach(qa => {
      const posVectorId = `cot_${qa.id}_pos`;
      const negVectorId = `cot_${qa.id}_neg`;
      const posVector = fetchResponse.records[posVectorId];
      const negVector = fetchResponse.records[negVectorId];
      
      console.log(`Q&A ${qa.id}: pos vector exists=${!!posVector}, neg vector exists=${!!negVector}`);
      
      if (posVector && negVector) {
        vectors.push(posVector.values, negVector.values);
        vectorMetadata.push(posVector.metadata, negVector.metadata);
      } else {
        console.warn(`Missing vectors for Q&A ${qa.id}: pos=${posVectorId}, neg=${negVectorId}`);
      }
    });

    // Check if we have any vectors to work with
    if (vectors.length === 0) {
      const foundVectorIds = fetchResponse.records ? Object.keys(fetchResponse.records) : [];
      return Response.json({ 
        success: false, 
        error: 'No CoT vectors found in Pinecone. Please generate CoTs first (Phase 1).',
        debug: {
          expectedVectorIds: allVectorIds,
          foundVectorIds: foundVectorIds,
          qaDataCount: qaDataWithCoTs.length,
          totalFoundVectors: foundVectorIds.length,
          sampleFoundIds: foundVectorIds.slice(0, 5)
        }
      });
    }

    console.log(`Successfully loaded ${vectors.length} vectors from Pinecone`);

    // Step 2: Build similarity graph using Pinecone similarity search
    console.log('Building similarity graph using Pinecone...');
    const n = vectors.length;
    const adjacencyMatrix = Array(n).fill().map(() => Array(n).fill(0));
    const similarities = [];
    
    // Use Pinecone similarity search for each vector to find similar ones
    for (let i = 0; i < n; i++) {
      const queryResponse = await index.query({
        vector: vectors[i],
        topK: Math.min(35, n), // Increased from 20 to 35 to find more potential matches
        includeMetadata: false,
        includeValues: false
      });
      
      queryResponse.matches.forEach(match => {
        // Find the index of this match in our vectors array
        const matchId = match.id;
        const matchIndex = allVectorIds.indexOf(matchId);
        
        if (matchIndex !== -1 && matchIndex !== i) {
          // Check if this is a cross-QA connection (different Q&A pairs)
          const iQAIndex = Math.floor(i / 2);
          const jQAIndex = Math.floor(matchIndex / 2);
          const isCrossQA = iQAIndex !== jQAIndex;
          
          // Use different thresholds for intra-QA vs cross-QA connections
          const threshold = isCrossQA ? 0.55 : 0.65; // Lower threshold for cross-QA connections
          
          if (match.score > threshold) {
            const j = matchIndex;
            let adjustedScore = match.score;
            
            // Give a small boost to cross-QA connections to encourage diversity
            if (isCrossQA) {
              adjustedScore = Math.min(1.0, match.score * 1.05); // 5% boost, capped at 1.0
            }
            
            adjacencyMatrix[i][j] = adjustedScore;
            adjacencyMatrix[j][i] = adjustedScore;
            similarities.push({ 
              i, 
              j, 
              similarity: adjustedScore,
              isCrossQA: isCrossQA,
              originalScore: match.score
            });
          }
        }
      });
      
      // Small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    // Log connection statistics
    const crossQAConnections = similarities.filter(s => s.isCrossQA);
    const intraQAConnections = similarities.filter(s => !s.isCrossQA);
    console.log(`Graph connectivity stats:
      - Total connections: ${similarities.length}
      - Cross-QA connections: ${crossQAConnections.length} (${(crossQAConnections.length/similarities.length*100).toFixed(1)}%)
      - Intra-QA connections: ${intraQAConnections.length} (${(intraQAConnections.length/similarities.length*100).toFixed(1)}%)
      - Average cross-QA similarity: ${crossQAConnections.length > 0 ? (crossQAConnections.reduce((sum, s) => sum + s.originalScore, 0) / crossQAConnections.length).toFixed(3) : 'N/A'}
      - Average intra-QA similarity: ${intraQAConnections.length > 0 ? (intraQAConnections.reduce((sum, s) => sum + s.originalScore, 0) / intraQAConnections.length).toFixed(3) : 'N/A'}`);
    

    // Step 3: Calculate counterfactual divergence scores
    console.log('Calculating divergence scores...');
    const divergenceScores = qaDataWithCoTs.map((qa, index) => {
      const posEmbedding = vectors[index * 2];
      const negEmbedding = vectors[index * 2 + 1];
      const divergence = 1 - cosineSimilarity(posEmbedding, negEmbedding);
      return { qaIndex: index, divergence, qa };
    });

    // Step 4: Calculate centrality scores
    console.log('Calculating centrality scores...');
    const centralityScores = calculateBetweennessCentrality(adjacencyMatrix);
    
    // Aggregate centrality scores by Q&A pair
    const qaCentralityScores = qaDataWithCoTs.map((qa, index) => {
      const posCentrality = centralityScores[index * 2];
      const negCentrality = centralityScores[index * 2 + 1];
      const avgCentrality = (posCentrality + negCentrality) / 2;
      return { qaIndex: index, centrality: avgCentrality, qa };
    });

    // Step 5: Combine scores and select top Q&As
    console.log('Selecting optimal Q&As for labeling...');
    const combinedScores = qaDataWithCoTs.map((qa, index) => {
      const divergence = divergenceScores[index].divergence;
      const centrality = qaCentralityScores[index].centrality;
      
      // Normalize scores (simple min-max normalization)
      const normalizedDivergence = divergence;
      const maxCentrality = Math.max(...qaCentralityScores.map(q => q.centrality));
      const normalizedCentrality = maxCentrality > 0 ? centrality / maxCentrality : 0;
      
      // Combined score (weighted combination)
      const combinedScore = 0.6 * normalizedDivergence + 0.4 * normalizedCentrality;
      
      return {
        qaIndex: index,
        qa,
        divergence,
        centrality,
        combinedScore,
        scores: {
          divergence: normalizedDivergence,
          centrality: normalizedCentrality,
          combined: combinedScore
        }
      };
    });

    // Sort by combined score and select top 5-7
    combinedScores.sort((a, b) => b.combinedScore - a.combinedScore);
    const selectedCount = Math.min(7, Math.max(5, Math.floor(qaDataWithCoTs.length * 0.2)));
    const selectedQAs = combinedScores.slice(0, selectedCount).map(item => item.qa);

    console.log(`Selected ${selectedQAs.length} Q&As for labeling`);

    // Prepare graph data for visualization
    const graphData = {
      nodes: vectors.map((vector, index) => {
        const metadata = vectorMetadata[index];
        const qaIndex = Math.floor(index / 2);
        const isPos = index % 2 === 0;
        
        return {
          id: allVectorIds[index],
          qaIndex,
          type: isPos ? 'pos' : 'neg',
          centrality: centralityScores[index],
          divergence: divergenceScores[qaIndex].divergence,
          combinedScore: combinedScores[qaIndex].combinedScore,
          isSelected: selectedQAs.some(selected => selected.id === qaDataWithCoTs[qaIndex].id),
          metadata: metadata || {
            qa_id: qaIndex,
            question: qaDataWithCoTs[qaIndex].q,
            answer: qaDataWithCoTs[qaIndex].a,
            cot_text: isPos ? qaDataWithCoTs[qaIndex].cot_pos : qaDataWithCoTs[qaIndex].cot_neg
          }
        };
      }),
      edges: similarities.map(sim => ({
        source: allVectorIds[sim.i],
        target: allVectorIds[sim.j],
        weight: sim.similarity,
        type: 'similarity'
      })),
      statistics: {
        totalNodes: vectors.length,
        totalEdges: similarities.length,
        selectedQAs: selectedQAs.length,
        avgDivergence: divergenceScores.reduce((sum, s) => sum + s.divergence, 0) / divergenceScores.length,
        avgCentrality: qaCentralityScores.reduce((sum, s) => sum + s.centrality, 0) / qaCentralityScores.length,
        maxCentrality: Math.max(...centralityScores),
        minCentrality: Math.min(...centralityScores)
      }
    };

    return Response.json({ 
      success: true, 
      selectedQAs,
      graphData,
      analysis: {
        totalQAs: qaDataWithCoTs.length,
        selectedCount: selectedQAs.length,
        avgDivergence: divergenceScores.reduce((sum, s) => sum + s.divergence, 0) / divergenceScores.length,
        avgCentrality: qaCentralityScores.reduce((sum, s) => sum + s.centrality, 0) / qaCentralityScores.length,
        selectionCriteria: 'Top Q&As by combined divergence and centrality scores'
      }
    });

  } catch (error) {
    console.error('Error in construct-cot-graph:', error);
    return Response.json({ 
      success: false, 
      error: error.message || 'Failed to construct graph and select CoTs'
    });
  }
} 