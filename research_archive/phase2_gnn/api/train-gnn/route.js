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

// Improved Graph Neural Network implementation
class SimpleGNN {
  constructor(nodeFeatures, adjacencyMatrix, labels) {
    this.nodeFeatures = nodeFeatures;
    this.adjacencyMatrix = adjacencyMatrix;
    this.labels = labels;
    
    // Initialize predictions with slight random variation for unlabeled nodes
    this.predictions = new Array(nodeFeatures.length);
    for (let i = 0; i < nodeFeatures.length; i++) {
      if (labels[i] !== null) {
        this.predictions[i] = labels[i]; // Set labeled nodes to their true values
      } else {
        // Add small random variation around 0.5 for unlabeled nodes
        this.predictions[i] = 0.5 + (Math.random() - 0.5) * 0.1; // Range: 0.45 to 0.55
      }
    }
    
    this.learningRate = 0.1; // optimal_tuned
    this.iterations = 200; // optimal_tuned
    this.convergenceThreshold = 0.001; // baseline
  }

  // Improved message passing function
  messagePass() {
    const newPredictions = [...this.predictions];
    const n = this.nodeFeatures.length;

    for (let i = 0; i < n; i++) {
      if (this.labels[i] !== null) {
        // Keep labeled nodes fixed at their true labels
        newPredictions[i] = this.labels[i];
        continue;
      }

      // Aggregate messages from neighbors
      let weightedSum = 0;
      let totalWeight = 0;
      let neighborCount = 0;

      for (let j = 0; j < n; j++) {
        if (i !== j && this.adjacencyMatrix[i][j] > 0) {
          const weight = this.adjacencyMatrix[i][j];
          weightedSum += weight * this.predictions[j];
          totalWeight += weight;
          neighborCount++;
        }
      }

      // Add self-loop with balanced weight for rich CoT features
      const selfWeight = 0.99; // baseline - adaptive
      weightedSum += selfWeight * this.predictions[i];
      totalWeight += selfWeight;

      // Update prediction with weighted average
      if (totalWeight > 0) {
        const newValue = weightedSum / totalWeight;
        // Apply learning rate for smoother convergence
        newPredictions[i] = this.predictions[i] + this.learningRate * (newValue - this.predictions[i]);
      } else {
        // If no neighbors, stay at current prediction (isolated node)
        console.warn(`Node ${i} has no neighbors in adjacency matrix`);
      }
    }

    return newPredictions;
  }

  // Train the GNN using iterative message passing with improved convergence check
  train() {
    console.log('Starting GNN training with message passing...');
    console.log(`Initial labeled nodes: ${this.labels.filter(l => l !== null).length}`);
    console.log(`Initial predictions: [${this.predictions.slice(0, 10).map(p => p.toFixed(3)).join(', ')}...]`);
    
    let stableIterations = 0;
    const requiredStableIterations = 8; // baseline
    
    for (let iter = 0; iter < this.iterations; iter++) {
      const oldPredictions = [...this.predictions];
      const newPredictions = this.messagePass();
      
      // Apply sigmoid to keep predictions in [0,1] for unlabeled nodes only
      for (let i = 0; i < newPredictions.length; i++) {
        if (this.labels[i] === null) {
          // Apply sigmoid activation
          newPredictions[i] = 1 / (1 + Math.exp(-4.871414807100163 * (newPredictions[i] - 0.5))); // Steeper sigmoid
        }
      }

      // Check for convergence
      let maxChange = 0;
      let totalChange = 0;
      let unlabeledChanges = 0;
      
      for (let i = 0; i < newPredictions.length; i++) {
        if (this.labels[i] === null) {
          const change = Math.abs(newPredictions[i] - oldPredictions[i]);
          maxChange = Math.max(maxChange, change);
          totalChange += change;
          unlabeledChanges++;
        }
      }

      this.predictions = newPredictions;

      // Log progress more frequently with detailed info
      if (iter % 5 === 0 || iter < 5) {
        const avgChange = unlabeledChanges > 0 ? totalChange / unlabeledChanges : 0;
        console.log(`Iteration ${iter}: max change = ${maxChange.toFixed(6)}, avg change = ${avgChange.toFixed(6)}, unlabeled nodes = ${unlabeledChanges}, stable: ${stableIterations}/${requiredStableIterations}`);
        
        // Show some sample predictions
        if (iter % 10 === 0) {
          const samplePreds = this.predictions.slice(0, 8).map((p, i) => 
            `${i}:${this.labels[i] !== null ? 'L' : 'U'}=${p.toFixed(3)}`
          ).join(' ');
          console.log(`Sample predictions: ${samplePreds}`);
        }
      }

      // Improved convergence check - require multiple stable iterations
      if (maxChange < this.convergenceThreshold) {
        stableIterations++;
        if (stableIterations >= requiredStableIterations && iter >= 20) {
          console.log(`Converged after ${iter + 1} iterations (${stableIterations} stable iterations)`);
          break;
        }
      } else {
        stableIterations = 0; // Reset if not stable
      }
      
      // Force at least 20 iterations for proper propagation with rich CoTs
      if (iter < 20) {
        stableIterations = 0; // Don't allow early convergence
        continue;
      }
    }

    console.log('GNN training completed');
    console.log(`Final predictions: [${this.predictions.slice(0, 10).map(p => p.toFixed(3)).join(', ')}...]`);
    return this.predictions;
  }
}

export async function POST(request) {
  try {
    const { qaDataWithCoTs, labeledData } = await request.json();
    
    if (!qaDataWithCoTs || !Array.isArray(qaDataWithCoTs)) {
      return Response.json({ success: false, error: 'Invalid Q&A data with CoTs provided' });
    }

    const isUnsupervised = !labeledData || Object.keys(labeledData).length === 0;
    
    if (isUnsupervised) {
      console.log('Starting GNN training in UNSUPERVISED mode - no labels provided');
    } else {
      console.log('Starting GNN training with', Object.keys(labeledData).length, 'labeled examples');
    }

    // Note: Labels should already be updated in Pinecone before calling this endpoint

    // Step 1: Fetch all CoT vectors from Pinecone
    console.log('Fetching CoT embeddings from Pinecone for GNN training...');
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    
    const allVectorIds = qaDataWithCoTs.flatMap(qa => [`cot_${qa.id}_pos`, `cot_${qa.id}_neg`]);
    const fetchResponse = await index.fetch(allVectorIds, { includeMetadata: true });
    
    const embeddings = [];
    const vectorMetadata = [];
    
    qaDataWithCoTs.forEach(qa => {
      const posVector = fetchResponse.records[`cot_${qa.id}_pos`];
      const negVector = fetchResponse.records[`cot_${qa.id}_neg`];
      
      if (posVector && negVector) {
        embeddings.push(posVector.values, negVector.values);
        vectorMetadata.push(posVector.metadata, negVector.metadata);
        
        // Debug: Log if these vectors have labels
        if (posVector.metadata && posVector.metadata.is_labeled) {
          console.log(`Found labeled vector: ${posVector.id} - ${posVector.metadata.label}`);
        }
        if (negVector.metadata && negVector.metadata.is_labeled) {
          console.log(`Found labeled vector: ${negVector.id} - ${negVector.metadata.label}`);
        }
      }
    });

    // Step 2: Build adjacency matrix using improved similarity thresholds
    const n = embeddings.length;
    const adjacencyMatrix = Array(n).fill().map(() => Array(n).fill(0));
    
    console.log(`Building adjacency matrix for ${n} nodes...`);
    
    // Use Pinecone similarity search to build adjacency matrix efficiently
    for (let i = 0; i < n; i++) {
      const queryResponse = await index.query({
        vector: embeddings[i],
        topK: Math.min(35, n), // Increased from 20 to match our graph construction
        includeMetadata: false,
        includeValues: false
      });
      
      queryResponse.matches.forEach(match => {
        const matchIndex = allVectorIds.indexOf(match.id);
        if (matchIndex !== -1 && matchIndex !== i) {
          // Use same improved thresholds as in graph construction
          const iQAIndex = Math.floor(i / 2);
          const jQAIndex = Math.floor(matchIndex / 2);
          const isCrossQA = iQAIndex !== jQAIndex;
          const threshold = isCrossQA ? 0.17078430977876277 : 0.9273746864921559;
          
          if (match.score > threshold) {
            let adjustedScore = match.score;
            if (isCrossQA) {
              adjustedScore = Math.min(1.0, match.score * 1.05);
            }
            adjacencyMatrix[i][matchIndex] = adjustedScore;
            adjacencyMatrix[matchIndex][i] = adjustedScore;
          }
        }
      });
      
      // Small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    // Log adjacency matrix statistics
    const totalEdges = adjacencyMatrix.flat().filter(w => w > 0).length / 2;
    console.log(`Adjacency matrix built with ${totalEdges} edges`);
    
    // Check for isolated nodes
    const isolatedNodes = [];
    for (let i = 0; i < n; i++) {
      const hasEdges = adjacencyMatrix[i].some(w => w > 0);
      if (!hasEdges) {
        isolatedNodes.push(i);
      }
    }
    
    if (isolatedNodes.length > 0) {
      console.warn(`Found ${isolatedNodes.length} isolated nodes: ${isolatedNodes.slice(0, 5).join(', ')}${isolatedNodes.length > 5 ? '...' : ''}`);
    }
    
    // Log connectivity stats
    const avgDegree = (totalEdges * 2) / n;
    console.log(`Average node degree: ${avgDegree.toFixed(2)}`);
    
    // Show sample adjacency matrix entries
    const sampleEdges = [];
    for (let i = 0; i < Math.min(5, n); i++) {
      for (let j = i + 1; j < Math.min(5, n); j++) {
        if (adjacencyMatrix[i][j] > 0) {
          sampleEdges.push(`${i}-${j}:${adjacencyMatrix[i][j].toFixed(3)}`);
        }
      }
    }
    console.log(`Sample edges: ${sampleEdges.slice(0, 10).join(', ')}`);

    // Step 3: Run in UNSUPERVISED mode - no labels provided to GNN
    const nodeLabels = new Array(n).fill(null); // All nodes unlabeled for unsupervised learning
    
    console.log('Running GNN in UNSUPERVISED mode - no labels provided to the model');
    console.log('The GNN will learn patterns purely from graph structure and embeddings');

    // Step 4: Train GNN
    console.log('Training GNN...');
    const gnn = new SimpleGNN(embeddings, adjacencyMatrix, nodeLabels);
    const nodePredictions = gnn.train();

    // Step 5: Convert node predictions to Q&A level predictions with nearest labeled CoTs
    const qaPredictions = {};
    
    for (let qaIndex = 0; qaIndex < qaDataWithCoTs.length; qaIndex++) {
      const qa = qaDataWithCoTs[qaIndex];
      const posNodeIndex = qaIndex * 2;
      const negNodeIndex = qaIndex * 2 + 1;
      
      const posScore = nodePredictions[posNodeIndex];
      const negScore = nodePredictions[negNodeIndex];
      
      // Determine which CoT has higher confidence
      const isHallucinated = posScore > negScore;
      
      // Calculate confidence as the max score (how confident we are in the chosen prediction)
      const confidence = Math.max(posScore, negScore);
      
      // Alternative: Calculate confidence as the difference (how much we prefer one over the other)
      const scoreDifference = Math.abs(posScore - negScore);
      
      // Find nearest labeled CoTs for this prediction using Pinecone
      let nearestLabeledCoTs = [];
      const chosenVector = isHallucinated ? embeddings[posNodeIndex] : embeddings[negNodeIndex];
      
      try {
        const queryResponse = await index.query({
          vector: chosenVector,
          topK: 5,
          filter: { is_labeled: true },
          includeMetadata: true,
          includeValues: false
        });
        
        nearestLabeledCoTs = queryResponse.matches.map(match => ({
          id: match.id,
          score: match.score,
          metadata: match.metadata
        }));
      } catch (error) {
        console.warn(`Could not find nearest labeled CoTs for Q&A ${qaIndex}:`, error.message);
      }
      
      // Calculate graph connectivity for debugging
      const posConnections = adjacencyMatrix[posNodeIndex].filter(w => w > 0).length;
      const negConnections = adjacencyMatrix[negNodeIndex].filter(w => w > 0).length;
      
      // Find strongest connections for each node
      const posStrongestConnections = adjacencyMatrix[posNodeIndex]
        .map((weight, idx) => ({ idx, weight }))
        .filter(conn => conn.weight > 0)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 3);
        
      const negStrongestConnections = adjacencyMatrix[negNodeIndex]
        .map((weight, idx) => ({ idx, weight }))
        .filter(conn => conn.weight > 0)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 3);

      qaPredictions[qaIndex] = {
        label: isHallucinated ? 'hallucinated' : 'not_hallucinated',
        confidence: confidence,
        scores: {
          hallucinated: posScore,
          not_hallucinated: negScore
        },
        chosenCot: isHallucinated ? qa.cot_pos : qa.cot_neg,
        nearestLabeledCoTs,
        debug: {
          posNodeIndex,
          negNodeIndex,
          scoreDifference,
          rawScores: { pos: posScore, neg: negScore },
          graphConnectivity: {
            posConnections,
            negConnections,
            posStrongestConnections,
            negStrongestConnections
          }
        }
      };
    }

    // Step 6: Add analysis metrics
    const labeledCount = isUnsupervised ? 0 : Object.keys(labeledData).length;
    const predictedCount = qaDataWithCoTs.length - labeledCount;
    
    const avgConfidence = Object.values(qaPredictions)
      .reduce((sum, pred) => sum + pred.confidence, 0) / qaDataWithCoTs.length;

    console.log('GNN training completed');

    return Response.json({ 
      success: true, 
      predictions: qaPredictions,
      analysis: {
        totalQAs: qaDataWithCoTs.length,
        labeledCount,
        predictedCount,
        avgConfidence,
        trainingIterations: gnn.iterations,
        graphStats: {
          nodes: n,
          edges: adjacencyMatrix.flat().filter(w => w > 0).length / 2
        }
      }
    });

  } catch (error) {
    console.error('Error in train-gnn:', error);
    return Response.json({ 
      success: false, 
      error: error.message || 'Failed to train GNN'
    });
  }
} 