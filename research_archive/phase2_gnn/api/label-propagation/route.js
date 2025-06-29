import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

// Cosine similarity calculation
function cosineSimilarity(vecA, vecB) {
  if (vecA.length !== vecB.length) {
    throw new Error('Vectors must have the same length');
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  if (normA === 0 || normB === 0) {
    return 0;
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Load data files
function loadData() {
  const dataPath = path.join(process.cwd(), 'data');
  
  const cotsData = JSON.parse(
    fs.readFileSync(path.join(dataPath, 'all_300_cots.json'), 'utf8')
  );
  
  const embeddingsData = JSON.parse(
    fs.readFileSync(path.join(dataPath, 'all_300_embeddings.json'), 'utf8')
  );
  
  return { cotsData, embeddingsData };
}

// Select diverse sample for labeling
function selectDiverseSample(cotsData, embeddingsData, sampleSize = 30) {
  // Create a simple diversity-based selection
  // We'll use a greedy approach to select diverse examples
  const selected = [];
  const remaining = [...cotsData];
  
  // First, select a random starting point
  const firstIndex = Math.floor(Math.random() * remaining.length);
  selected.push(remaining.splice(firstIndex, 1)[0]);
  
  // Then select the most diverse remaining examples
  while (selected.length < sampleSize && remaining.length > 0) {
    let maxMinDistance = -1;
    let bestIndex = 0;
    
    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];
      const candidateEmbedding = embeddingsData.find(e => e.id === candidate.id)?.embedding;
      
      if (!candidateEmbedding) continue;
      
      // Find minimum distance to already selected items
      let minDistance = Infinity;
      for (const selectedItem of selected) {
        const selectedEmbedding = embeddingsData.find(e => e.id === selectedItem.id)?.embedding;
        if (selectedEmbedding) {
          const similarity = cosineSimilarity(candidateEmbedding, selectedEmbedding);
          const distance = 1 - similarity; // Convert similarity to distance
          minDistance = Math.min(minDistance, distance);
        }
      }
      
      if (minDistance > maxMinDistance) {
        maxMinDistance = minDistance;
        bestIndex = i;
      }
    }
    
    selected.push(remaining.splice(bestIndex, 1)[0]);
  }
  
  return selected;
}

// Perform label propagation using k-NN
function propagateLabels(labeledData, unlabeledData, embeddingsData, k = 3) {
  const results = [];
  
  for (const unlabeledItem of unlabeledData) {
    const unlabeledEmbedding = embeddingsData.find(e => e.id === unlabeledItem.id)?.embedding;
    
    if (!unlabeledEmbedding) {
      results.push({
        ...unlabeledItem,
        predicted_label: 'uncertain',
        confidence: 0,
        nearest_neighbors: [],
        source: 'no_embedding'
      });
      continue;
    }
    
    // Calculate similarities to all labeled items
    const similarities = [];
    for (const labeledItem of labeledData) {
      const labeledEmbedding = embeddingsData.find(e => e.id === labeledItem.id)?.embedding;
      if (labeledEmbedding) {
        const similarity = cosineSimilarity(unlabeledEmbedding, labeledEmbedding);
        similarities.push({
          item: labeledItem,
          similarity: similarity,
          label: labeledItem.user_label
        });
      }
    }
    
    // Sort by similarity and take top k
    similarities.sort((a, b) => b.similarity - a.similarity);
    const topK = similarities.slice(0, Math.min(k, similarities.length));
    
    if (topK.length === 0) {
      results.push({
        ...unlabeledItem,
        predicted_label: 'uncertain',
        confidence: 0,
        nearest_neighbors: [],
        source: 'no_neighbors'
      });
      continue;
    }
    
    // Weighted voting based on similarity
    let positiveWeight = 0;
    let negativeWeight = 0;
    
    for (const neighbor of topK) {
      if (neighbor.label === 'correct') {
        positiveWeight += neighbor.similarity;
      } else if (neighbor.label === 'incorrect') {
        negativeWeight += neighbor.similarity;
      }
    }
    
    const totalWeight = positiveWeight + negativeWeight;
    let predicted_label = 'uncertain';
    let confidence = 0;
    
    if (totalWeight > 0) {
      if (positiveWeight > negativeWeight) {
        predicted_label = 'correct';
        confidence = positiveWeight / totalWeight;
      } else if (negativeWeight > positiveWeight) {
        predicted_label = 'incorrect';
        confidence = negativeWeight / totalWeight;
      } else {
        predicted_label = 'uncertain';
        confidence = 0.5;
      }
    }
    
    results.push({
      ...unlabeledItem,
      predicted_label,
      confidence,
      nearest_neighbors: topK.map(n => ({
        id: n.item.id,
        question: n.item.question.substring(0, 100) + '...',
        label: n.label,
        similarity: n.similarity
      })),
      source: 'propagated'
    });
  }
  
  return results;
}

export async function GET(req) {
  try {
    const { searchParams } = new URL(req.url);
    const sampleSize = parseInt(searchParams.get('sample_size') || '30');
    
    const { cotsData, embeddingsData } = loadData();
    
    // Select diverse sample for labeling
    const sampleForLabeling = selectDiverseSample(cotsData, embeddingsData, sampleSize);
    
    return NextResponse.json({
      success: true,
      sample: sampleForLabeling,
      total_available: cotsData.length
    });
    
  } catch (error) {
    console.error('Error in label propagation GET:', error);
    return NextResponse.json({
      error: 'Failed to prepare sample for labeling',
      details: error.message
    }, { status: 500 });
  }
}

export async function POST(req) {
  try {
    const { labeled_data, label_dimension } = await req.json();
    
    if (!labeled_data || !Array.isArray(labeled_data)) {
      return NextResponse.json({
        error: 'labeled_data must be an array'
      }, { status: 400 });
    }
    
    const { cotsData, embeddingsData } = loadData();
    
    // Separate labeled and unlabeled data
    const labeledIds = new Set(labeled_data.map(item => item.id));
    const unlabeledData = cotsData.filter(item => !labeledIds.has(item.id));
    
    // Perform label propagation
    const propagationResults = propagateLabels(labeled_data, unlabeledData, embeddingsData);
    
    // Combine labeled and propagated results
    const allResults = [
      ...labeled_data.map(item => ({
        ...item,
        predicted_label: item.user_label,
        confidence: 1.0,
        source: 'human_labeled',
        nearest_neighbors: []
      })),
      ...propagationResults
    ];
    
    // Calculate summary statistics
    const summary = {
      total_items: allResults.length,
      human_labeled: labeled_data.length,
      propagated: propagationResults.length,
      label_distribution: {
        correct: allResults.filter(r => r.predicted_label === 'correct').length,
        incorrect: allResults.filter(r => r.predicted_label === 'incorrect').length,
        uncertain: allResults.filter(r => r.predicted_label === 'uncertain').length
      },
      avg_confidence: propagationResults.reduce((sum, r) => sum + r.confidence, 0) / propagationResults.length,
      label_dimension: label_dimension || 'Quality Assessment'
    };
    
    return NextResponse.json({
      success: true,
      results: allResults,
      summary
    });
    
  } catch (error) {
    console.error('Error in label propagation POST:', error);
    return NextResponse.json({
      error: 'Failed to perform label propagation',
      details: error.message
    }, { status: 500 });
  }
} 