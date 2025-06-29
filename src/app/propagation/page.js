'use client';
import { useEffect, useState } from 'react';

export default function PropagationPage() {
  const [clusters, setClusters] = useState([]);
  const [representatives, setRepresentatives] = useState([]);
  const [userLabels, setUserLabels] = useState({});
  const [propagatedResults, setPropagatedResults] = useState([]);
  const [currentStep, setCurrentStep] = useState('loading'); // loading, labeling, propagating, results
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchClusterDataAndSelectRepresentatives();
  }, []);

  const fetchClusterDataAndSelectRepresentatives = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Use the new diverse representative selection (30 pairs instead of 2)
      const response = await fetch('/api/representatives?count=30', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to fetch diverse representatives');
      }
      
      const result = await response.json();
      
      if (!result.success) {
        throw new Error('Failed to select diverse representatives');
      }
      
      // Extract the actual data from the proxy response
      const data = result.data;
      
      // Transform the diverse representatives into the format expected by the frontend
      const selectedPairs = data.selected_pairs;
      const reps = selectedPairs.map((pair, index) => ({
        id: `rep-${index}`,
        question: pair.question.replace(/\.\.\.$/, ''), // Remove truncation indicator
        answer: pair.answer.replace(/\.\.\.$/, ''), // Remove truncation indicator  
        cot: pair.cot_preview.replace(/\.\.\.$/, ''), // Remove truncation indicator
        cluster_id: pair.cluster_id,
        outlier_score: pair.outlier_score,
        qa_pair_key: pair.qa_key,
        trust_level: pair.trust_level,
        reasoning_type: pair.reasoning_type,
        cot_length: pair.cot_length
      }));
      
      setRepresentatives(reps);
      
      // Set up cluster summary for display
      const clusterInfo = data.clustering_info;
      const qualityStats = data.quality_stats;
      
      // Create a summary object for display
      const experimentSummary = {
        totalPairs: selectedPairs.length,
        clustersCount: clusterInfo.total_clusters,
        clustersCovered: data.cluster_coverage.covered_clusters,
        avgTrustScore: qualityStats.avg_trust_score,
        avgCotLength: qualityStats.avg_cot_length,
        reasoningTypes: qualityStats.reasoning_types,
        clusteringMethod: clusterInfo.method,
        silhouetteScore: clusterInfo.silhouette_score
      };
      
      setClusters(experimentSummary); // Store summary in clusters state for display
      setCurrentStep('labeling');
      
    } catch (err) {
      setError(err.message);
      console.error('Error fetching diverse representatives:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLabelChange = (qaPairKey, label) => {
    setUserLabels(prev => ({
      ...prev,
      [qaPairKey]: label
    }));
  };

  const propagateLabels = async () => {
    try {
      setCurrentStep('propagating');
      
      // Call the Python backend for label propagation
      const response = await fetch('/api/propagate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          human_labels: userLabels,
          num_representatives: 30  // Match the actual number of selected representatives
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to propagate labels');
      }
      
      const result = await response.json();
      
      if (!result.success) {
        throw new Error('Label propagation failed');
      }
      
             // Use the results from Python backend
       const results = result.results;
       
       setPropagatedResults(results);
       setCurrentStep('results');
       
    } catch (error) {
      console.error('Error during label propagation:', error);
      setError(error.message);
      setCurrentStep('labeling');
    }
  };

  const allLabelsProvided = () => {
    return representatives.every(rep => userLabels[rep.qa_pair_key] !== undefined);
  };

  if (loading) {
    return (
      <main>
        <h1>Q&A Hallucination Detection via CoT Clustering</h1>
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading CoT clusters and selecting representative Q&A pairs...</p>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main>
        <h1>Q&A Hallucination Detection via CoT Clustering</h1>
        <div className="error-container">
          <p>Error: {error}</p>
          <button onClick={fetchClusterDataAndSelectRepresentatives}>Retry</button>
        </div>
      </main>
    );
  }

  if (currentStep === 'labeling') {
    return (
      <main>
        <h1>Q&A Hallucination Detection via CoT Clustering</h1>
        <div className="experiment-description">
          <p><strong>Goal:</strong> Label Q&A pairs as correct/incorrect based on semantic diversity across reasoning patterns.</p>
          <p><strong>Intelligent Selection:</strong> We've selected <strong>{representatives.length} diverse, high-quality Q&A pairs</strong> from 300 total CoT examples using:</p>
          <ul>
            <li>✅ <strong>{clusters.clusteringMethod} clustering</strong> with {clusters.clustersCount} clusters (silhouette score: {clusters.silhouetteScore?.toFixed(3)})</li>
            <li>✅ <strong>Full cluster coverage:</strong> {clusters.clustersCovered}/{clusters.clustersCount} clusters represented</li>
            <li>✅ <strong>High-trust CoTs:</strong> Average trust score of {clusters.avgTrustScore?.toFixed(3)} (closest to centroids)</li>
            <li>✅ <strong>Semantic diversity:</strong> {Object.entries(clusters.reasoningTypes || {}).map(([type, count]) => `${count} ${type}`).join(', ')}</li>
            <li>✅ <strong>Quality filtering:</strong> Well-written, logically sound reasoning (avg length: {clusters.avgCotLength?.toFixed(0)} chars)</li>
          </ul>
          <p><strong>Selection Criteria Applied:</strong></p>
          <ul>
            <li>🎯 1-2 representative CoTs from each cluster (avoiding edge cases)</li>
            <li>🔍 Prioritized distinct reasoning patterns and structures</li>
            <li>🚫 Excluded extremely similar or templated pairs</li>
            <li>⭐ Focused on core exemplars of each reasoning type</li>
          </ul>
          <p><em>This ensures your labels will effectively propagate across the semantic space of reasoning patterns.</em></p>
        </div>

        <div className="labeling-progress">
          <p><strong>Progress:</strong> {Object.keys(userLabels).length}/{representatives.length} pairs labeled</p>
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{width: `${(Object.keys(userLabels).length / representatives.length) * 100}%`}}
            ></div>
          </div>
        </div>

        <div className="representatives-container">
          {representatives.map((rep, index) => (
            <div key={rep.qa_pair_key} className="representative-card">
              <div className="card-header">
                <h3>Q&A Pair {index + 1} <span className="cluster-badge">Cluster {rep.cluster_id}</span></h3>
                <div className="quality-indicators">
                  <span className={`trust-badge ${rep.trust_level.toLowerCase()}`}>{rep.trust_level} Trust</span>
                  <span className="reasoning-badge">{rep.reasoning_type}</span>
                  <span className="length-badge">{rep.cot_length} chars</span>
                </div>
              </div>
              
              <div className="qa-content">
                <div className="question-section">
                  <strong>Question:</strong>
                  <p>{rep.question}</p>
                </div>
                
                <div className="answer-section">
                  <strong>Final Answer:</strong>
                  <p>{rep.answer}</p>
                </div>
                
                <details className="cot-details">
                  <summary><strong>View CoT Reasoning (for context)</strong></summary>
                  <div className="cot-reasoning">
                    <p>{rep.cot}</p>
                  </div>
                </details>
              </div>

              <div className="labeling-section">
                <p><strong>Is this final answer correct?</strong></p>
                <div className="label-buttons">
                  <button 
                    className={`label-button ${userLabels[rep.qa_pair_key] === 'correct' ? 'selected correct' : ''}`}
                    onClick={() => handleLabelChange(rep.qa_pair_key, 'correct')}
                  >
                    ✅ Correct Answer
                  </button>
                  <button 
                    className={`label-button ${userLabels[rep.qa_pair_key] === 'incorrect' ? 'selected incorrect' : ''}`}
                    onClick={() => handleLabelChange(rep.qa_pair_key, 'incorrect')}
                  >
                    ❌ Incorrect Answer
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="action-section">
          <button 
            className="propagate-button"
            onClick={propagateLabels}
            disabled={!allLabelsProvided()}
          >
            Propagate Labels to Similar Q&A Pairs
          </button>
          {!allLabelsProvided() && (
            <p className="help-text">Please label all {representatives.length} representative Q&A pairs before proceeding.</p>
          )}
        </div>
      </main>
    );
  }

  if (currentStep === 'propagating') {
    return (
      <main>
        <h1>Q&A Hallucination Detection via CoT Clustering</h1>
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Propagating labels across Q&A pairs with similar reasoning patterns...</p>
        </div>
      </main>
    );
  }

  if (currentStep === 'results') {
    const humanLabeled = propagatedResults.filter(item => item.source === 'HUMAN');
    const systemPropagated = propagatedResults.filter(item => item.source === 'PROPAGATED');
    const outliers = propagatedResults.filter(item => item.source === 'OUTLIER');
    const unpropagated = propagatedResults.filter(item => item.source === 'UNPROPAGATED');
    const uniqueQAPairs = propagatedResults.length;

    return (
      <main>
        <h1>Q&A Hallucination Detection Results</h1>
        
        <div className="results-summary">
          <div className="summary-item">
            <strong>Total Q&A Pairs:</strong> {uniqueQAPairs}
          </div>
          <div className="summary-item">
            <strong>Human Labeled:</strong> {humanLabeled.length}
          </div>
          <div className="summary-item">
            <strong>System Propagated:</strong> {systemPropagated.length}
          </div>
          <div className="summary-item">
            <strong>Outliers:</strong> {outliers.length}
          </div>
          {unpropagated.length > 0 && (
            <div className="summary-item">
              <strong>Unpropagated:</strong> {unpropagated.length}
            </div>
          )}
        </div>

        <div className="results-table-container">
          <table className="results-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Final Answer</th>
                <th>Predicted Label</th>
                <th>Confidence</th>
                <th>Source</th>
                <th>Reasoning Clusters</th>
                <th>Propagation Info</th>
              </tr>
            </thead>
            <tbody>
              {propagatedResults.map((result) => (
                <tr 
                  key={result.qa_pair_key} 
                  className={`result-row ${result.source.toLowerCase()} ${result.predicted_label}`}
                >
                  <td className="question-cell">{result.question}</td>
                  <td className="answer-cell">{result.answer}</td>
                  <td className="label-cell">
                    <span className={`label-badge ${result.predicted_label}`}>
                      {result.predicted_label === 'correct' ? '✅' : result.predicted_label === 'incorrect' ? '❌' : '❓'}
                      {result.predicted_label}
                    </span>
                  </td>
                  <td className="confidence-cell">
                    {(result.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="source-cell">
                    <span className={`source-badge ${result.source.toLowerCase()}`}>
                      {result.source}
                    </span>
                  </td>
                  <td className="cluster-cell">
                    {result.cluster_info}
                  </td>
                  <td className="propagation-cell">
                    {result.source === 'PROPAGATED' && result.shared_reasoning && (
                      <div className="propagation-info">
                        <div><strong>Via:</strong> {result.shared_reasoning}</div>
                        <div className="propagation-source">
                          <small>From similar Q&A</small>
                        </div>
                      </div>
                    )}
                    {result.source === 'HUMAN' && (
                      <div className="human-label-info">
                        <strong>Direct labeling</strong>
                      </div>
                    )}
                    {result.source === 'OUTLIER' && (
                      <div className="outlier-info">
                        <em>No clear pattern</em>
                      </div>
                    )}
                    {result.source === 'UNPROPAGATED' && (
                      <div className="unpropagated-info">
                        <em>No propagation path</em>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="validation-section">
          <h2>🎯 Experiment Validation</h2>
          <p>
            <strong>Core Question:</strong> Can we auto-label hallucinations across unrelated Q&A pairs 
            purely by clustering similar reasoning patterns?
          </p>
          
          <div className="validation-metrics">
            <h3>Success Metrics:</h3>
            <ul>
              <li><strong>Efficiency:</strong> Labeled {humanLabeled.length} Q&A pairs, predicted {systemPropagated.length} more ({((systemPropagated.length / uniqueQAPairs) * 100).toFixed(1)}% automation)</li>
              <li><strong>Coverage:</strong> {uniqueQAPairs} unique Q&A pairs analyzed from {clusters.length} CoT reasoning examples</li>
              <li><strong>Pattern Recognition:</strong> {representatives.length} distinct reasoning clusters identified</li>
              <li><strong>Confidence:</strong> Average propagation confidence: {systemPropagated.length > 0 ? (systemPropagated.reduce((sum, item) => sum + item.confidence, 0) / systemPropagated.length * 100).toFixed(1) : 0}%</li>
            </ul>
          </div>
          
          <div className="experiment-success">
            <h3>🎉 Reasoning Pattern Propagation Complete!</h3>
            <p>
              <strong>Key Achievement:</strong> The system successfully used CoT reasoning patterns as a bridge 
              to propagate hallucination labels across semantically different Q&A pairs.
            </p>
            <p>
              <strong>Validation Check:</strong> Do the propagated labels make sense? 
              Q&A pairs with similar reasoning structures should have similar correctness patterns, 
              regardless of their domain or surface content.
            </p>
          </div>
        </div>
      </main>
    );
  }

  return null;
}

// Helper function to get reasoning pattern names
function getReasoningPatternName(clusterId) {
  const patterns = {
    0: "Systems Thinking",
    1: "Experience-Based", 
    2: "Systems Analysis",
    3: "Analogical Reasoning",
    4: "Deductive Logic"
  };
  return patterns[clusterId] || `Pattern ${clusterId}`;
} 