'use client';
import { useEffect, useState } from 'react';

export default function ClustersPage() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    fetchClusterData();
  }, []);

  const fetchClusterData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/hdbscan');
      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.error || 'Failed to fetch cluster data');
      }
      
      setData(result.data);
      setSummary(result.summary);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching cluster data:', err);
    } finally {
      setLoading(false);
    }
  };

  const truncateText = (text, maxLength = 100) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  const getClusterColor = (clusterId) => {
    if (clusterId === -1) return '#ffebee'; // Light red for outliers
    
    // Generate consistent colors for clusters
    const colors = [
      '#e3f2fd', // light blue
      '#f3e5f5', // light purple
      '#e8f5e8', // light green
      '#fff3e0', // light orange
      '#fce4ec', // light pink
      '#f1f8e9', // light lime
      '#e0f2f1', // light teal
    ];
    
    return colors[clusterId % colors.length] || '#f5f5f5';
  };

  if (loading) {
    return (
      <main>
        <h1>CoT Clustering Results</h1>
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Running HDBSCAN clustering on CoT embeddings...</p>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main>
        <h1>CoT Clustering Results</h1>
        <div className="error-container">
          <p>Error: {error}</p>
          <button onClick={fetchClusterData}>Retry</button>
        </div>
      </main>
    );
  }

  return (
    <main>
      <h1>CoT Clustering Results Using HDBSCAN</h1>
      
      {summary && (
        <div className="summary-container">
          <div className="summary-item">
            <strong>Total CoTs:</strong> {summary.total_vectors}
          </div>
          <div className="summary-item">
            <strong>Clusters Found:</strong> {summary.clusters}
          </div>
          <div className="summary-item">
            <strong>Outliers:</strong> {summary.outliers}
          </div>
        </div>
      )}

      <div className="table-container">
        <table className="clusters-table">
          <thead>
            <tr>
              <th>CoT ID</th>
              <th>Cluster ID</th>
              <th>Cluster Size</th>
              <th>Outlier Score</th>
              <th>Is Outlier</th>
              <th>Answer</th>
              <th>Question</th>
              <th>CoT (Reasoning)</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr 
                key={item.id} 
                className={item.is_outlier ? 'outlier-row' : ''}
                style={{ backgroundColor: getClusterColor(item.cluster_id) }}
              >
                <td className="cot-id">{item.id}</td>
                <td className="cluster-id">
                  {item.cluster_id === -1 ? 'Outlier' : `Cluster ${item.cluster_id}`}
                </td>
                <td className="cluster-size">
                  {item.cluster_size || 'N/A'}
                </td>
                <td className="outlier-score">
                  {item.outlier_score.toFixed(4)}
                </td>
                <td className="is-outlier">
                  {item.is_outlier ? '✓' : '✗'}
                </td>
                <td className="answer">
                  {truncateText(item.answer, 60)}
                </td>
                <td className="question">
                  {truncateText(item.question, 60)}
                </td>
                <td className="cot-text">
                  {truncateText(item.cot, 100)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
} 