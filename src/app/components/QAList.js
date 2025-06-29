'use client';
import { useEffect, useState } from 'react';

export default function QAList() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Load the 300 Q&A pairs from cleaned CSV
    fetch('/cleaned_questions_answers.csv')
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.text();
      })
      .then((text) => {
        const lines = text.split('\n').slice(1).filter(line => line.trim()); // skip header and empty lines
        const parsed = lines.map((line, i) => {
          // Handle CSV with commas in content
          const parts = line.split(',');
          const question = parts[0];
          const answer = parts.slice(1).join(','); // Join back in case there are commas in the answer
          return { id: `qa-${i}`, question, answer, cot: '' };
        });
        setData(parsed);
        setLoading(false);
      })
      .catch((error) => {
        setError(error.message);
        setLoading(false);
      });
  }, []);

  const generateCots = async (qaId) => {
    const qa = data.find((item) => item.id === qaId);
    if (!qa) return;

    const res = await fetch('/api/generate-cots', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: qa.question, answer: qa.answer }),
    });

    const { content } = await res.json();

    const updated = data.map((item) =>
      item.id === qaId ? { ...item, cot: content } : item
    );
    setData(updated);
  };

  const generateAllCots = async () => {
    setLoading(true);
    const updated = [...data];
    
    for (let i = 0; i < updated.length; i++) {
      if (!updated[i].cot) {
        try {
          const res = await fetch('/api/generate-cots', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: updated[i].question, answer: updated[i].answer }),
          });
          const { content } = await res.json();
          updated[i].cot = content;
          setData([...updated]); // Update UI after each CoT
          
          // Small delay to respect rate limits
          await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (error) {
          console.error(`Error generating CoT for ${updated[i].id}:`, error);
        }
      }
    }
    setLoading(false);
  };

  const uploadToPinecone = async () => {
    const cotsWithData = data.filter(qa => qa.cot);
    
    if (cotsWithData.length === 0) {
      alert('No CoTs to upload. Generate CoTs first.');
      return;
    }
    
    setLoading(true);
    
    try {
      // Format data for backend
      const cotExamples = cotsWithData.map(qa => ({
        id: qa.id,
        question: qa.question,
        answer: qa.answer,
        cot: qa.cot,
        reasoning_pattern: null
      }));
      
      console.log(`Uploading ${cotExamples.length} CoTs to Pinecone...`);
      
      const response = await fetch('http://localhost:8000/api/v1/embeddings/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cotExamples)
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`✅ Successfully uploaded ${result.count} CoTs to Pinecone!`);
        console.log('Upload successful:', result);
      } else {
        const error = await response.text();
        alert(`❌ Upload failed: ${error}`);
        console.error('Upload failed:', error);
      }
      
    } catch (error) {
      alert(`❌ Upload error: ${error.message}`);
      console.error('Upload error:', error);
    }
    
    setLoading(false);
  };

  if (loading) return <div>Loading Q&A data...</div>;
  if (error) return <div>Error loading data: {error}</div>;
  if (data.length === 0) return <div>No Q&A data found.</div>;

  const cotsGenerated = data.filter(qa => qa.cot).length;

  return (
    <div>
      <div style={{ marginBottom: '1rem', padding: '1rem', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
        <h3>🧪 Large Scale Clustering Test - 300 Q&A Pairs</h3>
        <p><strong>Loaded:</strong> {data.length} Q&A pairs (no labels to avoid bias)</p>
        <p><strong>CoTs Generated:</strong> {cotsGenerated}/{data.length}</p>
        <p><strong>Format:</strong> Pure Logic CoTs for unbiased clustering</p>
        
        <div style={{ marginTop: '1rem' }}>
          <button 
            onClick={generateAllCots} 
            disabled={loading}
            style={{ 
              padding: '0.5rem 1rem', 
              marginRight: '1rem', 
              backgroundColor: '#007bff', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? 'Generating...' : 'Generate All CoTs'}
          </button>
          
          <button 
            onClick={uploadToPinecone}
            disabled={loading || cotsGenerated === 0}
            style={{ 
              padding: '0.5rem 1rem', 
              marginRight: '1rem', 
              backgroundColor: cotsGenerated > 0 ? '#fd7e14' : '#6c757d', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px',
              cursor: (cotsGenerated > 0 && !loading) ? 'pointer' : 'not-allowed'
            }}
          >
            Upload to Pinecone ({cotsGenerated})
          </button>
          
          <button 
            onClick={() => window.location.href = '/clusters'}
            disabled={cotsGenerated === 0}
            style={{ 
              padding: '0.5rem 1rem', 
              marginRight: '1rem', 
              backgroundColor: cotsGenerated > 0 ? '#28a745' : '#6c757d', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px',
              cursor: cotsGenerated > 0 ? 'pointer' : 'not-allowed'
            }}
          >
            Run Clustering ({cotsGenerated} CoTs)
          </button>
          
          <button 
            onClick={() => window.location.href = '/propagation'}
            disabled={cotsGenerated === 0}
            style={{ 
              padding: '0.5rem 1rem', 
              backgroundColor: cotsGenerated > 0 ? '#17a2b8' : '#6c757d', 
              color: 'white', 
              border: 'none', 
              borderRadius: '4px',
              cursor: cotsGenerated > 0 ? 'pointer' : 'not-allowed'
            }}
          >
            Test Propagation
          </button>
        </div>
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ backgroundColor: '#f8f9fa' }}>
            <th style={{ padding: '0.5rem', border: '1px solid #dee2e6', width: '5%' }}>ID</th>
            <th style={{ padding: '0.5rem', border: '1px solid #dee2e6', width: '35%' }}>Question</th>
            <th style={{ padding: '0.5rem', border: '1px solid #dee2e6', width: '40%' }}>Answer</th>
            <th style={{ padding: '0.5rem', border: '1px solid #dee2e6', width: '10%' }}>CoT Status</th>
            <th style={{ padding: '0.5rem', border: '1px solid #dee2e6', width: '10%' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 50).map((qa) => (
            <tr key={qa.id}>
              <td style={{ padding: '0.5rem', border: '1px solid #dee2e6', fontSize: '0.8rem' }}>
                {qa.id}
              </td>
              <td style={{ padding: '0.5rem', border: '1px solid #dee2e6', fontSize: '0.9rem' }}>
                {qa.question}
              </td>
              <td style={{ padding: '0.5rem', border: '1px solid #dee2e6', fontSize: '0.9rem' }}>
                {qa.answer?.substring(0, 150)}...
              </td>
              <td style={{ padding: '0.5rem', border: '1px solid #dee2e6', textAlign: 'center' }}>
                {qa.cot ? '✅' : '⏳'}
              </td>
              <td style={{ padding: '0.5rem', border: '1px solid #dee2e6', textAlign: 'center' }}>
                <button 
                  onClick={() => generateCots(qa.id)}
                  style={{ 
                    padding: '0.25rem 0.5rem', 
                    fontSize: '0.8rem',
                    backgroundColor: qa.cot ? '#6c757d' : '#007bff',
                    color: 'white', 
                    border: 'none', 
                    borderRadius: '3px',
                    cursor: 'pointer'
                  }}
                >
                  {qa.cot ? 'Regenerate' : 'Generate'}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {data.length > 50 && (
        <p style={{ marginTop: '1rem', fontStyle: 'italic', color: '#666' }}>
          Showing first 50 of {data.length} Q&A pairs. Use "Generate All CoTs" to process all pairs.
        </p>
      )}
    </div>
  );
}
