'use client';
import { useEffect, useState } from 'react';

export default function QAList() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('/test.csv')
      .then((res) => res.text())
      .then((text) => {
        const lines = text.split('\n').slice(1); // skip header
        const parsed = lines.map((line, i) => {
          const [question, answer, label] = line.split(',');
          return { id: `qa-${i}`, question, answer, label, cot: '' };
        });
        setData(parsed);
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

  return (
    <table>
      <thead>
        <tr>
          <th>Question</th>
          <th>Answer</th>
          <th>Label</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {data.map((qa) => (
          <tr key={qa.id}>
            <td>{qa.question}</td>
            <td>{qa.answer}</td>
            <td>{qa.label}</td>
            <td>
              <button onClick={() => generateCots(qa.id)}>Generate CoTs</button>
              {qa.cot && (
                <div style={{ marginTop: '0.5rem', whiteSpace: 'pre-wrap' }}>
                  <strong>CoTs:</strong>
                  <br />
                  {qa.cot}
                </div>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
