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
          return { id: `qa-${i}`, question, answer, label };
        });
        setData(parsed);
      });
  }, []);

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
              <button onClick={() => alert(`TODO: Generate CoTs for ${qa.id}`)}>
                Generate CoTs
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
} 