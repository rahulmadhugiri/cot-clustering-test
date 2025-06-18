'use client';
import QAList from '../components/QAList';
import Link from 'next/link';

export default function HomePage() {
  return (
    <main>
      <h1>Reasoning Cluster Labeling Test</h1>
      <div className="navigation-links">
        <Link href="/clusters" className="nav-link">
          View CoT Clusters
        </Link>
      </div>
      <QAList />
    </main>
  );
}
