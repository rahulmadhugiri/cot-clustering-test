#!/usr/bin/env python3
"""
Diagnostic script to test propagation logic
Run this to debug propagation issues
"""

import requests
import json

def test_propagation():
    base_url = "http://localhost:8000/api/v1"
    
    print("🔍 Testing Propagation Logic...")
    
    # Test 1: Check if backend is running
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Backend health check: {response.json()}")
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        return
    
    # Test 2: Get Q&A pairs
    try:
        response = requests.get(f"{base_url}/qa-pairs")
        qa_pairs = response.json()
        print(f"✅ Found {len(qa_pairs)} Q&A pairs")
        
        # Check cluster distribution
        cluster_counts = {}
        for qa_key, qa_data in qa_pairs.items():
            num_clusters = len(qa_data.get('clusters', []))
            cluster_counts[num_clusters] = cluster_counts.get(num_clusters, 0) + 1
        
        print(f"📊 Cluster distribution: {dict(sorted(cluster_counts.items()))}")
        
    except Exception as e:
        print(f"❌ Error getting Q&A pairs: {e}")
        return
    
    # Test 3: Get representatives
    try:
        response = requests.get(f"{base_url}/representatives-diverse/30")
        representatives_data = response.json()
        representatives = representatives_data.get('selected_pairs', [])
        print(f"✅ Found {len(representatives)} representatives")
        
        # Show first few representatives
        for i, rep in enumerate(representatives[:3]):
            print(f"  Rep {i+1}: Cluster {rep['cluster_id']}, Trust {rep['trust_level']:.3f}")
            
    except Exception as e:
        print(f"❌ Error getting representatives: {e}")
        return
    
    # Test 4: Simulate propagation with sample labels
    sample_labels = {}
    for i, rep in enumerate(representatives[:5]):  # Label first 5
        qa_key = rep['qa_key']
        # Alternate between correct/incorrect for testing
        sample_labels[qa_key] = "correct" if i % 2 == 0 else "incorrect"
    
    print(f"📝 Testing with {len(sample_labels)} sample labels...")
    
    try:
        response = requests.post(f"{base_url}/propagate", json={
            "human_labels": sample_labels,
            "num_representatives": 30
        })
        
        if response.ok:
            result = response.json()
            summary = result.get('summary', {})
            print(f"🎯 Propagation Results:")
            print(f"  - Human labeled: {summary.get('human_labeled', 0)}")
            print(f"  - Propagated: {summary.get('propagated', 0)}")
            print(f"  - Unpropagated: {summary.get('unpropagated', 0)}")
            print(f"  - Outliers: {summary.get('outliers', 0)}")
            print(f"  - Coverage rate: {summary.get('coverage_rate', 0):.1%}")
            print(f"  - Automation rate: {summary.get('automation_rate', 0):.1%}")
            
            if summary.get('propagated', 0) == 0:
                print("\n🚨 Still 0 propagations - checking individual results...")
                qa_pairs_result = result.get('qa_pairs', [])
                
                # Check cluster overlap between labeled and unlabeled
                labeled_clusters = set()
                for qa_pair in qa_pairs_result:
                    if qa_pair.get('source') == 'HUMAN':
                        labeled_clusters.update(qa_pair.get('clusters', []))
                
                unlabeled_with_overlap = 0
                for qa_pair in qa_pairs_result:
                    if qa_pair.get('source') not in ['HUMAN', 'PROPAGATED']:
                        qa_clusters = set(qa_pair.get('clusters', []))
                        if qa_clusters.intersection(labeled_clusters):
                            unlabeled_with_overlap += 1
                
                print(f"  - Labeled clusters: {sorted(labeled_clusters)}")
                print(f"  - Unlabeled with cluster overlap: {unlabeled_with_overlap}")
                
        else:
            print(f"❌ Propagation failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during propagation: {e}")

if __name__ == "__main__":
    test_propagation() 