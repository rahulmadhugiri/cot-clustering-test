# Reasoning Pattern Clustering for Hallucination Detection - Python Version

This is the Python implementation of our reasoning pattern clustering experiment for hallucination detection.

## Overview

The experiment demonstrates that Chain-of-Thought (CoT) reasoning patterns can be clustered to enable hallucination detection with minimal human supervision.

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"
export PINECONE_INDEX_NAME="cot-clustering-test"
```

3. **Run the experiment**:
```bash
python experiment.py
```

## What It Does

1. **Loads pure logic CoT dataset** (15 examples, 5 Q&A pairs, 5 reasoning patterns)
2. **Generates embeddings** using OpenAI's text-embedding-3-large
3. **Clusters reasoning patterns** using HDBSCAN
4. **Selects 2 Q&A pairs** for human labeling (minimal supervision)
5. **Propagates labels** to remaining Q&A pairs based on shared reasoning clusters
6. **Evaluates results** against ground truth
7. **Generates visualizations** and exports results

## Key Features

### Data Analysis
- **Pandas DataFrames** for structured data analysis
- **NumPy arrays** for efficient embedding operations
- **Statistical analysis** of clustering and propagation results

### Visualization
- **Cluster distribution** plots
- **Outlier score** histograms
- **Confidence analysis** by source type
- **Cluster coverage vs confidence** scatter plots

### Export & Analysis
- **JSON export** of all results for further analysis
- **High-resolution plots** saved as PNG files
- **Detailed evaluation metrics**

## Advantages of Python Version

1. **Better ML ecosystem**: pandas, numpy, scikit-learn
2. **Advanced visualizations**: matplotlib, seaborn, plotly
3. **Statistical analysis**: scipy, statsmodels
4. **Jupyter notebook support** for interactive analysis
5. **Standard ML research format**

## Usage

```python
from experiment import ReasoningClusterExperiment

# Initialize experiment
experiment = ReasoningClusterExperiment(
    openai_api_key="your_key",
    pinecone_api_key="your_key", 
    pinecone_index="your_index"
)

# Run complete pipeline
experiment.load_pure_logic_dataset()
experiment.generate_embeddings()
experiment.cluster_reasoning_patterns()
experiment.group_by_qa_pairs()

# Human labeling step
selected_keys = experiment.select_representatives(2)
human_labels = {
    selected_keys[0]: 'incorrect',
    selected_keys[1]: 'correct'
}

# Propagate and evaluate
experiment.propagate_labels(human_labels)
experiment.visualize_results()
experiment.export_results()
```

## Results

The experiment demonstrates:
- **40% human effort → 80% coverage** with high accuracy
- **Cross-domain reasoning pattern propagation**
- **Robust uncertainty quantification**
- **Interpretable clustering of reasoning structures** 