# Reasoning Pattern Clustering Backend

A FastAPI-based backend for CoT reasoning pattern clustering and hallucination detection research.

## 🏗️ Architecture

```
backend/
├── src/
│   ├── models/         # Pydantic data models
│   ├── services/       # Core ML services
│   ├── api/           # FastAPI routes
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter analysis notebooks
├── tests/            # Test suite
├── requirements.txt  # Python dependencies
└── main.py          # FastAPI application entry point
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
HOST=0.0.0.0
PORT=8000
```

### 3. Start the Server

```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📊 Research Workflow

### 1. Data Loading
```python
# Load datasets
GET /api/v1/datasets
GET /api/v1/dataset/{dataset_name}
```

### 2. Embedding Generation
```python
# Generate embeddings for CoT examples
POST /api/v1/embeddings/generate
```

### 3. Clustering
```python
# Cluster CoT examples using HDBSCAN
POST /api/v1/cluster
```

### 4. Representative Selection
```python
# Select Q&A pairs for human labeling
GET /api/v1/representatives/{num_representatives}
```

### 5. Label Propagation
```python
# Propagate human labels to other Q&A pairs
POST /api/v1/propagate
```

### 6. Evaluation
```python
# Evaluate results against ground truth
POST /api/v1/evaluate
```

## 🔬 Core Services

### ClusteringService
- HDBSCAN clustering of CoT embeddings
- Q&A pair grouping and analysis
- Representative selection algorithms
- Similarity matrix calculations

### PropagationService
- Label propagation based on reasoning pattern similarity
- Confidence score calculation
- Performance evaluation metrics

### EmbeddingService
- OpenAI embedding generation
- Pinecone vector database integration
- Similarity search capabilities

## 📈 Analysis & Visualization

### Jupyter Notebooks
```bash
cd backend/notebooks
jupyter notebook experiment_analysis.ipynb
```

The analysis notebook provides:
- Clustering quality assessment
- Propagation effectiveness analysis
- Interactive visualizations
- Research-ready result exports

### Key Metrics
- **Human Effort Reduction**: Percentage of manual labeling saved
- **Automation Rate**: Percentage of labels propagated automatically
- **Coverage Rate**: Percentage of Q&A pairs with confident predictions
- **Cluster Coherence**: Quality of reasoning pattern separation

## 🧪 Research Features

### Advanced Clustering
- Cosine similarity-based HDBSCAN
- Outlier detection and scoring
- Cluster quality metrics (silhouette score)
- Multi-dimensional reasoning pattern analysis

### Intelligent Propagation
- Jaccard similarity for cluster overlap
- Confidence-bounded label transfer
- Cross-reasoning pattern propagation
- Uncertainty handling for outliers

### Comprehensive Evaluation
- Ground truth comparison
- Source-stratified accuracy metrics
- Confidence calibration analysis
- Statistical significance testing

## 🔧 Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Type checking
mypy src/

# Linting
flake8 src/

# Formatting
black src/
```

### Adding New Features

1. **New Endpoints**: Add to `src/api/routes.py`
2. **New Services**: Create in `src/services/`
3. **New Models**: Define in `src/models/schemas.py`
4. **New Utilities**: Add to `src/utils/`

## 📚 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/datasets` | List available datasets |
| GET | `/dataset/{name}` | Load specific dataset |
| POST | `/embeddings/generate` | Generate embeddings |
| POST | `/cluster` | Perform clustering |
| GET | `/qa-pairs` | Get Q&A pairs |
| GET | `/representatives/{n}` | Select representatives |
| POST | `/propagate` | Propagate labels |
| POST | `/evaluate` | Evaluate results |

### Analysis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/experiment/results` | Complete experiment data |
| GET | `/analysis/similarity-matrix` | Q&A pair similarity |

## 🎯 Research Applications

This backend enables research in:

- **Hallucination Detection**: Identify unreliable AI responses
- **Reasoning Pattern Analysis**: Study different types of logical structures
- **Few-Shot Learning**: Minimal supervision for maximum coverage
- **Cross-Domain Transfer**: Reasoning patterns across different topics
- **Quality Assessment**: Automated content moderation

## 📊 Performance Characteristics

- **Embedding Generation**: ~100ms per CoT (OpenAI API dependent)
- **Clustering**: ~50ms for 15 examples (HDBSCAN)
- **Propagation**: ~10ms for 5 Q&A pairs
- **Memory Usage**: ~50MB base + ~1KB per embedding

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This research code is provided for academic and research purposes. 