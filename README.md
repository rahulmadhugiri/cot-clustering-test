# 🧠 CoT Reasoning Pattern Clustering Research

A comprehensive research platform for **Chain-of-Thought reasoning pattern clustering** and **hallucination detection** using minimal human supervision.

## 🎯 Research Goal

Demonstrate that **reasoning patterns** (not semantic content) can be clustered to enable **label propagation** for hallucination detection with minimal human effort.

### Key Innovation
- **40% human labeling** → **80% coverage** via reasoning pattern similarity
- Cross-domain propagation (automotive Q&A → viscosity questions)
- Pure logical structure analysis (mathematical notation)

## 🏗️ Architecture

```
cot-clustering-research/
├── backend/                 # 🐍 Python ML Backend (FastAPI)
│   ├── src/
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # ML services (clustering, propagation)
│   │   ├── api/           # FastAPI routes
│   │   └── utils/         # Data loading utilities
│   ├── notebooks/         # 📊 Jupyter analysis
│   ├── requirements.txt   # Python dependencies
│   └── main.py           # FastAPI server
├── app/                   # ⚛️ Next.js Frontend (existing structure)
│   ├── api/              # API routes (call Python backend)
│   ├── clusters/         # Clustering page
│   ├── propagation/      # Label propagation page
│   └── layout.js         # App layout
├── components/           # React UI components
├── styles/              # CSS styling (pure CSS, no inline)
├── data/                # 📁 Datasets
│   ├── pure-logic-cots.js  # Core research dataset
│   ├── mixed-cots.js      # Mixed domain examples
│   └── abstract-cots.js   # Abstract examples
└── python_version/      # 🔬 Standalone Python research
```

## 🚀 Quick Start

### 1. Setup Backend (Python ML)

```bash
cd backend
pip3 install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
echo "PINECONE_API_KEY=your_key_here" >> .env

# Start FastAPI server
python3 main.py
```

### 2. Setup Frontend (Next.js)

```bash
# Install Next.js dependencies (in root directory)
npm install

# Start Next.js frontend
npm run dev
```

### 3. Quick Start (Both Services)

```bash
# Easy way: Start both backend + frontend
./start-dev.sh
```

### 4. Access the Platform

- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8000/docs
- **Jupyter Analysis**: `cd backend && jupyter notebook`

## 🔬 Research Workflow

### Phase 1: Data Preparation
1. **Pure Logic Dataset**: 5 Q&A pairs × 3 reasoning patterns each
2. **Embedding Generation**: OpenAI text-embedding-3-large (1024d)
3. **Vector Storage**: Pinecone for similarity search

### Phase 2: Clustering Analysis
1. **HDBSCAN Clustering**: Density-based reasoning pattern detection
2. **Pattern Identification**: 5 distinct reasoning clusters found
3. **Quality Assessment**: Silhouette score, outlier detection

### Phase 3: Label Propagation
1. **Representative Selection**: Choose 2 Q&A pairs (40% of data)
2. **Human Labeling**: Manual correct/incorrect labels
3. **Automatic Propagation**: Cluster overlap → label transfer
4. **Confidence Scoring**: Jaccard similarity-based confidence

### Phase 4: Evaluation
1. **Coverage Analysis**: 4/5 Q&A pairs successfully labeled
2. **Accuracy Assessment**: 80% overall accuracy achieved
3. **Cross-Domain Validation**: Reasoning patterns transfer across topics

## 📊 Key Results

| Metric | Value | Significance |
|--------|-------|-------------|
| **Human Effort** | 40% | Minimal supervision required |
| **System Coverage** | 80% | High automation achievement |
| **Cross-Domain Transfer** | ✅ | Reasoning > semantic content |
| **Cluster Separation** | 5 distinct | Clear pattern differentiation |
| **Propagation Accuracy** | 100% | Perfect transfer when possible |

## 🧪 Core Technologies

### Backend (Python)
- **FastAPI**: High-performance async API
- **HDBSCAN**: Density-based clustering
- **OpenAI**: State-of-the-art embeddings
- **Pinecone**: Vector similarity search
- **Pydantic**: Type-safe data validation
- **Jupyter**: Interactive analysis

### Frontend (JavaScript)
- **Next.js 15**: React framework
- **React 19**: Modern UI components
- **Pure CSS**: No inline styling (user preference)
- **Responsive Design**: Mobile-friendly interface

## 🔧 Development Commands

```bash
# Full development setup
npm run setup              # Install all dependencies
npm run dev:full          # Start both backend + frontend

# Individual services
npm run dev:backend       # Python FastAPI server
npm run dev:frontend      # Next.js development server

# Analysis & Research
npm run notebook          # Launch Jupyter notebooks
npm run test:backend      # Run Python tests

# Maintenance
npm run clean            # Clean build artifacts
npm run lint:frontend    # Code quality checks
```

## 📈 Research Applications

### Immediate Applications
- **Content Moderation**: Automated unreliable response detection
- **Training Data Curation**: Identify problematic reasoning patterns
- **Quality Assurance**: Real-time hallucination monitoring

### Research Extensions
- **Multi-Modal Reasoning**: Extend to vision + language
- **Domain Adaptation**: Transfer patterns across fields
- **Reasoning Taxonomy**: Comprehensive pattern classification
- **Causal Analysis**: Why certain patterns indicate hallucinations

## 🎓 Academic Contributions

### Novel Insights
1. **Reasoning Structure Primacy**: Logic patterns > semantic similarity
2. **Minimal Supervision Scaling**: 40% effort → 80% coverage
3. **Cross-Domain Transferability**: Patterns generalize across topics
4. **Clustering-Based Propagation**: Effective alternative to traditional methods

### Methodological Advances
- Pure logical notation for pattern isolation
- Confidence-bounded label propagation
- Multi-cluster representative selection
- Uncertainty handling for outliers

## 📊 Experimental Validation

### Dataset Design
- **5 unique Q&A pairs** (automotive domain)
- **3 reasoning patterns each** (15 total CoTs)
- **Pure mathematical notation** (no domain vocabulary)
- **Known ground truth** for validation

### Results Summary
```
Total Q&A Pairs: 5
Human Labeled: 2 (40%)
System Propagated: 2 (40%)
Correct Predictions: 4/5 (80%)
Unpropagated: 1 (20% - appropriate uncertainty)
```

## 🤝 Contributing

### Research Contributions
1. **New Datasets**: Add domain-specific reasoning examples
2. **Advanced Clustering**: Implement alternative algorithms
3. **Evaluation Metrics**: Develop better assessment methods
4. **Visualization**: Create interactive analysis tools

### Technical Contributions
1. **Performance Optimization**: Scale to larger datasets
2. **Model Integration**: Support additional embedding models
3. **Real-time Processing**: Stream-based clustering
4. **Deployment**: Production-ready containerization

## 📚 Documentation

- **Backend API**: http://localhost:8000/docs (FastAPI auto-docs)
- **Research Notebooks**: `backend/notebooks/experiment_analysis.ipynb`
- **Frontend Components**: Documented React components
- **Python Research**: `python_version/README.md`

## 🏆 Research Impact

This work demonstrates that **reasoning pattern clustering** enables:

1. **Scalable Hallucination Detection** with minimal human supervision
2. **Cross-Domain Knowledge Transfer** based on logical structure
3. **Efficient Content Moderation** for AI-generated responses
4. **Research Methodology** for studying reasoning patterns

The approach shows promise for **scaling AI safety** through automated detection of unreliable reasoning, particularly valuable as language models become more prevalent in critical applications.

---

**Research Team**: [Your Institution]  
**Contact**: [Your Email]  
**License**: MIT (Academic/Research Use)
