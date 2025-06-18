# CoT Embedding and Pinecone Upload

This project includes scripts to embed Chain-of-Thought (CoT) reasoning and upload them to Pinecone for similarity search.

## Setup

### 1. Environment Variables

Create a `.env` file in the root directory with your API keys:

```bash
cp env.example .env
```

Then edit `.env` and add your actual API keys:

```env
# OpenAI API Key
OPENAI_API_KEY=your_actual_openai_api_key

# Pinecone API Key  
PINECONE_API_KEY=your_actual_pinecone_api_key

# Pinecone Index Name - Set this to your specific index
PINECONE_INDEX_NAME=cot-clustering-test
```

### 2. Pinecone Index Setup

Make sure you have a Pinecone index created with:
- **Dimension**: 1024 (using text-embedding-3-small with dimensions=1024)
- **Metric**: cosine (recommended for text embeddings)
- **Index Name**: `cot-clustering-test` (or whatever you named your index)

**Your Index Settings:**
- Name: `cot-clustering-test`
- Dimensions: 1024 ✅
- Metric: cosine ✅

## Usage

### Test Connections

Before running the embedding script, test your API connections:

```bash
npm run test-connection
```

This will verify:
- ✅ OpenAI API key works and can create embeddings
- ✅ Pinecone API key works and can access your index
- 📊 Show current index stats (vector count, dimensions)

### Run Embedding Process

Once connections are verified, run the embedding script:

```bash
npm run embed
```

This script will:
1. **Load CoTs**: Read all 15 CoTs from `data/cots.js`
2. **Create Embeddings**: Use OpenAI's `text-embedding-3-small` model
3. **Upload to Pinecone**: Store vectors with metadata
4. **Process in Batches**: Handle rate limits gracefully

## Data Structure

### CoT Data Format
Each CoT is structured as:
```javascript
{
  id: 'cot-1',
  question: 'What is the recommended viscosity?',
  answer: 'The recommended viscosity and quality grades...',
  cot: 'The question is asking for the recommended viscosity...'
}
```

### Pinecone Vector Format
Each vector uploaded to Pinecone contains:
```javascript
{
  id: 'cot-1',
  values: [0.1, -0.2, 0.3, ...], // 1024-dimensional embedding
  metadata: {
    question: '...',
    answer: '...',
    cot: '...'
  }
}
```

## What Gets Embedded

**Only the CoT text** is embedded - not the question or answer. This allows you to:
- Find similar reasoning patterns
- Cluster CoTs by reasoning approach
- Search for specific types of logical thinking

The question and answer are stored as metadata for context and debugging.

## Files

- `data/cots.js` - All 15 CoTs structured for processing
- `scripts/embed-and-upload.js` - Main embedding and upload script
- `scripts/test-connection.js` - Connection testing utility
- `env.example` - Environment variables template

## Troubleshooting

### Rate Limits
The script includes delays to respect OpenAI rate limits:
- 100ms between individual embeddings
- 1 second between batches
- Processes 5 CoTs per batch

### Common Issues
1. **"Invalid API Key"** - Check your `.env` file
2. **"Index not found"** - Verify index name and that it exists
3. **"Dimension mismatch"** - Ensure index dimension is 1024
4. **Rate limit errors** - The script handles these automatically with retries

## Next Steps

After uploading, you can:
1. Query similar CoTs using vector similarity
2. Cluster CoTs to find reasoning patterns  
3. Build a reasoning recommendation system
4. Analyze different types of logical approaches 