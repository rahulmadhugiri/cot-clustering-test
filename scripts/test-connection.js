import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

async function testConnections() {
  console.log('Testing API connections...\n');
  
  // Test OpenAI
  try {
    console.log('🔍 Testing OpenAI connection...');
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: 'This is a test embedding.',
      dimensions: 1024, // Match your Pinecone index dimension
    });
    console.log('✅ OpenAI connection successful');
    console.log(`   Embedding dimension: ${response.data[0].embedding.length}`);
  } catch (error) {
    console.error('❌ OpenAI connection failed:', error.message);
    return;
  }
  
  // Test Pinecone
  try {
    console.log('\n🔍 Testing Pinecone connection...');
    const indexName = process.env.PINECONE_INDEX_NAME || 'cot-embeddings';
    const index = pinecone.Index(indexName);
    
    // Try to get index stats
    const stats = await index.describeIndexStats();
    console.log('✅ Pinecone connection successful');
    console.log(`   Index: ${indexName}`);
    console.log(`   Total vectors: ${stats.totalVectorCount || 0}`);
    console.log(`   Dimension: ${stats.dimension || 'Unknown'}`);
  } catch (error) {
    console.error('❌ Pinecone connection failed:', error.message);
    console.log('   Make sure your index exists and the name is correct');
    return;
  }
  
  console.log('\n🎉 All connections successful! Ready to run the embedding script.');
}

async function main() {
  try {
    // Check environment variables
    if (!process.env.OPENAI_API_KEY) {
      console.error('❌ OPENAI_API_KEY environment variable is required');
      process.exit(1);
    }
    
    if (!process.env.PINECONE_API_KEY) {
      console.error('❌ PINECONE_API_KEY environment variable is required');
      process.exit(1);
    }
    
    await testConnections();
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

main(); 