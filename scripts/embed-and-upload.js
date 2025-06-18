import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { cotsData } from '../data/cots.js';

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

// Configuration
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'cot-embeddings';
const EMBEDDING_MODEL = 'text-embedding-3-small';
const BATCH_SIZE = 5; // Process in batches to avoid rate limits

async function createEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: text,
      dimensions: 1024, // Match your Pinecone index dimension
    });
    return response.data[0].embedding;
  } catch (error) {
    console.error('Error creating embedding:', error);
    throw error;
  }
}

async function uploadToPinecone(vectors) {
  try {
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    await index.upsert(vectors);
    console.log(`Successfully uploaded ${vectors.length} vectors to Pinecone`);
  } catch (error) {
    console.error('Error uploading to Pinecone:', error);
    throw error;
  }
}

async function processCoTs() {
  console.log(`Starting to process ${cotsData.length} CoTs...`);
  
  const vectors = [];
  
  for (let i = 0; i < cotsData.length; i += BATCH_SIZE) {
    const batch = cotsData.slice(i, i + BATCH_SIZE);
    console.log(`Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(cotsData.length / BATCH_SIZE)}`);
    
    for (const cot of batch) {
      try {
        console.log(`Creating embedding for ${cot.id}...`);
        
        // Create embedding for the CoT text only
        const embedding = await createEmbedding(cot.cot);
        
        // Prepare vector for Pinecone
        const vector = {
          id: cot.id,
          values: embedding,
          metadata: {
            question: cot.question,
            answer: cot.answer,
            cot: cot.cot,
          }
        };
        
        vectors.push(vector);
        console.log(`✅ Created embedding for ${cot.id}`);
        
        // Small delay to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        console.error(`❌ Error processing ${cot.id}:`, error);
        throw error;
      }
    }
    
    // Upload batch to Pinecone
    if (vectors.length > 0) {
      await uploadToPinecone(vectors);
      vectors.length = 0; // Clear the array
    }
    
    // Delay between batches
    if (i + BATCH_SIZE < cotsData.length) {
      console.log('Waiting before next batch...');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  console.log('🎉 All CoTs have been embedded and uploaded to Pinecone!');
}

async function main() {
  try {
    // Check environment variables
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY environment variable is required');
    }
    
    if (!process.env.PINECONE_API_KEY) {
      throw new Error('PINECONE_API_KEY environment variable is required');
    }
    
    console.log('Environment variables check passed ✅');
    console.log(`Using embedding model: ${EMBEDDING_MODEL}`);
    console.log(`Pinecone index: ${PINECONE_INDEX_NAME}`);
    
    await processCoTs();
    
  } catch (error) {
    console.error('❌ Script failed:', error);
    process.exit(1);
  }
}

// Run the script
main(); 