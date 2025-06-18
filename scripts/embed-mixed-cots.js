import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { mixedCotsData } from '../data/mixed-cots.js';

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

// Configuration
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'cot-clustering-test';
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

async function clearExistingVectors() {
  try {
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    
    // Delete all existing vectors
    console.log('Clearing existing vectors from Pinecone...');
    await index.deleteAll();
    console.log('✅ Existing vectors cleared');
    
    // Wait a moment for the deletion to propagate
    await new Promise(resolve => setTimeout(resolve, 2000));
  } catch (error) {
    console.error('Error clearing vectors:', error);
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
  console.log(`Starting to process ${mixedCotsData.length} mixed CoTs...`);
  
  const vectors = [];
  
  for (let i = 0; i < mixedCotsData.length; i += BATCH_SIZE) {
    const batch = mixedCotsData.slice(i, i + BATCH_SIZE);
    console.log(`Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(mixedCotsData.length / BATCH_SIZE)}`);
    
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
    if (i + BATCH_SIZE < mixedCotsData.length) {
      console.log('Waiting before next batch...');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  console.log('🎉 All mixed CoTs have been embedded and uploaded to Pinecone!');
  console.log('');
  console.log('📊 Expected clustering patterns:');
  console.log('- Cluster 1: Deductive reasoning (cot-1, cot-2, cot-3)');
  console.log('- Cluster 2: Experience-based reasoning (cot-4, cot-5, cot-6)');
  console.log('- Cluster 3: Systems thinking (cot-7, cot-8, cot-9)');
  console.log('- Cluster 4: Procedural/step-by-step (cot-10, cot-11, cot-12)');
  console.log('- Cluster 5: Analogical reasoning (cot-13, cot-14, cot-15)');
  console.log('');
  console.log('Now visit /clusters to see if HDBSCAN discovers these reasoning patterns!');
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
    console.log('');
    
    // Clear existing vectors first
    await clearExistingVectors();
    
    // Process new mixed CoTs
    await processCoTs();
    
  } catch (error) {
    console.error('❌ Script failed:', error);
    process.exit(1);
  }
}

// Run the script
main(); 