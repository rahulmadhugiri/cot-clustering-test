import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { abstractCotsData } from '../data/abstract-cots.js';

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
const BATCH_SIZE = 5;

async function createEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: text,
      dimensions: 1024,
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
    console.log('Clearing existing vectors from Pinecone...');
    await index.deleteAll();
    console.log('✅ Existing vectors cleared');
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
  console.log(`Starting to process ${abstractCotsData.length} abstract CoTs...`);
  
  const vectors = [];
  
  for (let i = 0; i < abstractCotsData.length; i += BATCH_SIZE) {
    const batch = abstractCotsData.slice(i, i + BATCH_SIZE);
    console.log(`Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(abstractCotsData.length / BATCH_SIZE)}`);
    
    for (const cot of batch) {
      try {
        console.log(`Creating embedding for ${cot.id}...`);
        
        const embedding = await createEmbedding(cot.cot);
        
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
        
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        console.error(`❌ Error processing ${cot.id}:`, error);
        throw error;
      }
    }
    
    if (vectors.length > 0) {
      await uploadToPinecone(vectors);
      vectors.length = 0;
    }
    
    if (i + BATCH_SIZE < abstractCotsData.length) {
      console.log('Waiting before next batch...');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  console.log('🎉 All abstract CoTs have been embedded and uploaded to Pinecone!');
  console.log('');
  console.log('🧪 EXPERIMENT: Abstract Reasoning Patterns');
  console.log('This dataset uses domain-neutral language (System X, Property Y, etc.)');
  console.log('to test if HDBSCAN can cluster by reasoning structure rather than content.');
  console.log('');
  console.log('📊 Expected clustering patterns BY REASONING TYPE:');
  console.log('- Deductive reasoning: cot-1, cot-2, cot-3 (Given X, therefore Y)');
  console.log('- Experience-based: cot-4, cot-5, cot-6 (From experience, I learned...)');
  console.log('- Systems thinking: cot-7, cot-8, cot-9 (Interconnected networks...)');
  console.log('- Step-by-step: cot-10, cot-11, cot-12 (Step 1, Step 2...)');
  console.log('- Analogical: cot-13, cot-14, cot-15 (Like X, similar to Y...)');
  console.log('');
  console.log('🔬 If clustering still groups by question topic, it suggests');
  console.log('that even abstract domain language has stronger embedding signal');
  console.log('than reasoning structure patterns.');
}

async function main() {
  try {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY environment variable is required');
    }
    
    if (!process.env.PINECONE_API_KEY) {
      throw new Error('PINECONE_API_KEY environment variable is required');
    }
    
    console.log('🧪 ABSTRACT REASONING EXPERIMENT');
    console.log('================================');
    console.log('Environment variables check passed ✅');
    console.log(`Using embedding model: ${EMBEDDING_MODEL}`);
    console.log(`Pinecone index: ${PINECONE_INDEX_NAME}`);
    console.log('');
    
    await clearExistingVectors();
    await processCoTs();
    
  } catch (error) {
    console.error('❌ Script failed:', error);
    process.exit(1);
  }
}

main(); 