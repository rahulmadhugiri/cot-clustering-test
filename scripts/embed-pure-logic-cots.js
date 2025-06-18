import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { pureLogicCotsData } from '../data/pure-logic-cots.js';

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
  console.log(`Starting to process ${pureLogicCotsData.length} pure logic CoTs...`);
  
  const vectors = [];
  
  for (let i = 0; i < pureLogicCotsData.length; i += BATCH_SIZE) {
    const batch = pureLogicCotsData.slice(i, i + BATCH_SIZE);
    console.log(`Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(pureLogicCotsData.length / BATCH_SIZE)}`);
    
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
    
    if (i + BATCH_SIZE < pureLogicCotsData.length) {
      console.log('Waiting before next batch...');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  console.log('🎉 All pure logic CoTs have been embedded and uploaded to Pinecone!');
  console.log('');
  console.log('🧪 ULTIMATE EXPERIMENT: Pure Logical Structures');
  console.log('This dataset uses ONLY mathematical and logical notation');
  console.log('with zero domain-specific content to test if embeddings');
  console.log('can cluster by pure reasoning structure.');
  console.log('');
  console.log('📊 Expected clustering patterns BY REASONING TYPE:');
  console.log('- Deductive: cot-1, cot-2, cot-3 (Given P1, P2 → Therefore C)');
  console.log('- Experience: cot-4, cot-5, cot-6 (Pattern analysis, frequency data)');
  console.log('- Systems: cot-7, cot-8, cot-9 (Network topology, state vectors)');
  console.log('- Procedural: cot-10, cot-11, cot-12 (Step 1, Step 2, Algorithm)');
  console.log('- Analogical: cot-13, cot-14, cot-15 (A:B :: C:D, mapping functions)');
  console.log('');
  console.log('🔬 THIS IS THE ULTIMATE TEST:');
  console.log('If clustering STILL groups by question topic despite using');
  console.log('pure mathematical notation, it proves that embedding models');
  console.log('are fundamentally biased toward semantic content over structure.');
  console.log('');
  console.log('🚀 If clustering finally groups by reasoning pattern,');
  console.log('it shows that logical structure CAN be detected when');
  console.log('domain semantics are completely eliminated!');
}

async function main() {
  try {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY environment variable is required');
    }
    
    if (!process.env.PINECONE_API_KEY) {
      throw new Error('PINECONE_API_KEY environment variable is required');
    }
    
    console.log('🧪 PURE LOGIC REASONING EXPERIMENT');
    console.log('==================================');
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