const fs = require('fs');
const fetch = require('node-fetch');

async function sequentialUpload300() {
  try {
    console.log('📖 Reading 300 Q&A pairs from CSV...');
    const csvContent = fs.readFileSync('public/cleaned_questions_answers.csv', 'utf8');
    const lines = csvContent.split('\n').slice(1).filter(line => line.trim());
    
    console.log(`Found ${lines.length} Q&A pairs to process`);
    
    // Parse CSV data
    const qaData = [];
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const parts = line.split(',');
      const question = parts[0].replace(/"/g, '').trim();
      const answer = parts.slice(1).join(',').replace(/"/g, '').trim();
      
      if (question && answer) {
        qaData.push({ 
          id: `cot-${i + 1}`, 
          question, 
          answer 
        });
      }
    }
    
    console.log(`Parsed ${qaData.length} valid Q&A pairs`);
    console.log('🧠 Generating CoTs sequentially (one at a time)...');
    
    const cotExamples = [];
    const embeddings = [];
    
    for (let i = 0; i < qaData.length; i++) {
      const qa = qaData[i];
      const progress = `${i + 1}/${qaData.length}`;
      
      console.log(`[${progress}] Processing: ${qa.question.substring(0, 60)}...`);
      
      try {
        // Step 1: Generate CoT
        const cotResponse = await fetch('http://localhost:3000/api/generate-cots', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: qa.question,
            answer: qa.answer
          })
        });
        
        if (!cotResponse.ok) {
          console.error(`❌ Failed to generate CoT for ${qa.id}`);
          continue;
        }
        
        const cotData = await cotResponse.json();
        
        const cotExample = {
          id: qa.id,
          question: qa.question,
          answer: qa.answer,
          cot: cotData.content,
          reasoning_pattern: null
        };
        
        cotExamples.push(cotExample);
        
        // Step 2: Generate embedding and upload to Pinecone
        try {
          const embeddingResponse = await fetch('http://localhost:8000/api/v1/embeddings/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify([cotExample])
          });
          
          if (embeddingResponse.ok) {
            // Step 3: Get the embedding vector for local storage
            try {
              const openaiResponse = await fetch('https://api.openai.com/v1/embeddings', {
                method: 'POST',
                headers: {
                  'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                  model: 'text-embedding-3-small',
                  input: cotData.content,
                  dimensions: 1024
                })
              });
              
              if (openaiResponse.ok) {
                const embeddingData = await openaiResponse.json();
                embeddings.push({
                  id: qa.id,
                  question: qa.question,
                  answer: qa.answer,
                  cot: cotData.content,
                  embedding: embeddingData.data[0].embedding
                });
                console.log(`✅ [${progress}] Generated CoT + embedding for ${qa.id}`);
              } else {
                console.log(`⚠️ [${progress}] CoT uploaded to Pinecone but failed to save embedding locally for ${qa.id}`);
              }
            } catch (embeddingError) {
              console.log(`⚠️ [${progress}] CoT uploaded to Pinecone but failed to save embedding locally for ${qa.id}`);
            }
          } else {
            console.error(`❌ Failed to upload ${qa.id} to Pinecone`);
          }
        } catch (uploadError) {
          console.error(`❌ Error uploading ${qa.id} to Pinecone:`, uploadError.message);
        }
        
        // Delay between requests to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 1000));
        
      } catch (error) {
        console.error(`❌ Error processing ${qa.id}:`, error.message);
      }
      
      // Save progress every 50 items
      if ((i + 1) % 50 === 0) {
        console.log(`💾 Saving progress... (${i + 1}/${qaData.length})`);
        fs.writeFileSync('data/cots_progress.json', JSON.stringify(cotExamples, null, 2));
        fs.writeFileSync('data/embeddings_progress.json', JSON.stringify(embeddings, null, 2));
      }
    }
    
    console.log(`🎉 Generated ${cotExamples.length} CoTs and ${embeddings.length} embeddings!`);
    
    // Save final results
    console.log('💾 Saving final results...');
    fs.writeFileSync('data/all_300_cots.json', JSON.stringify(cotExamples, null, 2));
    fs.writeFileSync('data/all_300_embeddings.json', JSON.stringify(embeddings, null, 2));
    console.log('✅ Saved CoTs to data/all_300_cots.json');
    console.log('✅ Saved embeddings to data/all_300_embeddings.json');
    
    // Test clustering to verify everything works
    console.log('🧪 Testing clustering...');
    try {
      const clusterResponse = await fetch('http://localhost:8000/api/v1/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_cluster_size: 2, min_samples: 1 })
      });
      
      if (clusterResponse.ok) {
        const clusterResult = await clusterResponse.json();
        console.log(`✅ Clustering test successful! Found ${clusterResult.summary.num_clusters} clusters with ${clusterResult.summary.total_examples} examples.`);
      } else {
        console.log('⚠️ Clustering test failed, but data is uploaded');
      }
    } catch (clusterError) {
      console.log('⚠️ Could not test clustering, but data is uploaded');
    }
    
    console.log('\n🎉 All done! Your 300 Q&A pairs are ready.');
    console.log('📍 Files created:');
    console.log('   - data/all_300_cots.json (CoTs)');
    console.log('   - data/all_300_embeddings.json (embeddings)');
    console.log('📍 Next steps:');
    console.log('   1. Go to http://localhost:3000/clusters to see clustering results');
    console.log('   2. Go to http://localhost:3000/propagation to test label propagation');
    
  } catch (error) {
    console.error('❌ Error:', error);
  }
}

// Load environment variables
require('dotenv').config();

sequentialUpload300(); 