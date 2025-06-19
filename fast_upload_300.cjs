const fs = require('fs');
const fetch = require('node-fetch');

async function fastUpload300() {
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
    console.log('🧠 Generating CoTs with parallel processing...');
    
    // Process in batches of 10 for faster generation
    const batchSize = 10;
    const cotExamples = [];
    
    for (let i = 0; i < qaData.length; i += batchSize) {
      const batch = qaData.slice(i, i + batchSize);
      const batchNum = Math.floor(i / batchSize) + 1;
      const totalBatches = Math.ceil(qaData.length / batchSize);
      
      console.log(`[Batch ${batchNum}/${totalBatches}] Processing ${batch.length} Q&A pairs...`);
      
      // Process batch in parallel
      const batchPromises = batch.map(async (qa) => {
        try {
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
            return null;
          }
          
          const cotData = await cotResponse.json();
          
          return {
            id: qa.id,
            question: qa.question,
            answer: qa.answer,
            cot: cotData.content,
            reasoning_pattern: null
          };
          
        } catch (error) {
          console.error(`❌ Error generating CoT for ${qa.id}:`, error.message);
          return null;
        }
      });
      
      const batchResults = await Promise.all(batchPromises);
      const validResults = batchResults.filter(result => result !== null);
      cotExamples.push(...validResults);
      
      console.log(`✅ Completed batch ${batchNum}/${totalBatches} - Generated ${validResults.length}/${batch.length} CoTs`);
      
      // Small delay between batches to respect rate limits
      if (i + batchSize < qaData.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    console.log(`🎉 Generated ${cotExamples.length} CoTs total!`);
    
    // Save locally as backup immediately
    console.log('💾 Saving CoTs locally as backup...');
    fs.writeFileSync('data/all_300_cots.json', JSON.stringify(cotExamples, null, 2));
    console.log('✅ Saved to data/all_300_cots.json');
    
    console.log('📤 Uploading all embeddings to Pinecone via backend...');
    
    // Upload all CoTs to backend for embedding generation and Pinecone storage
    try {
      const uploadResponse = await fetch('http://localhost:8000/api/v1/embeddings/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cotExamples)
      });
      
      if (uploadResponse.ok) {
        const result = await uploadResponse.json();
        console.log(`🎉 Successfully uploaded ${result.count} embeddings to Pinecone!`);
        
        // Test clustering to verify everything works
        console.log('🧪 Testing clustering...');
        const clusterResponse = await fetch('http://localhost:8000/api/v1/cluster', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ min_cluster_size: 2, min_samples: 1 })
        });
        
        if (clusterResponse.ok) {
          const clusterResult = await clusterResponse.json();
          console.log(`✅ Clustering test successful! Found ${clusterResult.summary.num_clusters} clusters with ${clusterResult.summary.total_examples} examples.`);
        }
        
      } else {
        const error = await uploadResponse.text();
        console.error('❌ Failed to upload to backend:', error);
      }
      
    } catch (error) {
      console.error('❌ Error uploading to backend:', error.message);
    }
    
    console.log('\n🎉 All done! Your 300 Q&A pairs are now ready for clustering.');
    console.log('📍 Next steps:');
    console.log('   1. Go to http://localhost:3000/clusters to see clustering results');
    console.log('   2. Go to http://localhost:3000/propagation to test label propagation');
    
  } catch (error) {
    console.error('❌ Error:', error);
  }
}

fastUpload300(); 