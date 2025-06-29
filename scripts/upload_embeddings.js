const fs = require('fs');
const fetch = require('node-fetch');

async function uploadAllEmbeddings() {
  try {
    console.log('📖 Reading CSV file...');
    const csvContent = fs.readFileSync('public/cleaned_questions_answers.csv', 'utf8');
    const lines = csvContent.split('\n').slice(1).filter(line => line.trim());
    
    console.log(`Found ${lines.length} Q&A pairs to process`);
    
    // Parse CSV data
    const qaData = lines.map((line, i) => {
      const parts = line.split(',');
      const question = parts[0].replace(/"/g, '');
      const answer = parts.slice(1).join(',').replace(/"/g, '');
      return { id: `qa-${i}`, question, answer };
    });
    
    console.log('🧠 Generating CoTs and uploading to Pinecone...');
    
    for (let i = 0; i < qaData.length; i++) {
      const qa = qaData[i];
      console.log(`Processing ${i + 1}/${qaData.length}: ${qa.question.substring(0, 50)}...`);
      
      try {
        // Generate CoT
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
        
        // Upload to backend (which will generate embedding and store in Pinecone)
        const uploadResponse = await fetch('http://localhost:8000/api/v1/embed', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            id: qa.id,
            question: qa.question,
            answer: qa.answer,
            cot: cotData.content
          })
        });
        
        if (uploadResponse.ok) {
          console.log(`✅ Uploaded ${qa.id}`);
        } else {
          console.error(`❌ Failed to upload ${qa.id}`);
        }
        
        // Small delay to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        console.error(`❌ Error processing ${qa.id}:`, error.message);
      }
    }
    
    console.log('🎉 Upload complete!');
    
  } catch (error) {
    console.error('❌ Error:', error);
  }
}

uploadAllEmbeddings(); 