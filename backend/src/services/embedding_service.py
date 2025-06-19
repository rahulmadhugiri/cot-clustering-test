import openai
import numpy as np
from typing import List, Optional
import logging
from pinecone import Pinecone
import os
from dotenv import load_dotenv

from ..models.schemas import CoTExample

load_dotenv()
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "cot-clustering-test")
        
        # Try to connect to existing index
        try:
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to existing Pinecone index: {self.index_name}")
        except Exception as e:
            logger.warning(f"Could not connect to Pinecone index: {e}")
            self.index = None
    
    async def generate_embeddings(self, cot_examples: List[CoTExample]) -> List[CoTExample]:
        """Generate embeddings for CoT examples using OpenAI"""
        logger.info(f"Generating embeddings for {len(cot_examples)} CoT examples")
        
        embedded_examples = []
        
        for cot in cot_examples:
            try:
                # Create embedding for the CoT reasoning
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=cot.cot,
                    dimensions=1024  # Match Pinecone index dimension
                )
                
                embedding = response.data[0].embedding
                
                # Create updated CoT with embedding
                updated_cot = cot.model_copy()
                updated_cot.embedding = embedding
                embedded_examples.append(updated_cot)
                
                # Store in Pinecone if available
                if self.index:
                    metadata = {
                        "question": cot.question,
                        "answer": cot.answer,
                        "cot": cot.cot
                    }
                    # Only add reasoning_pattern if it's not None
                    if cot.reasoning_pattern:
                        metadata["reasoning_pattern"] = cot.reasoning_pattern.value if hasattr(cot.reasoning_pattern, 'value') else str(cot.reasoning_pattern)
                    
                    self.index.upsert(
                        vectors=[(
                            cot.id,
                            embedding,
                            metadata
                        )]
                    )
                
            except Exception as e:
                logger.error(f"Error generating embedding for CoT {cot.id}: {e}")
                # Add without embedding for now - but this will cause clustering to fail
                # Instead, let's skip this CoT or raise the error
                raise Exception(f"Failed to generate embedding for CoT {cot.id}: {e}")
        
        logger.info(f"Generated embeddings for {len(embedded_examples)} CoT examples")
        return embedded_examples
    
    async def fetch_embeddings(self) -> List[CoTExample]:
        """Fetch all embeddings from Pinecone"""
        if not self.index:
            logger.warning("No Pinecone index available")
            return []
        
        try:
            # Query all vectors (this is a simplified approach)
            # In production, you'd want pagination for large datasets
            response = self.index.query(
                vector=[0.0] * 1024,  # Dummy vector matching index dimension
                top_k=10000,  # Large number to get all
                include_values=True,
                include_metadata=True
            )
            
            cot_examples = []
            for match in response.matches:
                metadata = match.metadata
                cot_example = CoTExample(
                    id=match.id,
                    question=metadata.get("question", ""),
                    answer=metadata.get("answer", ""),
                    cot=metadata.get("cot", ""),
                    reasoning_pattern=metadata.get("reasoning_pattern"),
                    embedding=match.values
                )
                cot_examples.append(cot_example)
            
            logger.info(f"Fetched {len(cot_examples)} CoT examples from Pinecone")
            return cot_examples
            
        except Exception as e:
            logger.error(f"Error fetching embeddings from Pinecone: {e}")
            return []
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def search_similar_cots(
        self, 
        query_cot: CoTExample, 
        top_k: int = 5
    ) -> List[CoTExample]:
        """Find similar CoT examples using vector search"""
        if not self.index or not hasattr(query_cot, 'embedding') or not query_cot.embedding:
            return []
        
        try:
            response = self.index.query(
                vector=query_cot.embedding,
                top_k=top_k,
                include_values=True,
                include_metadata=True
            )
            
            similar_cots = []
            for match in response.matches:
                if match.id != query_cot.id:  # Exclude self
                    metadata = match.metadata
                    similar_cot = CoTExample(
                        id=match.id,
                        question=metadata.get("question", ""),
                        answer=metadata.get("answer", ""),
                        cot=metadata.get("cot", ""),
                        reasoning_pattern=metadata.get("reasoning_pattern"),
                        embedding=match.values
                    )
                    similar_cots.append(similar_cot)
            
            return similar_cots
            
        except Exception as e:
            logger.error(f"Error searching for similar CoTs: {e}")
            return [] 