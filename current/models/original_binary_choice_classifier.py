import torch
import torch.nn as nn
import numpy as np

class BinaryChoiceClassifier(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(BinaryChoiceClassifier, self).__init__()
        
        # Separate processors for positive and negative CoTs
        self.pos_processor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        self.neg_processor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Combined decision network
        self.combined_network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Explicit choice mechanism
        self.choice_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, pos_embeddings, neg_embeddings):
        # Process embeddings
        pos_features = self.pos_processor(pos_embeddings)
        neg_features = self.neg_processor(neg_embeddings)
        
        # Combine features
        combined = torch.cat([pos_features, neg_features], dim=1)
        
        # Get outputs
        binary_output = self.combined_network(combined)
        choice_logits = self.choice_layer(combined)
        
        return {
            'binary_output': binary_output,
            'choice_logits': choice_logits
        } 