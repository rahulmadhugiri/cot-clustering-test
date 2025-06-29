import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class AdvancedBinaryChoiceClassifier(nn.Module):
    """Advanced Binary Choice Classifier with attention and sophisticated training"""
    def __init__(self, embedding_dim, hidden_dims=[768, 384, 192, 96], dropout_rate=0.3):
        super(AdvancedBinaryChoiceClassifier, self).__init__()
        
        # Separate processors for positive and negative embeddings
        self.pos_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.Dropout(dropout_rate)
        )
        
        self.neg_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0] // 2,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Main classifier
        layers = []
        input_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.extend([nn.Linear(input_dim, 1), nn.Sigmoid()])
        self.classifier = nn.Sequential(*layers)
        
        # Choice network
        self.choice_network = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 2)
        )
    
    def forward(self, pos_embeddings, neg_embeddings):
        # Process embeddings
        pos_features = self.pos_processor(pos_embeddings)
        neg_features = self.neg_processor(neg_embeddings)
        
        # Apply attention
        combined_for_attention = torch.stack([pos_features, neg_features], dim=1)
        attended_features, _ = self.attention(
            combined_for_attention, combined_for_attention, combined_for_attention
        )
        pos_features = attended_features[:, 0, :]
        neg_features = attended_features[:, 1, :]
        
        # Combine features
        combined_features = torch.cat([pos_features, neg_features], dim=1)
        
        # Get outputs
        binary_output = self.classifier(combined_features)
        choice_logits = self.choice_network(combined_features)
        
        return {
            'binary_output': binary_output,
            'choice_logits': choice_logits
        } 