import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class AdvancedBinaryChoiceDataLoader:
    def __init__(self):
        self.pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'cot-clustering-test')
        self.index = self.pinecone.Index(self.index_name)
        
        # Ground truth labels
        self.ground_truth = {
            "0": "not_hallucinated", "1": "not_hallucinated", "2": "hallucinated",
            "3": "not_hallucinated", "4": "hallucinated", "5": "not_hallucinated",
            "6": "hallucinated", "7": "hallucinated", "8": "not_hallucinated",
            "9": "not_hallucinated", "10": "hallucinated", "11": "not_hallucinated",
            "12": "not_hallucinated", "13": "hallucinated", "14": "not_hallucinated",
            "15": "not_hallucinated", "16": "hallucinated", "17": "not_hallucinated",
            "18": "hallucinated", "19": "not_hallucinated", "20": "hallucinated",
            "21": "not_hallucinated", "22": "hallucinated", "23": "hallucinated",
            "24": "not_hallucinated", "25": "not_hallucinated", "26": "hallucinated",
            "27": "hallucinated", "28": "not_hallucinated", "29": "not_hallucinated"
        }
    
    def load_qa_data(self):
        """Load Q&A data from CSV"""
        df = pd.read_csv('../public/cleaned_questions_answers.csv')
        qa_data = []
        
        for i, row in df.iterrows():
            if i < 30:  # Only first 30 examples
                qa_data.append({
                    'id': i,
                    'question': row.iloc[0] if len(row) > 0 else '',
                    'answer': row.iloc[1] if len(row) > 1 else '',
                    'label': 1 if self.ground_truth[str(i)] == "hallucinated" else 0
                })
        
        return qa_data
    
    def load_dual_embeddings_cache(self, filename='dual_embeddings_cache.npz'):
        """Load dual embeddings from cache file"""
        try:
            cache = np.load(filename)
            pos_embeddings = cache['pos_embeddings']
            neg_embeddings = cache['neg_embeddings']
            labels = cache['labels']
            print(f"✅ Loaded dual embeddings cache from {filename}")
            return pos_embeddings, neg_embeddings, labels
        except FileNotFoundError:
            print(f"⚠️ Cache file {filename} not found")
            return None, None, None


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
            nn.Linear(hidden_dims[1], 2),
            nn.Softmax(dim=1)
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
        choice_scores = self.choice_network(combined_features)
        
        return binary_output, choice_scores


class AdvancedTrainer:
    def __init__(self, model, device='cpu', config=None):
        self.model = model.to(device)
        self.device = device
        
        # Default training configuration
        if config is None:
            config = {
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'scheduler_patience': 15,
                'scheduler_factor': 0.7,
                'binary_loss_weight': 1.0,
                'choice_loss_weight': 0.5,
                'confidence_loss_weight': 0.3,
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0,
                'use_focal_loss': True
            }
        
        self.config = config
        
        # Loss functions
        if config['use_focal_loss']:
            self.binary_criterion = FocalLoss(alpha=config['focal_loss_alpha'], gamma=config['focal_loss_gamma'])
        else:
            self.binary_criterion = nn.BCELoss()
        
        self.choice_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            patience=config['scheduler_patience'], 
            factor=config['scheduler_factor'],
            verbose=True
        )
        
        # Metrics tracking
        self.metrics = defaultdict(list)
    
    def train_epoch(self, pos_train, neg_train, y_train, batch_size=16):
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = (len(pos_train) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(pos_train))
            
            batch_pos = torch.FloatTensor(pos_train[start_idx:end_idx]).to(self.device)
            batch_neg = torch.FloatTensor(neg_train[start_idx:end_idx]).to(self.device)
            batch_y = torch.FloatTensor(y_train[start_idx:end_idx]).to(self.device)
            
            self.optimizer.zero_grad()
            
            binary_outputs, choice_scores = self.model(batch_pos, batch_neg)
            binary_outputs = binary_outputs.squeeze()
            
            # Calculate losses
            binary_loss = self.binary_criterion(binary_outputs, batch_y)
            choice_targets = batch_y.long()
            choice_loss = self.choice_criterion(choice_scores, choice_targets)
            
            total_loss_batch = binary_loss + 0.5 * choice_loss
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_metrics['total_loss'] += total_loss_batch.item()
            epoch_metrics['binary_loss'] += binary_loss.item()
            epoch_metrics['choice_loss'] += choice_loss.item()
            
            # Accuracy metrics
            binary_predicted = (binary_outputs > 0.5).float()
            choice_predicted = torch.argmax(choice_scores, dim=1)
            
            epoch_metrics['binary_accuracy'] += (binary_predicted == batch_y).sum().item() / len(batch_y)
            epoch_metrics['choice_accuracy'] += (choice_predicted == choice_targets).sum().item() / len(batch_y)
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate(self, pos_val, neg_val, y_val):
        self.model.eval()
        with torch.no_grad():
            pos_val_tensor = torch.FloatTensor(pos_val).to(self.device)
            neg_val_tensor = torch.FloatTensor(neg_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            binary_outputs, choice_scores = self.model(pos_val_tensor, neg_val_tensor)
            binary_outputs = binary_outputs.squeeze()
            
            # Calculate losses
            binary_loss = self.binary_criterion(binary_outputs, y_val_tensor)
            choice_targets = y_val_tensor.long()
            choice_loss = self.choice_criterion(choice_scores, choice_targets)
            
            total_loss_batch = binary_loss + 0.5 * choice_loss
            
            # Calculate accuracies
            binary_predicted = (binary_outputs > 0.5).float()
            choice_predicted = torch.argmax(choice_scores, dim=1)
            
            binary_accuracy = (binary_predicted == y_val_tensor).sum().item() / len(y_val)
            choice_accuracy = (choice_predicted == choice_targets).sum().item() / len(y_val)
            
            # AUC score
            try:
                auc_score = roc_auc_score(y_val, binary_outputs.cpu().numpy())
            except:
                auc_score = 0.5
            
            val_metrics = {
                'total_loss': total_loss_batch.item(),
                'binary_loss': binary_loss.item(),
                'choice_loss': choice_loss.item(),
                'binary_accuracy': binary_accuracy,
                'choice_accuracy': choice_accuracy,
                'auc_score': auc_score
            }
            
            return val_metrics, binary_outputs.cpu().numpy(), choice_scores.cpu().numpy()
    
    def train(self, pos_train, neg_train, y_train, pos_val, neg_val, y_val, epochs=300, patience=30):
        print(f"🚀 Starting advanced training for {epochs} epochs...")
        print(f"📊 Training set: {len(pos_train)} samples")
        print(f"📊 Validation set: {len(pos_val)} samples")
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(pos_train, neg_train, y_train)
            
            # Validation
            val_metrics, _, _ = self.validate(pos_val, neg_val, y_val)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Store metrics
            for key, value in train_metrics.items():
                self.metrics[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.metrics[f'val_{key}'].append(value)
            
            # Print progress
            if epoch % 20 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}: Train Loss={train_metrics['total_loss']:.4f}, "
                      f"Train Acc={train_metrics['binary_accuracy']:.3f}, "
                      f"Val Loss={val_metrics['total_loss']:.4f}, "
                      f"Val Acc={val_metrics['binary_accuracy']:.3f}, "
                      f"AUC={val_metrics['auc_score']:.3f}")
            
            # Early stopping
            if val_metrics['binary_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['binary_accuracy']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_advanced_binary_choice_model.pth')
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch} (patience: {patience})")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_advanced_binary_choice_model.pth'))
        print(f"✅ Training completed! Best validation accuracy: {best_val_accuracy:.3f}")
        
        return best_val_accuracy
    
    def plot_training_history(self):
        """Plot comprehensive training metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss plots
        axes[0, 0].plot(self.metrics['train_total_loss'], label='Train', color='blue')
        axes[0, 0].plot(self.metrics['val_total_loss'], label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.metrics['train_binary_loss'], label='Train Binary', color='blue')
        axes[0, 1].plot(self.metrics['val_binary_loss'], label='Val Binary', color='red')
        axes[0, 1].set_title('Binary Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(self.metrics['train_choice_loss'], label='Train Choice', color='blue')
        axes[0, 2].plot(self.metrics['val_choice_loss'], label='Val Choice', color='red')
        axes[0, 2].set_title('Choice Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Accuracy plots
        axes[1, 0].plot(self.metrics['train_binary_accuracy'], label='Train Binary', color='blue')
        axes[1, 0].plot(self.metrics['val_binary_accuracy'], label='Val Binary', color='red')
        axes[1, 0].set_title('Binary Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.metrics['train_choice_accuracy'], label='Train Choice', color='blue')
        axes[1, 1].plot(self.metrics['val_choice_accuracy'], label='Val Choice', color='red')
        axes[1, 1].set_title('Choice Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        axes[1, 2].plot(self.metrics['val_auc_score'], label='Validation AUC', color='green')
        axes[1, 2].set_title('AUC Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('advanced_binary_choice_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def hyperparameter_search(pos_embeddings, neg_embeddings, labels):
    """Perform hyperparameter search using cross-validation"""
    print("🔍 Performing hyperparameter search...")
    
    # Define hyperparameter grid
    param_grid = [
        {
            'hidden_dims': [[512, 256, 128], [768, 384, 192, 96], [1024, 512, 256, 128]],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.0005, 0.001, 0.002],
            'use_attention': [True, False]
        }
    ]
    
    best_score = 0
    best_params = None
    results = []
    
    for params in param_grid:
        for hidden_dims in params['hidden_dims']:
            for dropout_rate in params['dropout_rate']:
                for learning_rate in params['learning_rate']:
                    for use_attention in params['use_attention']:
                        
                        config = {
                            'hidden_dims': hidden_dims,
                            'dropout_rate': dropout_rate,
                            'use_batch_norm': True,
                            'use_residual': False,
                            'activation': 'relu',
                            'use_attention': use_attention
                        }
                        
                        train_config = {
                            'learning_rate': learning_rate,
                            'weight_decay': 1e-5,
                            'scheduler_patience': 10,
                            'scheduler_factor': 0.7,
                            'binary_loss_weight': 1.0,
                            'choice_loss_weight': 0.5,
                            'confidence_loss_weight': 0.3,
                            'use_focal_loss': True,
                            'focal_loss_alpha': 0.25,
                            'focal_loss_gamma': 2.0
                        }
                        
                        # Cross-validation
                        cv_scores = []
                        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        
                        for train_idx, val_idx in skf.split(pos_embeddings, labels):
                            pos_train, pos_val = pos_embeddings[train_idx], pos_embeddings[val_idx]
                            neg_train, neg_val = neg_embeddings[train_idx], neg_embeddings[val_idx]
                            y_train, y_val = labels[train_idx], labels[val_idx]
                            
                            # Create and train model
                            model = AdvancedBinaryChoiceClassifier(
                                embedding_dim=pos_embeddings.shape[1], 
                                config=config
                            )
                            trainer = AdvancedTrainer(model, config=train_config)
                            
                            # Quick training for hyperparameter search
                            trainer.train(pos_train, neg_train, y_train, pos_val, neg_val, y_val, 
                                        epochs=50, patience=10)
                            
                            # Evaluate
                            val_metrics, _, _ = trainer.validate(pos_val, neg_val, y_val)
                            cv_scores.append(val_metrics['binary_accuracy'])
                        
                        avg_score = np.mean(cv_scores)
                        results.append({
                            'config': config,
                            'train_config': train_config,
                            'cv_score': avg_score,
                            'cv_std': np.std(cv_scores)
                        })
                        
                        print(f"Config: {hidden_dims}, dropout={dropout_rate}, lr={learning_rate}, "
                              f"attention={use_attention} | CV Score: {avg_score:.3f} ± {np.std(cv_scores):.3f}")
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = (config, train_config)
    
    print(f"\n🏆 Best hyperparameters found with CV score: {best_score:.3f}")
    return best_params, results


def main():
    print("🧠 ADVANCED BINARY CHOICE HALLUCINATION CLASSIFIER")
    print("🎯 Sophisticated training with attention mechanism")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Initialize data loader
    data_loader = AdvancedBinaryChoiceDataLoader()
    
    # Load Q&A data
    qa_data = data_loader.load_qa_data()
    print(f"📊 Loaded {len(qa_data)} Q&A examples")
    
    # Load cached embeddings
    pos_embeddings, neg_embeddings, labels = data_loader.load_dual_embeddings_cache()
    
    if pos_embeddings is None:
        print("❌ No cached dual embeddings found. Run the basic binary choice classifier first.")
        return
    
    print(f"📐 Positive embeddings shape: {pos_embeddings.shape}")
    print(f"📐 Negative embeddings shape: {neg_embeddings.shape}")
    print(f"🏷️  Labels shape: {labels.shape}")
    print(f"📊 Label distribution: {np.bincount(labels)} (0=Not Hallucinated, 1=Hallucinated)")
    
    # Hyperparameter search (optional - uncomment to run)
    # best_params, hp_results = hyperparameter_search(pos_embeddings, neg_embeddings, labels)
    # best_config, best_train_config = best_params
    
    # Use optimized configuration
    best_config = {
        'hidden_dims': [768, 384, 192, 96],
        'dropout_rate': 0.3,
        'use_batch_norm': True,
        'use_residual': False,
        'activation': 'relu',
        'use_attention': True
    }
    
    best_train_config = {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'scheduler_patience': 20,
        'scheduler_factor': 0.7,
        'binary_loss_weight': 1.0,
        'choice_loss_weight': 0.5,
        'confidence_loss_weight': 0.3,
        'use_focal_loss': True,
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 2.0
    }
    
    # Split data
    pos_train, pos_test, neg_train, neg_test, y_train, y_test = train_test_split(
        pos_embeddings, neg_embeddings, labels, 
        test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n🔄 Data Split:")
    print(f"  Training: {len(pos_train)} samples")
    print(f"  Testing:  {len(pos_test)} samples")
    
    # Create model
    embedding_dim = pos_embeddings.shape[1]
    model = AdvancedBinaryChoiceClassifier(
        embedding_dim=embedding_dim, 
        config=best_config
    ).to(device)
    
    print(f"\n🏗️  Advanced Model Architecture:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Hidden layers: {' → '.join(map(str, best_config['hidden_dims']))}")
    print(f"  Attention mechanism: {best_config['use_attention']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, device, best_train_config)
    
    # Train model
    best_accuracy = trainer.train(
        pos_train, neg_train, y_train, 
        pos_test, neg_test, y_test, 
        epochs=400, patience=40
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Final evaluation
    val_metrics, binary_outputs, choice_scores = trainer.validate(pos_test, neg_test, y_test)
    
    print(f"\n🏆 FINAL ADVANCED RESULTS")
    print(f"=" * 50)
    print(f"🎯 Best Validation Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    print(f"🎯 Final Binary Accuracy: {val_metrics['binary_accuracy']:.3f} ({val_metrics['binary_accuracy']*100:.1f}%)")
    print(f"🎯 Final Choice Accuracy: {val_metrics['choice_accuracy']:.3f} ({val_metrics['choice_accuracy']*100:.1f}%)")
    print(f"🎯 Final AUC Score: {val_metrics['auc_score']:.3f}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_advanced_binary_choice_model.pth')
    print(f"💾 Model saved to: final_advanced_binary_choice_model.pth")


if __name__ == "__main__":
    main() 