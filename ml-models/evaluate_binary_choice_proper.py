import torch
import numpy as np
from binary_choice_classifier import BinaryChoiceClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def evaluate_binary_choice_model_proper():
    print("🔍 PROPER EVALUATION - ONLY ON TEST SET")
    print("=" * 60)
    
    # Load cached dual embeddings
    try:
        cache = np.load('dual_embeddings_cache.npz')
        pos_embeddings = cache['pos_embeddings']
        neg_embeddings = cache['neg_embeddings']
        labels = cache['labels']
        print(f"✅ Loaded dual embeddings cache")
    except FileNotFoundError:
        pos_embeddings, neg_embeddings, labels = None, None, None
    
    if pos_embeddings is None:
        print("❌ No cached dual embeddings found. Run the main classifier first.")
        return
    
    # CRITICAL: Use the SAME split as training
    pos_train, pos_test, neg_train, neg_test, y_train, y_test, train_indices, test_indices = train_test_split(
        pos_embeddings, neg_embeddings, labels, range(len(labels)), 
        test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Data Split:")
    print(f"Total samples: {len(labels)}")
    print(f"Training samples: {len(y_train)} (indices: {train_indices})")
    print(f"Test samples: {len(y_test)} (indices: {test_indices})")
    print(f"Test label distribution: {np.bincount(y_test)} (0=Not Hallucinated, 1=Hallucinated)")
    
    # Load the trained model
    embedding_dim = pos_embeddings.shape[1]
    model = BinaryChoiceClassifier(embedding_dim=embedding_dim)
    
    try:
        model.load_state_dict(torch.load('best_binary_choice_model.pth'))
        print("✅ Loaded trained binary choice model successfully")
    except FileNotFoundError:
        print("❌ No trained binary choice model found. Train the model first.")
        return
    
    # Evaluate ONLY on test data (unseen during training)
    model.eval()
    with torch.no_grad():
        pos_test_tensor = torch.FloatTensor(pos_test)
        neg_test_tensor = torch.FloatTensor(neg_test)
        
        outputs = model(pos_test_tensor, neg_test_tensor)
        binary_probabilities = outputs['binary_output'].squeeze().numpy()
        choice_probabilities = torch.softmax(outputs['choice_logits'], dim=1).numpy()
        
        binary_predictions = (binary_probabilities > 0.5).astype(int)
        choice_predictions = np.argmax(choice_probabilities, axis=1)
    
    # Calculate accuracies on PROPER test set
    binary_accuracy = (binary_predictions == y_test).mean()
    choice_accuracy = (choice_predictions == y_test).mean()
    
    print(f"\n🎯 PROPER TEST SET PERFORMANCE")
    print(f"=" * 50)
    print(f"Binary Classification Accuracy: {binary_accuracy:.3f} ({binary_accuracy*100:.1f}%)")
    print(f"Choice Mechanism Accuracy: {choice_accuracy:.3f} ({choice_accuracy*100:.1f}%)")
    print(f"Correct predictions: {int(binary_accuracy * len(y_test))}/{len(y_test)}")
    
    # Show per-class performance
    correct_not_hallucinated = ((binary_predictions == 0) & (y_test == 0)).sum()
    total_not_hallucinated = (y_test == 0).sum()
    correct_hallucinated = ((binary_predictions == 1) & (y_test == 1)).sum()
    total_hallucinated = (y_test == 1).sum()
    
    print(f"\nPer-Class Performance (Binary):")
    print(f"Not Hallucinated: {correct_not_hallucinated}/{total_not_hallucinated} ({correct_not_hallucinated/total_not_hallucinated*100:.1f}%)")
    print(f"Hallucinated: {correct_hallucinated}/{total_hallucinated} ({correct_hallucinated/total_hallucinated*100:.1f}%)")
    
    # Show detailed predictions for each test sample
    print(f"\n🔍 Detailed Test Set Predictions:")
    print(f"=" * 80)
    
    # Load Q&A data
    import pandas as pd
    df = pd.read_csv('../data/cleaned_questions_answers.csv')
    qa_data = []
    for i, row in df.iterrows():
        if i < 30:
            qa_data.append({
                'question': row.iloc[0] if len(row) > 0 else '',
                'answer': row.iloc[1] if len(row) > 1 else ''
            })
    
    for i, test_idx in enumerate(test_indices):
        binary_pred = binary_predictions[i]
        choice_pred = choice_predictions[i]
        binary_prob = binary_probabilities[i]
        choice_probs = choice_probabilities[i]
        true_label = y_test[i]
        
        binary_status = "✅ CORRECT" if binary_pred == true_label else "❌ WRONG"
        choice_status = "✅ CORRECT" if choice_pred == true_label else "❌ WRONG"
        
        binary_label = "Hallucinated" if binary_pred == 1 else "Not Hallucinated"
        choice_label = "Positive CoT" if choice_pred == 1 else "Negative CoT"
        true_label_str = "Hallucinated" if true_label == 1 else "Not Hallucinated"
        
        qa = qa_data[test_idx]
        
        print(f"Test Sample {i+1} (QA {test_idx}):")
        print(f"  Question: {qa['question'][:60]}...")
        print(f"  Binary: {binary_label} ({binary_prob:.3f}) {binary_status}")
        print(f"  Choice: {choice_label} (pos: {choice_probs[1]:.3f}, neg: {choice_probs[0]:.3f}) {choice_status}")
        print(f"  True: {true_label_str}")
        print()
    
    # Compare with previous inflated results
    print(f"\n📊 COMPARISON:")
    print(f"=" * 50)
    print(f"❌ Previous (WRONG - tested on training data): 96.7% (29/30)")
    print(f"✅ Correct (ONLY test data): {binary_accuracy:.3f} ({binary_accuracy*100:.1f}%) ({int(binary_accuracy * len(y_test))}/{len(y_test)})")
    print(f"")
    print(f"🧠 This is much more realistic for a model trained on only {len(y_train)} samples!")
    
    return binary_accuracy, choice_accuracy


if __name__ == "__main__":
    evaluate_binary_choice_model_proper() 