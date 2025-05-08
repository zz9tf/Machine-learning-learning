import os
import numpy as np
import torch
import pandas as pd
from gnn import GraphDataset, GNN, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
from torch.utils.data import random_split

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UserWarning)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            
            all_preds.extend(pred)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Handle potential warnings using zero_division=0
    try:
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        precision = recall = f1 = 0.0
    
    # Generate confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
    except:
        cm = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def main():
    print("Starting GNN training and prediction...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.csv')
    
    print(f"Loading datasets from {data_dir}")
    
    # Load datasets
    train_dataset = GraphDataset(root=data_dir, mode='train')
    test_dataset = GraphDataset(root=data_dir, mode='test')
    
    # Split train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    print(f"Total training dataset size: {len(train_dataset)} graphs")
    print(f"Training subset size: {len(train_subset)} graphs")
    print(f"Validation subset size: {len(val_subset)} graphs")
    print(f"Test dataset size: {len(test_dataset)} graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Determine number of classes
    num_classes = max(train_dataset.labels_map.values()) + 1
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = GNN(in_channels=1, hidden_channels=32, out_channels=num_classes).to(device)
    
    # Learning rate settings
    initial_lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Learning rate decay settings
    max_epochs = 150
    lr_decay_epochs = [150, 175]  # epochs at which to reduce learning rate
    lr_decay_factor = 0.1  # multiply lr by this factor at each step
    
    # For tracking metrics
    train_losses = []
    val_metrics = []
    current_lr = initial_lr
    
    # Early stopping settings
    patience = 20
    best_val_f1 = 0
    best_model_state = None
    counter = 0
    
    # Training loop
    print("Starting training...")
    print(f"Initial learning rate: {initial_lr}")
    print(f"Learning rate will decrease by factor of {lr_decay_factor} at epochs: {lr_decay_epochs}")
    
    for epoch in range(max_epochs):
        # Check if we should decrease learning rate
        if epoch in lr_decay_epochs:
            current_lr *= lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Reducing learning rate to {current_lr}")
        
        # Training phase
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        train_loss = total_loss / len(train_subset)
        train_losses.append(train_loss)
        
        # Training evaluation
        metrics = evaluate_model(model, train_loader, device)
        
        # Validation evaluation
        v_metrics = evaluate_model(model, val_loader, device)
        val_metrics.append(v_metrics)
        
        # Print metrics every 10 epochs
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f'Epoch: {epoch:03d}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {metrics["accuracy"]:.4f}, Train F1: {metrics["f1"]:.4f}, '
                  f'Val Acc: {v_metrics["accuracy"]:.4f}, Val F1: {v_metrics["f1"]:.4f}')
        
        # # Early stopping check
        # if metrics['f1'] > best_val_f1:
        #     best_val_f1 = metrics['f1']
        #     best_model_state = model.state_dict().copy()
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break
    
    # Load best model for final evaluation and prediction
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation F1: {best_val_f1:.4f}")
    
    # Final evaluation on validation set
    print("\nFinal Evaluation on Validation Set:")
    final_val_metrics = evaluate_model(model, val_loader, device)
    print(f"Validation Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {final_val_metrics['precision']:.4f}")
    print(f"Validation Recall: {final_val_metrics['recall']:.4f}")
    print(f"Validation F1 Score: {final_val_metrics['f1']:.4f}")
    
    if final_val_metrics['confusion_matrix'] is not None:
        print("\nValidation Confusion Matrix:")
        print(final_val_metrics['confusion_matrix'])
    
    # Make predictions on test set (no evaluation since we don't have labels)
    print("\nGenerating predictions on test set...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).cpu().numpy()
            
            for i in range(len(data.graph_id)):
                predictions.append((int(data.graph_id[i].item()), int(pred[i].item())))
    
    # Save predictions
    result_df = pd.DataFrame(predictions, columns=['graph_id', 'graph_label'])
    result_df.to_csv(output_path, index=False, header=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()