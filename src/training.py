import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import numpy as np
import time

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10, scheduler=None, use_wandb=False, device='cuda'):
    """
    Train a given model with training and validation data loaders.
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        epochs: Number of training epochs.
        scheduler: Learning rate scheduler (optional).
        use_wandb: Whether to log metrics to Weights & Biases.
        device: Device to run the training on ('cuda' or 'cpu').
    Returns:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        val_accuracies: List of validation accuracies per epoch.
        total_time: Total training time in seconds.
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for rgb, lidar, labels in train_loader:
            rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device).long().squeeze()
            
            optimizer.zero_grad()
            outputs = model(rgb, lidar)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if scheduler:
            scheduler.step()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for rgb, lidar, labels in val_loader:
                rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device).long().squeeze()
                outputs = model(rgb, lidar)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc
            })
            
    total_time = time.time() - start_time
    return train_losses, val_losses, val_accuracies, total_time

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test dataset and compute accuracy, precision, recall, and F1-score.
    Args:
        model: The trained model.
        test_loader: DataLoader for test data.
        device: Device to run the evaluation on ('cuda' or 'cpu').
    Returns:
        acc: Accuracy of the model on the test set.
        precision: Precision of the model on the test set.
        recall: Recall of the model on the test set.
        f1: F1-score of the model on the test set.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for rgb, lidar, labels in test_loader:
            rgb, lidar, labels = rgb.to(device), lidar.to(device), labels.to(device).long().squeeze()
            outputs = model(rgb, lidar)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return acc, precision, recall, f1

def train_classifier(model, train_loader, optimizer, criterion, epochs=15, device='cuda'):
    """
    Train a LiDAR classifier model.
    Args:
        model: The classifier model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        epochs: Number of training epochs.
        device: Device to run the training on ('cuda' or 'cpu').
    Returns:
        model: The trained classifier model.
    """
    print("Training Classifier...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            # Handle different batch structures (rgb, lidar, label) or (data, label)
            if len(batch) == 3:
                _, data, label = batch # Assuming we train on lidar
            else:
                data, label = batch
                
            data, label = data.to(device), label.to(device).long().squeeze()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Acc: {100 * correct / total:.2f}%")
    return model

def train_cilp(model, train_loader, val_loader, optimizer, epochs=15, device='cuda', use_wandb=False, loss_img=None, loss_lidar=None):
    """
    Train a CILP model with contrastive loss on image and LiDAR embeddings.
    Args:
        model: The CILP model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        epochs: Number of training epochs.
        device: Device to run the training on ('cuda' or 'cpu').
        use_wandb: Whether to log metrics to Weights & Biases.
        loss_img: Loss function for image embeddings (default: CrossEntropyLoss).
        loss_lidar: Loss function for LiDAR embeddings (default: CrossEntropyLoss).
    Returns:
        train_losses: List of training losses per epoch.
        valid_losses: List of validation losses per epoch.
    """
    if loss_img is None: loss_img = nn.CrossEntropyLoss()
    if loss_lidar is None: loss_lidar = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    print("Starting CILP Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            rgb, lidar, _ = batch
            rgb, lidar = rgb.to(device), lidar.to(device)
            
            optimizer.zero_grad()
            logits_per_img, logits_per_lidar = model(rgb, lidar)
            
            ground_truth = torch.arange(len(rgb), dtype=torch.long).to(device)
            
            total_loss = (loss_img(logits_per_img, ground_truth) + loss_lidar(logits_per_lidar, ground_truth)) / 2
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb, lidar, _ = batch
                rgb, lidar = rgb.to(device), lidar.to(device)
                logits_per_img, logits_per_lidar = model(rgb, lidar)
                ground_truth = torch.arange(len(rgb), dtype=torch.long).to(device)
                loss = (loss_img(logits_per_img, ground_truth) + loss_lidar(logits_per_lidar, ground_truth)) / 2
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        valid_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss {avg_train_loss:.4f}, Valid Loss {avg_val_loss:.4f}")
        if use_wandb:
            wandb.log({
                "cilp_epoch": epoch + 1,
                "cilp_train_loss": avg_train_loss,
                "cilp_val_loss": avg_val_loss
            })
            
    return train_losses, valid_losses

def train_projector(projector, cilp_model, lidar_clf, train_loader, optimizer, criterion, epochs=15, device='cuda', use_wandb=False):
    """
    Train a projector model to map RGB embeddings to LiDAR embedding space.
    Args:
        projector: The projector model to train.
        cilp_model: The pre-trained CILP model for obtaining RGB embeddings.
        lidar_clf: The pre-trained LiDAR classifier for obtaining target LiDAR embeddings.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        epochs: Number of training epochs.
        device: Device to run the training on ('cuda' or 'cpu').
        use_wandb: Whether to log metrics to Weights & Biases.
    Returns:
        proj_losses: List of projector training losses per epoch.
    """
    proj_losses = []

    print("Training Projector...")
    for epoch in range(epochs):
        projector.train()
        epoch_loss = 0
        for batch in train_loader:
            rgb, lidar, _ = batch
            rgb, lidar = rgb.to(device), lidar.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Get RGB embeddings from CILP
                img_emb = cilp_model.img_embedder(rgb)
                
                # Get LiDAR embeddings from Classifier
                lidar_target = lidar_clf.get_embs(lidar)
            
            pred_lidar_emb = projector(img_emb)
            loss = criterion(pred_lidar_emb, lidar_target)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        proj_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Projector Loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({
                "projector_epoch": epoch + 1,
                "projector_loss": avg_loss
            })
            
    return proj_losses

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    Args:
        model: The model to count parameters for.
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
