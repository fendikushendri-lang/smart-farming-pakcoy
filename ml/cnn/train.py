import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
from ml.cnn.model import PakcoyCNN

def train_cnn():
    BASE_DIR = r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new"
    DATA_DIR = os.path.join(BASE_DIR, "data", "images")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Image Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = ImageFolder(root=DATA_DIR)
    classes = full_dataset.classes
    print("Detected classes:", classes)
    
    # Split 70% Train, 15% Val, 15% Test
    total = len(full_dataset)
    train_len = int(0.70 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len
    
    torch.manual_seed(42)
    train_subset, val_subset, test_subset = random_split(full_dataset, [train_len, val_len, test_len])
    
    # Custom Dataset Wrapper to apply distinct transforms
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return self.transform(x), y
        def __len__(self):
            return len(self.subset)
            
    train_data = TransformedDataset(train_subset, train_transform)
    val_data = TransformedDataset(val_subset, val_transform)
    test_data = TransformedDataset(test_subset, val_transform)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 2. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = PakcoyCNN(num_classes=len(classes)).to(device)
    
    # Class weights for balanced learning
    train_targets = [train_subset[i][1] for i in range(len(train_subset))]
    class_counts = np.bincount(train_targets, minlength=len(classes))
    weights = 1.0 / (class_counts + 1e-5)
    weights = weights / weights.sum()
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    epochs = 20
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "cnn_pakcoy_model.pt")
    
    print("=== START TRAINING CNN ===")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total_samples += images.size(0)
            
        scheduler.step()
        train_loss = running_loss / total_samples
        train_acc = correct / total_samples
        
        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += torch.sum(preds == labels.data).item()
                v_samples += images.size(0)
                
        val_loss = v_loss / v_samples
        val_acc = v_correct / v_samples
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            
        print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    test_acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    print("\n=== TEST EVALUATION METRICS ===")
    print(f"Test Accuracy : {test_acc*100:.2f}%")
    print(f"Macro Precision: {p*100:.2f}%")
    print(f"Macro Recall   : {r*100:.2f}%")
    print(f"Macro F1-Score : {f1*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    
    metrics = {
        "version": "1.0.0",
        "model_type": "CNN - Pakcoy Growth Stage Classifier",
        "classes": classes,
        "test_accuracy": float(test_acc),
        "test_precision": float(p),
        "test_recall": float(r),
        "test_f1": float(f1),
        "confusion_matrix": cm,
        "history": history
    }
    
    with open(os.path.join(MODELS_DIR, "cnn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("CNN Model training and evaluation completed successfully!")

if __name__ == "__main__":
    train_cnn()
