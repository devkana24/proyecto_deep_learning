import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.hub import slowfast_r50
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class SlowFastClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super().__init__()
        
        # Cargar modelo pre-entrenado
        self.model = slowfast_r50(pretrained=pretrained)
        

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        original_classifier = self.model.blocks[-1].proj
        in_features = original_classifier.in_features
 
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        
        # Configuración de pathways
        self.slow_alpha = 4
        self.fast_alpha = 1
    
    def forward(self, x):
        """
        Input: x [B, C, T, H, W]
        """
        # Preparar pathways para SlowFast
        slow_pathway = x[:, :, ::self.slow_alpha, :, :]  # [B, C, T//4, H, W]
        fast_pathway = x  # [B, C, T, H, W]

        inputs = [slow_pathway, fast_pathway]
        
      
        return self.model(inputs)
    


def save_checkpoint(model, optimizer, epoch, metric, path="best_model.pth"):
    """
    Guarda el modelo, optimizador y estado adicional.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric': metric,
        'model_class': 'SlowFastClassifier'
    }, path)
    print(f"✅ Checkpoint guardado en '{path}' (epoch={epoch}, metric={metric:.4f})")


def load_model(model, optimizer, path="best_model.pth", device="cuda"):
    """
    Carga modelo y optimizador desde archivo.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metric = checkpoint['metric']
    print(f"✅ Checkpoint cargado desde '{path}' (epoch={epoch}, metric={metric:.4f})")
    return epoch, metric


def evaluate_model(model, data_loader, device="cuda", class_names=None):
    """
    Evalúa el modelo con métricas detalladas.
    """
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Guardar para métricas detalladas
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total
    acc = correct / total
    
    print(f"🎯 Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")
    
    # Métricas detalladas
    if class_names is None:
        class_names = ['No Violence', 'Violence']
    
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\n🔍 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    return avg_loss, acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, num_epochs=20, device="cuda"):
    """
    Función de entrenamiento mejorada con mejores prácticas.
    """
    # Configuración del optimizador y scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Early stopping
    best_val_loss = np.inf
    patience = 5
    trigger_times = 0
    
    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        running_loss, correct, total = 0, 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Progress tracking
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        
        # === VALIDATION ===
        model.eval()
        running_loss_eval, correct_eval, total_eval = 0, 0, 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                running_loss_eval += loss.item()
                predicted = outputs.argmax(dim=1)
                correct_eval += (predicted == labels).sum().item()
                total_eval += labels.size(0)
        
        val_loss = running_loss_eval / len(val_loader)
        val_acc = correct_eval / total_eval * 100
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Tracking
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping & save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            save_checkpoint(model, optimizer, epoch, val_acc, path="best_model.pth")
            print(f"🎯 New best model saved! Val Loss: {val_loss:.4f}")
        else:
            trigger_times += 1
            print(f"⚠️ EarlyStopping counter: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("⛔ Early stopping triggered.")
                break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


def test_model(model, test_loader, device="cuda"):
    """
    Evaluación final en test set.
    """
    print("🧪 Evaluating on test set...")
    test_loss, test_acc, preds, labels = evaluate_model(
        model, test_loader, device, class_names=['No Violence', 'Violence']
    )
    
    # Métricas específicas para violencia
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1]
    )
    
    print(f"\n📋 Test Results Summary:")
    print(f"Overall Accuracy: {test_acc*100:.2f}%")
    print(f"No Violence - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1: {f1[0]:.3f}")
    print(f"Violence - Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}, F1: {f1[1]:.3f}")
    
    return test_acc, precision, recall, f1