# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Backbone MViT base 16x4 (preentrenado en Kinetics-400)
from pytorchvideo.models.hub import mvit_base_16x4


# ============================================================
# Modelo: Transformer (MViT base 16x4)
# ============================================================
class TransformerVideoClassifier(nn.Module):
    """
    Clasificador de video basado en MViT (mvit_base_16x4).
    Entrada: [B, C, T, H, W] (p.ej. [B, 3, 16, 224, 224]).
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False, head_dropout=0.30):
        super().__init__()
        self.backbone = mvit_base_16x4(pretrained=pretrained)

        # Congelar backbone si se pide (la head nueva seguirá entrenable)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Reemplazo de la head con Dropout (maneja Linear o Conv3d según versión)
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "proj"):
            proj = self.backbone.head.proj
            if isinstance(proj, nn.Linear):
                in_features = proj.in_features
                self.backbone.head.proj = nn.Sequential(
                    nn.Dropout(p=head_dropout),
                    nn.Linear(in_features, num_classes)
                )
            elif isinstance(proj, nn.Conv3d):
                in_channels = proj.in_channels
                # Pool -> Flatten -> Dropout -> Linear
                self.backbone.head = nn.Sequential(
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=head_dropout),
                    nn.Linear(in_channels, num_classes)
                )
            else:
                # Fallback genérico: buscar última Linear como fuente de features
                in_features = None
                for m in reversed(list(self.backbone.modules())):
                    if isinstance(m, nn.Linear):
                        in_features = m.in_features
                        break
                if in_features is None:
                    raise RuntimeError("No se encontró capa adecuada para construir la head.")
                self.backbone.head = nn.Sequential(
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=head_dropout),
                    nn.Linear(in_features, num_classes)
                )
        else:
            # Fallback si no existe .head.proj (versiones raras)
            in_features = None
            for m in reversed(list(self.backbone.modules())):
                if isinstance(m, nn.Linear):
                    in_features = m.in_features
                    break
            if in_features is None:
                raise RuntimeError("No se encontró capa adecuada para construir la head.")
            self.backbone.head = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Dropout(p=head_dropout),
                nn.Linear(in_features, num_classes)
            )

        self.num_classes = num_classes

    def forward(self, x):
        # x: [B, C, T, H, W]
        return self.backbone(x)
# ============================================================
# Checkpoints
# ============================================================
def save_checkpoint(model, optimizer, epoch, metric, path="best_model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric': metric,
        'model_class': 'TransformerVideoClassifier'
    }, path)
    print(f"✅ Checkpoint guardado en '{path}' (epoch={epoch}, metric={metric:.4f})")


def load_model(model, optimizer, path="best_model.pth", device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt.get('epoch', 0)
    metric = ckpt.get('metric', 0.0)
    print(f"✅ Checkpoint cargado desde '{path}' (epoch={epoch}, metric={metric:.4f})")
    return epoch, metric


# ============================================================
# Evaluación
# ============================================================
@torch.no_grad()
def evaluate_model(model, data_loader, device="cuda", class_names=None):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in data_loader:
        # inputs: [B, C, T, H, W], labels: [B]
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    if class_names is None:
        class_names = ['No Violence', 'Violence']

    print(f"🎯 Loss: {avg_loss:.4f}, Accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return avg_loss, acc, all_preds, all_labels


# ============================================================
# Entrenamiento
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    device="cuda",
    lr=2e-4,
    weight_decay=1e-4,
    label_smoothing=0.05,
    use_amp=True,
    grad_clip=1.0,
):
    """
    Entrenamiento con AdamW + ReduceLROnPlateau (val_loss), AMP y gradient clipping.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))

    best_val = float("inf")
    patience, triggers = 5, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for bidx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)   # [B,C,T,H,W] (T=16)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            run_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if bidx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {bidx}/{len(train_loader)} | Loss {loss.item():.4f}")

        train_loss = run_loss / max(1, len(train_loader))
        train_acc = (correct / max(1, total)) * 100.0

        # ---------- VAL ----------
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                vloss += loss.item()
                vcorrect += (logits.argmax(dim=1) == y).sum().item()
                vtotal += y.size(0)

        val_loss = vloss / max(1, len(val_loader))
        val_acc = (vcorrect / max(1, vtotal)) * 100.0

        scheduler.step(val_loss)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);   val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            triggers = 0
            save_checkpoint(model, optimizer, epoch, val_acc, path="best_model.pth")
            print(f"🎯 New best model! Val Loss: {val_loss:.4f}")
        else:
            triggers += 1
            print(f"⚠️ EarlyStopping counter: {triggers}/{patience}")
            if triggers >= patience:
                print("⛔ Early stopping activado.")
                break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


# ============================================================
# Test final
# ============================================================
@torch.no_grad()
def test_model(model, test_loader, device="cuda"):
    print("🧪 Evaluando en test set...")
    test_loss, test_acc, preds, labels = evaluate_model(
        model, test_loader, device=device, class_names=['No Violence', 'Violence']
    )
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1])

    print(f"\n📋 Test Results Summary:")
    print(f"Overall Accuracy: {test_acc*100:.2f}%")
    print(f"No Violence - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1: {f1[0]:.3f}")
    print(f"Violence     - Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}, F1: {f1[1]:.3f}")

    return test_acc, precision, recall, f1
