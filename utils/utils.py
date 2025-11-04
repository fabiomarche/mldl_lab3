import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random, numpy as np, torch


def set_seed(seed:int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_for_one_epoch(model: torch.nn.Module, train_loader, loss_fn, optimizer: torch.optim.Optimizer, device: str, epoch: int, log_every: int = 100) -> Dict[str, float]:
    """Executes one epoch of training. It returns a dict with average loss/acc"""
    
    model.train()
    total, correct, current_loss = 0, 0, 0.0
    
    for step, (X,y) in enumerate(train_loader, start = 1):
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        
        if step % log_every == 0:
            print(f"[Epoch {epoch} step {step}]"
                  f"Loss={current_loss/total:.4f} acc={correct/total:.4f}")
    return {
        "loss": current_loss / max(total,1),
        "acc": correct / max(total,1)
    }
    
@torch.no_grad()
def evaluate(model: torch.nn.Module, eval_loader, device: str, loss_fn) -> Dict[str, float]:
    """Eval loop"""
    
    model.eval()
    total, correct, current_loss = 0, 0, 0.0
    
    for X, y in eval_loader:
        X = X.to(device)
        y = y.to(device)
        
        logits = model(X)
        
        loss = loss_fn(logits, y)
        current_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        
    return {
        "loss": current_loss / max(total,1),
        "acc": correct / max(total,1)
    }

def save_checkpoint(path: str, model: torch.nn.Module):
    to_save = {"state_dict": model.state_dict()}
    torch.save(to_save, path)

def load_checkpoint(path: str, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    return {k: v for k, v in ckpt.items() if k != "state_dict"}

def show_n_images(rows:int, cols:int, loader, idx_to_labels):
  figure = plt.figure(figsize=(9,9))
  for i in range(1, cols * rows + 1):
    idx = torch.randint(0, len(loader.dataset), size = (1,)).item()
    img, label = loader.dataset[idx]
    figure.add_subplot(rows, cols, i)
    plt.title(f"Label: {label} - Human label: {idx_to_labels[label]}")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.tight_layout()
  plt.show()
  
def plot_history(history: Dict[str, List[float]]) -> None:
    """
    Plotta l'andamento di loss/acc nel tempo.
    Esempio di history:
        history = {
          'train_loss': [...], 'val_loss': [...],
          'train_acc':  [...], 'val_acc':  [...]
        }
    """
    # LOSS
    if "train_loss" in history or "val_loss" in history:
        plt.figure()
        if "train_loss" in history: plt.plot(history["train_loss"], label="train_loss")
        if "val_loss"   in history: plt.plot(history["val_loss"],   label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Loss")
        plt.show()

    # ACC
    if "train_acc" in history or "val_acc" in history:
        plt.figure()
        if "train_acc" in history: plt.plot(history["train_acc"], label="train_acc")
        if "val_acc"   in history: plt.plot(history["val_acc"],   label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.show()