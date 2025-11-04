import argparse, os, torch
from dataset.dataset import build_datasets, build_loaders, get_data_info
from models.model import build_model
from utils.utils import set_seed, train_for_one_epoch, evaluate, save_checkpoint, plot_history

def parse_args():
    p = argparse.ArgumentParser(description="Train SimpleImageClassifier (FashionMNIST)")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    
    device = "cuda" if torch.accelerator.is_available() else "cpu"
    print(device)
    
    train_ds, val_ds = build_datasets(data_dir=args.data_dir)
    train_loader, val_loader = build_loaders(train_ds, val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    
    info = get_data_info()
    model = build_model(num_classes=info["num_classes"]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_path = 0.0, os.path.join(args.out_dir, "best.pt")
    
    for epoch in range(1, args.epochs + 1):
        tr = train_for_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, log_every=100)
        va = evaluate(model, val_loader, device, loss_fn)
        
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        
        print(f"[epoch {epoch}] train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
              f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}")
        
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            save_checkpoint(best_path, model)
            print(f"  ↳ new best! acc={best_acc:.4f} saved -> {best_path}")
    
    print(f"Done. Best val_acc = {best_acc:.4f}")
    plot_history(history)

if __name__ == "__main__":
    main()