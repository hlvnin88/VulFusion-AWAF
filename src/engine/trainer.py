import os
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.metrics import compute_metrics


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
            return True
        self.count += 1
        if self.count >= self.patience:
            self.stop = True
        return False


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    y_true, y_prob = [], []

    for b in tqdm(loader, leave=False):
        ids_c = b["input_ids_c"].to(device)
        msk_c = b["attention_mask_c"].to(device)
        ids_g = b["input_ids_g"].to(device)
        msk_g = b["attention_mask_g"].to(device)
        pos_g = b["position_idx_g"].to(device)
        y = b["label"].to(device)                    # long (B,)

        with torch.set_grad_enabled(train_mode):
            logits = model(ids_c, msk_c, ids_g, msk_g, pos_g)   # (B, 2)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

        total_loss += loss.item() * y.size(0)
        # Eq.(10): Softmax → lấy xác suất class 1
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
        y_prob.extend(probs)
        y_true.extend(y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, compute_metrics(y_true, y_prob)


def train_engine(model, train_loader, val_loader, train_cfg, device, ckpt_path):
    # Eq.(11): Cross-Entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = max(len(train_loader) * train_cfg["epochs"], 1)
    warmup_steps = int(train_cfg["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    stopper = EarlyStopping(train_cfg["early_stopping_patience"], train_cfg["min_delta"])
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    # luôn có checkpoint để load lại kể cả khi không epoch nào cải thiện
    torch.save(model.state_dict(), ckpt_path)

    for ep in range(1, train_cfg["epochs"] + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        va_loss, va_m = run_epoch(model, val_loader, device, criterion, None, None)
        print(f"[Epoch {ep}] TrainLoss={tr_loss:.4f} ValLoss={va_loss:.4f} "
              f"ValF1={va_m['f1']:.4f} ValMCC={va_m['mcc']:.4f}")

        if stopper.step(va_loss):
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved {ckpt_path}")

        if stopper.stop:
            print("  -> early stop")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


@torch.no_grad()
def evaluate(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    return run_epoch(model, test_loader, device, criterion, None, None)