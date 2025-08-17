import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

from utils import load_data, preprocess
from GCN import GCN, Model
from GAT import GAT
from TGAR import TGAR

# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hiddim', type=int, default=256, help='Hidden units.')
parser.add_argument('--droprate', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--year', type=int, default=2020)
parser.add_argument('--quarter', type=int, default=2)
parser.add_argument('--epochs', type=int,  default=1000, help='Epochs.')
parser.add_argument('--batchsize', type=int, default=4548, help='Batch size (seed nodes).')
parser.add_argument('--numneighbors', type=int, default=30, help='Neighbors (unused with -1).')
parser.add_argument('--hidlayers', type=int, default=2, help='Hidden layers.')
parser.add_argument('--net', type=str, default="GCN", choices=["GCN", "GAT", "TGAR"])
args = parser.parse_args()

os.makedirs("./results", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Data (single quarter file that contains [older | newer] halves)
# ──────────────────────────────────────────────────────────────────────────────
label_to_index, labels, features, edge_index = load_data(args.year, args.quarter)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Halves: first = older (train pool), second = newer (test)
N = labels.size(0)
N_old = N // 2
idx_old = np.arange(0, N_old)
idx_new = np.arange(N_old, N)

# Stratify ONLY within the older half
y_old = labels[idx_old].cpu().numpy()
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
tr_old, val_old = next(sss.split(idx_old, y_old))

train_idx = idx_old[tr_old]
val_idx   = idx_old[val_old]
test_idx  = idx_new

# Masks
train_mask = torch.zeros(N, dtype=torch.bool); train_mask[torch.from_numpy(train_idx)] = True
val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[torch.from_numpy(val_idx)]     = True
test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[torch.from_numpy(test_idx)]   = True

# Safe scaling (fit on TRAIN nodes only)
X_np = features.numpy() if isinstance(features, torch.Tensor) else features
scaler = MinMaxScaler()
scaler.fit(X_np[train_idx])
X_norm = torch.from_numpy(scaler.transform(X_np)).float()

data = Data(
    x=X_norm,
    edge_index=edge_index.t().contiguous(),
    y=labels
).to(device)

# Sanity: no train↔test edges
src, dst = data.edge_index
cross = (train_mask[src] & test_mask[dst]) | (test_mask[src] & train_mask[dst])
print("cross-split edges:", int(cross.sum().item()))  # expect 0

# Loaders
train_loader = NeighborLoader(
    data, num_neighbors=[-1], batch_size=min(args.batchsize, len(train_idx)), input_nodes=train_mask
)
val_loader = NeighborLoader(
    data, num_neighbors=[-1], batch_size=max(1, len(val_idx)), input_nodes=val_mask
)
test_loader = NeighborLoader(
    data, num_neighbors=[-1], batch_size=min(args.batchsize, len(test_idx)), input_nodes=test_mask
)

# Class weights (train-only)
from sklearn.utils.class_weight import compute_class_weight
num_classes = len(label_to_index)
cw = compute_class_weight(class_weight='balanced',
                          classes=np.arange(num_classes),
                          y=labels[train_mask].cpu().numpy())
class_weight = torch.tensor(cw, dtype=torch.float, device=device)
print(f"class weights (train-only): {cw}")

# Weighted wrapper
class WeightedModel(Model):
    def __init__(self, model, args, device, class_weight=None):
        super().__init__(model, args, device)
        self.class_weight = class_weight.to(device) if class_weight is not None else None

    def fit(self, batch):
        self.optimizer.zero_grad()
        out = self.model(batch)  # log-probs [B,C]
        loss = F.nll_loss(
            out[:batch.batch_size],
            batch.y[:batch.batch_size],
            weight=self.class_weight
        )
        loss.backward()
        self.optimizer.step()

# Net
if args.net == "GCN":
    gnnnet = GCN(features.shape[1], num_classes, hiddim=args.hiddim, droprate=args.droprate, hidlayers=args.hidlayers, p=1).to(device)
elif args.net == "GAT":
    gnnnet = GAT(features.shape[1], num_classes, hiddim=args.hiddim, droprate=args.droprate, hidlayers=args.hidlayers, p=1).to(device)
else:  # TGAR
    gnnnet = TGAR(args.batchsize, features.shape[1], num_classes, hiddim=args.hiddim, droprate=args.droprate, hidlayers=args.hidlayers, p=1, hyper_k=4).to(device)

model = WeightedModel(gnnnet, args, device, class_weight=class_weight)

# ──────────────────────────────────────────────────────────────────────────────
# Train loop — keep only LAST epoch metrics for saving
# ──────────────────────────────────────────────────────────────────────────────
last_acc = last_prec = last_rec = last_f1 = 0.0
last_report = ""
for epoch in range(args.epochs):
    # Train
    for batch in train_loader:
        model.fit(batch)

    # Eval on test
    with torch.no_grad():
        tru_all, pred_all = [], []
        for tb in test_loader:
            t, p = model.test(tb)
            tru_all.append(t); pred_all.append(p)
        tru_t = torch.cat(tru_all).cpu().numpy()
        pred_t = torch.cat(pred_all).cpu().numpy()

    # Metrics (macro for multi-class)
    last_acc  = accuracy_score(tru_t, pred_t)
    last_prec = precision_score(tru_t, pred_t, average='macro', zero_division=0)
    last_rec  = recall_score(tru_t, pred_t, average='macro', zero_division=0)
    last_f1   = f1_score(tru_t, pred_t, average='macro', zero_division=0)
    last_report = classification_report(tru_t, pred_t, digits=5, zero_division=0)

    # (Optional) live log each epoch
    if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
        print(f"[E{epoch+1}/{args.epochs}] ACC={last_acc:.4f} | P={last_prec:.4f} | R={last_rec:.4f} | F1={last_f1:.4f}")
        print(last_report)
# ──────────────────────────────────────────────────────────────────────────────
# Save ONLY the last epoch results
# ──────────────────────────────────────────────────────────────────────────────
tag = f"{args.year}Q{args.quarter}_{args.net}"
txt_path = f'./results/classification_report_{tag}.txt'
csv_path = f'./results/summary.csv'

# TXT (overwrite with last-epoch report only)
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f"{gnnnet.model_name} | {tag} | epochs={args.epochs}\n")
    f.write(f"ACC={last_acc:.6f} | PREC_macro={last_prec:.6f} | RECALL_macro={last_rec:.6f} | F1_macro={last_f1:.6f}\n\n")
    f.write(last_report)
print(f"Wrote TXT: {txt_path}")

# CSV (append one row)
import pandas as pd
row = {
    "year": args.year,
    "quarter": args.quarter,
    "model": gnnnet.model_name,
    "net_flag": args.net,
    "epochs": args.epochs,
    "acc": last_acc,
    "precision_macro": last_prec,
    "recall_macro": last_rec,
    "f1_macro": last_f1,
}
df = pd.DataFrame([row])
header_needed = not os.path.isfile(csv_path)
df.to_csv(csv_path, mode='a', header=header_needed, index=False)
print(f"Appended CSV: {csv_path}")
