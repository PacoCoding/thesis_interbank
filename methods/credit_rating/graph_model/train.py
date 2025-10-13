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
from methods.credit_rating.graph_model.utils import load_data, preprocess, set_seed, _slug, download_file, zip_folder
from methods.credit_rating.graph_model.TGAR import TGAR, Model
from sklearn.utils.class_weight import compute_class_weight
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os, shutil

# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

# Useful for testing the models and different hypperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hiddim', type=int, default=256, help='Hidden units.')
parser.add_argument('--droprate', type=float, default=0.3, help='Dropout rate.')
parser.add_argument('--year', type=int, default=2020)
parser.add_argument('--runs', type=int, default=2, help='How many independent runs (different seeds).')
parser.add_argument('--base_seed', type=int, default=42, help='Base seed; each run adds +i.')
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
label_to_index, labels, features, edge_index, dead_mask, edge_weight = load_data(args.year, args.quarter)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  divide the initial database that is composed by new + old graph in the two distinct databases
N = labels.size(0)
N_old = N // 2
# 2 indices array for the old and new
idx_old = np.arange(0, N_old)
idx_new = np.arange(N_old, N)

# Stratify ONLY within the older half
y_old = labels[idx_old].cpu().numpy()
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
# next is used for extracting the results
tr_old, val_old = next(sss.split(idx_old, y_old))

train_idx = idx_old[tr_old]
val_idx   = idx_old[val_old]
test_idx  = idx_new

# Masks creation, first full of False then index for train,val and test with True, this is for the OLD graph
train_mask = torch.zeros(N, dtype=torch.bool); train_mask[torch.from_numpy(train_idx)] = True
val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[torch.from_numpy(val_idx)]     = True
test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[torch.from_numpy(test_idx)]   = True

# Safe scaling (fit on TRAIN nodes only)
# Evaluate whehther transforming features in numpy if they are not already, fundamental for the normalization
X_np = features.numpy() if isinstance(features, torch.Tensor) else features
fit_idx = train_idx[~dead_mask[train_idx].numpy()]
scaler = MinMaxScaler()
scaler.fit(X_np[fit_idx])                     # Only fitting index in train and ALIVE
X_norm = torch.from_numpy(scaler.transform(X_np)).float()
edge_weight = torch.clamp(edge_weight, min=0.0) # Clapping weights for the Transformer 
# Just for making the vector in column shape
edge_attr = edge_weight.view(-1, 1)

data = Data(
    x=X_norm,
    edge_index=edge_index.t().contiguous(),
    edge_attr=edge_attr,   
    y=labels,
    dead=dead_mask
).to(device)

print("Total nodes:", data.num_nodes)
print("Dead nodes (all):", int(data.dead.sum()))
print("Dead in TRAIN:", int(data.dead[train_mask].sum()))
print("Dead in VAL:",   int(data.dead[val_mask].sum()))
print("Dead in TEST:",  int(data.dead[test_mask].sum()))
# Sanity: no train↔test edges
src, dst = data.edge_index
touch_dead = (data.dead[src] | data.dead[dst]).sum().item()
print("Edges incident to dead (should be 0):", touch_dead)
cross = (train_mask[src] & test_mask[dst]) | (test_mask[src] & train_mask[dst])
print("cross-split edges:", int(cross.sum().item()))  # expect 0

# Loaders objects
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
        # out is a tensor with rows the nodes and column the classses, for each a probability
        out = self.model(batch)  # log-probs [num_nodes_in_subgraph, C]
        # nodes.x has information also about the neighbours, we only want the actual node for the los
        logits = out[:batch.batch_size]
        y      = batch.y[:batch.batch_size]
        # drop dead seeds
        # check if the object batch has attribute
        if hasattr(batch, 'dead'):
            keep = ~batch.dead[:batch.batch_size]
            logits = logits[keep]
            y      = y[keep]
        if logits.numel() == 0:         # all were dead → skip step
            return
        loss = F.nll_loss(logits, y, weight=self.class_weight)
        loss.backward()
        self.optimizer.step()

    def test(self, batch):
        self.model.eval()
        out = self.model(batch)
        pred = out.cpu().max(dim=1).indices[:batch.batch_size]
        tru  = batch.y.cpu()[:batch.batch_size]
        if hasattr(batch, 'dead'):
            m = ~batch.dead[:batch.batch_size].cpu()
            pred = pred[m]
            tru  = tru[m]
        self.model.train()
        return tru, pred

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
tag = f"{args.year}Q{args.quarter}_{args.net}"
result_dir = f'./results/{tag}'
os.makedirs(result_dir, exist_ok=True)

def build_net():
    if args.net == "GCN":
        return GCN(features.shape[1], num_classes, hiddim=args.hiddim,
                   droprate=args.droprate, hidlayers=args.hidlayers, p=1).to(device)
    elif args.net == "GAT":
        return GAT(features.shape[1], num_classes, hiddim=args.hiddim,
                   droprate=args.droprate, hidlayers=args.hidlayers, p=1).to(device)
    else:  # TGAR
        return TGAR(args.batchsize, features.shape[1], num_classes, hiddim=args.hiddim,
                    droprate=args.droprate, hidlayers=args.hidlayers, p=1, hyper_k=4).to(device)

def save_cm(y_true_np, y_pred_np, run_idx, year, quarter, model_name, out_dir):
    classes = np.arange(num_classes)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=classes, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in classes])
    fig = disp.plot(include_values=True, cmap='Blues', values_format=".2f", colorbar=False).figure_
    title = f"{year} Q{quarter} — {model_name} — Confusion Matrix)"
    plt.title(title)
    fname = f"cm_{year}Q{quarter}_{_slug(model_name)}_run{run_idx}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return cm

runs_rows = []         # per-run metrics rows (for CSV)
cms_normalized = []    # per-run normalized confusion matrices

for i in range(args.runs):
    seed = args.base_seed + i
    print(f"\n===== RUN {i+1}/{args.runs} (seed={seed}) =====")
    set_seed(seed)

    # fresh model each run
    gnnnet = build_net()
    model  = WeightedModel(gnnnet, args, device, class_weight=class_weight)

    # ── train
    for epoch in range(args.epochs):
        for batch in train_loader:
            model.fit(batch)

        # (optional) progress log every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                tru_all, pred_all = [], []
                for tb in val_loader:
                    t, p = model.test(tb)
                    tru_all.append(t); pred_all.append(p)
                tru_t  = torch.cat(tru_all).cpu().numpy()
                pred_t = torch.cat(pred_all).cpu().numpy()

            acc  = accuracy_score(tru_t, pred_t)
            prec = precision_score(tru_t, pred_t, average='macro', zero_division=0)
            rec  = recall_score(tru_t, pred_t, average='macro', zero_division=0)
            f1   = f1_score(tru_t, pred_t, average='macro', zero_division=0)

            print(f"[RUN {i+1} | E{epoch+1}/{args.epochs}] ACC={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f}")

    # ── final eval for this run
    with torch.no_grad():
        tru_all, pred_all = [], []
        for tb in test_loader:
            t, p = model.test(tb)
            tru_all.append(t); pred_all.append(p)
        y_true = torch.cat(tru_all).cpu().numpy()
        y_pred = torch.cat(pred_all).cpu().numpy()

    # metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, digits=5, zero_division=0)

    # save per-run classification report
    txt_path = os.path.join(result_dir, f"classification_report_run{i+1}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{gnnnet.model_name} | {tag} | seed={seed} | epochs={args.epochs}\n")
        f.write(f"ACC={acc:.6f} | PREC_macro={prec:.6f} | RECALL_macro={rec:.6f} | F1_macro={f1:.6f}\n\n")
        f.write(report)
    print(f"Wrote TXT: {txt_path}")

    # confusion matrix (normalized)
    cm_norm = save_cm(
        y_true, y_pred, run_idx=i+1,
        year=args.year, quarter=args.quarter,
        model_name=gnnnet.model_name,
        out_dir=result_dir
    )
    cms_normalized.append(cm_norm)

    # collect row for per-tag runs CSV
    runs_rows.append({
        "year": args.year, "quarter": args.quarter,
        "model": gnnnet.model_name, "net_flag": args.net,
        "seed": seed, "epochs": args.epochs,
        "acc": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1
    })

# ── Save per-run CSV under tag folder
runs_df = pd.DataFrame(runs_rows)
runs_csv = os.path.join(result_dir, "runs.csv")
runs_df.to_csv(runs_csv, index=False)
print(f"Wrote per-run metrics: {runs_csv}")

# ── Mean/Std aggregate under tag folder
agg = runs_df[["acc","precision_macro","recall_macro","f1_macro"]].agg(['mean','std'])
agg_csv = os.path.join(result_dir, "aggregate_metrics.csv")
agg.to_csv(agg_csv)
print(f"Wrote aggregate mean/std: {agg_csv}")

# ── Also append a single aggregate row to global summary.csv (root)
summary_row = {
    "year": args.year, "quarter": args.quarter, "model": gnnnet.model_name,
    "net_flag": f"{args.net}_mean{args.runs}", "epochs": args.epochs,
    "acc": agg.loc['mean','acc'], "precision_macro": agg.loc['mean','precision_macro'],
    "recall_macro": agg.loc['mean','recall_macro'], "f1_macro": agg.loc['mean','f1_macro']
}
global_csv_path = './results/summary.csv'
header_needed = not os.path.isfile(global_csv_path)
pd.DataFrame([summary_row]).to_csv(global_csv_path, mode='a', header=header_needed, index=False)
print(f"Appended aggregate to: {global_csv_path}")

# ── Mean confusion matrix (average of normalized CMs)
if len(cms_normalized) > 0:
    import numpy as np
    mean_cm = np.mean(np.stack(cms_normalized, axis=0), axis=0)
    disp = ConfusionMatrixDisplay(mean_cm, display_labels=[str(i) for i in range(num_classes)])
    fig = disp.plot(include_values=True, cmap='Blues', values_format=".2f", colorbar=False).figure_
    mean_title = f"{args.year} Q{args.quarter} — {gnnnet.model_name} — Mean Confusion Matrix over {args.runs} runs"
    plt.title(mean_title)
    mean_cm_path = os.path.join(result_dir, f"cm_{args.year}Q{args.quarter}_{_slug(gnnnet.model_name)}_mean.png")
    fig.savefig(mean_cm_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote mean confusion matrix: {mean_cm_path}")
zip_path = f"/content/{os.path.basename(result_dir)}.zip"
zip_folder(result_dir, zip_path)
download_file(zip_path)
