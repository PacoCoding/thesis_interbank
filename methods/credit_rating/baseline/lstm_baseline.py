###########
# COMPARISON LSTM  (with multi-run + mean/std aggregates)
########
# LSTM baseline — no-leakage, uses all (safe) features + degrees
# Train:  windows ending at t-2  -> predict t-1
# Test :  window  ending at t-1  -> predict t
# Preprocess = heuristic log1p + RobustScaler on all features (fit ≤ t-1)

import os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import shutil
from pathlib import Path
from methods.credit_rating.utils.data_utils import (
    TARGET, _set_seed, _clean_id,
    enrich_nodes, make_feature_cols, make_preprocessor,
    build_dead_masks_up_to, build_label_for_next
)
import torch
ROOT = Path(__file__).resolve().parents[3]     # repo root
DATASETS = ROOT / "datasets"

# use repo-relative patterns (nodes/ and edges/ directories)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "rank_next_quarter"
NODES_PAT = str((DATASETS / "nodes" / "[0-9][0-9][0-9][0-9]Q[1-4].csv").resolve())
EDGES_PAT = str((DATASETS / "edges" / "edge_[0-9][0-9][0-9][0-9]Q[1-4].csv").resolve())


SEQ_LEN     = 6
EPOCHS      = 60
LR          = 1e-3
WEIGHT_DECAY= 5e-4
D_LSTM_HID  = 128
D_LSTM_LAY  = 3
D_MLP_HID   = 64
DROP        = 0.1
GRAD_CLIP   = 1.0

# ── NEW: multi-run controls (leave RUNS=1 for old behavior)
RUNS       = 4         # set >1 to compute mean/std aggregates
BASE_SEED  = 42
RESULTS_ROOT = "./results_lstm"   # all outputs land here
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ───────────── model ─────────────
class TemporalLSTM(nn.Module):
    def __init__(self, in_feats, lstm_hidden=128, lstm_layers=2, mlp_hidden=128, num_classes=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )
    def forward(self, x):  # x: [N, T, F]
        h, _ = self.lstm(x)
        h_last = h[:, -1, :]
        return self.head(self.dropout(h_last))

# ───────────── main walk-forward ─────────────
def run_walkforward_lstm(start_quarter=None, end_quarter=None, show_cm=False,
                         runs=1, base_seed=42, EPOCHS = 60):
    import glob, os
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, classification_report,
                                 confusion_matrix, ConfusionMatrixDisplay)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    # Load files
    nodes_paths = sorted(glob.glob(NODES_PAT))
    edges_paths = sorted(glob.glob(EDGES_PAT))
    quarters    = [os.path.basename(p).replace('.csv','') for p in nodes_paths]

    # Enrich all quarters
    enriched = []
    for pn, pe, q in zip(nodes_paths, edges_paths, quarters):
        nd = pd.read_csv(pn); ed = pd.read_csv(pe)
        df = enrich_nodes(nd, ed)
        if TARGET in df.columns:
            enriched.append((df, ed, q))
    usable = enriched

    quarters_list = [q for _,_,q in usable]
    print("Loaded quarters:", quarters_list)

    # Features
    sample_df = usable[0][0]
    FEATURE_COLS = make_feature_cols(sample_df)
    print("feautesss", FEATURE_COLS)
    assert TARGET not in FEATURE_COLS
    print("All quarters with TARGET:", [q for _,_,q in enriched])
    print("Usable (drop last):", [q for _,_,q in usable])
    # Class count (safe)
    labels_flat = np.concatenate([df[TARGET].values-1 for df,_,_ in usable[SEQ_LEN:]])

    #OUT_CLASSES = int(np.unique(labels_flat).max() + 1)
    OUT_CLASSES = 4
    print(f"OUT_CLASSES = {OUT_CLASSES} | #features = {len(FEATURE_COLS)}")

    # Range
    if start_quarter is None:
        start_idx = SEQ_LEN + 1
    else:
        if start_quarter not in quarters_list:
            raise ValueError(f"start_quarter '{start_quarter}' not found. Have: {quarters_list}")
        start_idx = max(quarters_list.index(start_quarter), SEQ_LEN + 1)
    if end_quarter is None:
        end_idx = len(usable) - 1
    else:
        if end_quarter not in quarters_list:
            raise ValueError(f"end_quarter '{end_quarter}' not found. Have: {quarters_list}")
        end_idx = quarters_list.index(end_quarter)

    print(f"Walk-forward from {quarters_list[start_idx]} to {quarters_list[end_idx]}")

    # Dead masks
    all_dead_masks = build_dead_masks_up_to(usable, len(usable)-1)

    ROOT_DIR = "./results"
    os.makedirs(ROOT_DIR, exist_ok=True)
    global_rows = []

    for t in range(start_idx, end_idx + 1):
        qn = quarters_list[t]             # e.g. "2019Q3"
        tag = f"{qn}_LSTM"
        out_dir = os.path.join(ROOT_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== Target {qn} (t={t}) — running {runs} seeds ===")

        # Fit preprocessor on ≤ t-1
        pre_hist = pd.concat([usable[i][0][FEATURE_COLS] for i in range(0, t)], ignore_index=True)
        pre, _ = make_preprocessor(pre_hist)
        pre.fit(pre_hist)

        # Fixed test window (t-7..t-1 -> predict t)
        left_test = t - SEQ_LEN + 1
        seq_te = []
        for i in range(left_test, t + 1):
            df_i, _, _ = usable[i]
            Xt = pre.transform(df_i[FEATURE_COLS])
            dead_i = all_dead_masks[i]
            Xt[dead_i, :] = 0
            seq_te.append(Xt)
        X_te_np = np.stack(seq_te, axis=0).transpose(1, 0, 2)  # [N,T,F]
        X_test = torch.tensor(X_te_np, dtype=torch.float32, device=DEVICE)
        y_test = build_label_for_next(usable, t-1, DEVICE)
        y_true_full = y_test.detach().cpu().numpy()
        alive_mask = (y_true_full != -100)

        run_rows = []
        cms_norm = []

        for r in range(runs):
            seed = base_seed + r
            _set_seed(seed)
            print(f"  -- run {r+1}/{runs} (seed={seed})")

            # Training pool: e = SEQ_LEN .. t-2
            pool = []
            for e in range(SEQ_LEN, t):
                left = e - SEQ_LEN + 1
                seq = []
                for i in range(left, e + 1):
                    df_i, _, _ = usable[i]
                    Xt = pre.transform(df_i[FEATURE_COLS])
                    dead_i = all_dead_masks[i]
                    Xt[dead_i, :] = 0
                    seq.append(Xt)
                X_np = np.stack(seq, axis=0).transpose(1,0,2)
                X_seq = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
                y_e = build_label_for_next(usable, e-1, DEVICE)
                pool.append((X_seq, y_e))

            # Class weights from alive labels in pool
            all_y = np.concatenate([y.cpu().numpy() for _, y in pool])
            alive = (all_y != -100)
            classes = np.unique(all_y[alive])
            weight = torch.ones(OUT_CLASSES, dtype=torch.float32, device=DEVICE)
            if classes.size > 0:
                cw = compute_class_weight('balanced', classes=classes, y=all_y[alive])
                weight[classes.astype(int)] = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

            # Model / opt / loss
            model = TemporalLSTM(
                in_feats=len(FEATURE_COLS),
                lstm_hidden=D_LSTM_HID,
                lstm_layers=D_LSTM_LAY,
                mlp_hidden=D_MLP_HID,
                num_classes=OUT_CLASSES,
                dropout=DROP
            ).to(DEVICE)
            opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            crit = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)

            # Train
            for ep in range(1, EPOCHS+1):
                model.train()
                total_loss = 0.0
                for X_seq, y in pool:
                    opt.zero_grad(set_to_none=True)
                    logits = model(X_seq)
                    loss = crit(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    total_loss += float(loss.detach().cpu())

                if ep == 1 or ep % 10 == 0 or ep == EPOCHS:
                    model.eval()
                    with torch.no_grad():
                        logits_te = model(X_test)
                        preds_te = logits_te.argmax(dim=1).cpu().numpy()
                    m = alive_mask
                    if m.sum():
                        acc_te = accuracy_score(y_true_full[m], preds_te[m])
                        f1_te  = f1_score(y_true_full[m], preds_te[m], average='macro')
                        print(f"      [E{ep:02d}/{EPOCHS}] loss={total_loss/len(pool):.4f} | TEST ACC={acc_te:.3f} F1={f1_te:.3f}")
                    else:
                        print(f"      [E{ep:02d}/{EPOCHS}] loss={total_loss/len(pool):.4f} | no alive labels")

            # Final eval this run
            model.eval()
            with torch.no_grad():
                preds = model(X_test).argmax(dim=1).cpu().numpy()

            m = alive_mask
            if m.sum():
                acc  = accuracy_score(y_true_full[m], preds[m])
                prec = precision_score(y_true_full[m], preds[m], average='macro', zero_division=0)
                rec  = recall_score(y_true_full[m], preds[m], average='macro', zero_division=0)
                f1   = f1_score(y_true_full[m], preds[m], average='macro')
                report = classification_report(y_true_full[m], preds[m], digits=5, zero_division=0)
            else:
                acc = prec = rec = f1 = float('nan')
                report = "No alive labels for this quarter."

            # Save per-run report
            txt_path = os.path.join(out_dir, f"classification_report_{qn}_run{r+1}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"LSTM | {qn} | seed={seed} | epochs={EPOCHS}\n")
                f.write(f"ACC={acc:.6f} | PREC_macro={prec:.6f} | RECALL_macro={rec:.6f} | F1_macro={f1:.6f}\n\n")
                f.write(report)
            print(f"  Wrote TXT: {txt_path}")

            # Confusion matrix (normalized): title has NO run index
            if m.sum():
                cm = confusion_matrix(y_true_full[m], preds[m],
                                      labels=np.arange(OUT_CLASSES), normalize='true')
                disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(OUT_CLASSES)])
                fig = disp.plot(include_values=True, cmap='Blues', values_format=".2f", colorbar=False).figure_
                plt.title(f"{qn} — LSTM — Confusion Matrix")
                cm_path = os.path.join(out_dir, f"cm_{qn}_LSTM_run{r+1}.png")  # run only in filename
                fig.savefig(cm_path, dpi=220, bbox_inches='tight'); plt.close(fig)
                cms_norm.append(cm)
                print(f"  Wrote CM: {cm_path}")

            # rows
            run_rows.append({
                "quarter": qn, "seed": seed, "epochs": EPOCHS,
                "acc": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1
            })
            global_rows.append({
                "quarter": qn, "model": "LSTM", "seed": seed, "epochs": EPOCHS,
                "acc": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1
            })

            del model, opt, crit, pool
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save per-quarter runs.csv
        runs_df = pd.DataFrame(run_rows)
        runs_csv = os.path.join(out_dir, f"runs_{qn}.csv")
        runs_df.to_csv(runs_csv, index=False)
        print(f"Saved per-run metrics: {runs_csv}")

        # Save per-quarter mean/std
        if len(runs_df):
            agg = runs_df[["acc","precision_macro","recall_macro","f1_macro"]].agg(['mean','std'])
            agg_csv = os.path.join(out_dir, f"aggregate_metrics_{qn}.csv")
            agg.to_csv(agg_csv)
            print(f"Saved aggregate mean/std: {agg_csv}")

        # Save mean confusion matrix over runs
        if len(cms_norm) > 0:
            mean_cm = np.mean(np.stack(cms_norm, axis=0), axis=0)
            disp = ConfusionMatrixDisplay(mean_cm, display_labels=[str(i) for i in range(OUT_CLASSES)])
            fig = disp.plot(include_values=True, cmap='Blues', values_format=".2f", colorbar=False).figure_
            plt.title(f"{qn} — LSTM — Mean Confusion Matrix")
            mean_cm_path = os.path.join(out_dir, f"cm_{qn}_LSTM_mean.png")
            fig.savefig(mean_cm_path, dpi=220, bbox_inches='tight'); plt.close(fig)
            print(f"Saved mean CM: {mean_cm_path}")
        zip_path = shutil.make_archive(
        base_name=out_dir,                       # -> "./results/2018Q2_LSTM"
        format='zip',
        root_dir=os.path.dirname(out_dir),       # -> "./results"
        base_dir=os.path.basename(out_dir)       # -> "2018Q2_LSTM"
    )
    print(f"Zipped quarter folder: {zip_path}")

    # Global summary across all quarters (optional)
    if len(global_rows):
        pd.DataFrame(global_rows).to_csv(os.path.join(ROOT_DIR, "LSTM_walkforward_runs_all_quarters.csv"), index=False)
        print("Saved global summary: ./results/LSTM_walkforward_runs_all_quarters.csv")
