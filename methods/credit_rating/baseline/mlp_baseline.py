import os, glob, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from methods.credit_rating.utils.data_utils import (
    TARGET, _set_seed, _clean_id,
    enrich_nodes, make_feature_cols, make_preprocessor,
    build_dead_masks_up_to, build_label_for_next
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]     # repo root
DATASETS = ROOT / "datasets"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "rank_next_quarter"
NODES_PAT = str((DATASETS / "nodes" / "[0-9][0-9][0-9][0-9]Q[1-4].csv").resolve())
EDGES_PAT = str((DATASETS / "edges" / "edge_[0-9][0-9][0-9][0-9]Q[1-4].csv").resolve())

EPOCHS      = 100
LR          = 1e-3
WEIGHT_DECAY= 5e-4
D_MLP_HID   = 256
DROP        = 0.1
GRAD_CLIP   = 1.0

RUNS       = 4
BASE_SEED  = 42
SEQ_LEN    = 6

# ───────────── model ─────────────
class SimpleMLP(nn.Module):
    def __init__(self, in_feats, mlp_hidden=128, num_classes=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )
    def forward(self, x):  # x: [N, F]
        return self.net(x)

# ───────────── walk-forward ─────────────
def run_walkforward_mlp(start_quarter=None, end_quarter=None, runs=1, base_seed=42, EPOCHS=EPOCHS):
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

    FEATURE_COLS = make_feature_cols(usable[0][0])
    labels_flat = np.concatenate([df[TARGET].values-1 for df,_,_ in usable[1:]])
    OUT_CLASSES = int(np.unique(labels_flat).max() + 1)
    print(f"OUT_CLASSES = {OUT_CLASSES} | #features = {len(FEATURE_COLS)}")

    # range
    if start_quarter is None:
        start_idx = SEQ_LEN
    else:
        start_idx = max(quarters_list.index(start_quarter), SEQ_LEN)
    if end_quarter is None:
        end_idx = len(usable) - 1
    else:
        end_idx = quarters_list.index(end_quarter)

    ROOT_DIR = "./results_mlp"
    os.makedirs(ROOT_DIR, exist_ok=True)
    global_rows = []

    for t in range(start_idx, end_idx+1):
        qn = quarters_list[t]
        tag = f"{qn}_MLP"
        out_dir = os.path.join(ROOT_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== Target {qn} (t={t}) — running {runs} seeds ===")

        # Fit preprocessor on ≤ t-1
        pre_hist = pd.concat([usable[i][0][FEATURE_COLS] for i in range(0, t)], ignore_index=True)
        pre, _ = make_preprocessor(pre_hist)
        pre.fit(pre_hist)

        # Test features = quarter t
        df_t, _, _ = usable[t]
        X_te = pre.transform(df_t[FEATURE_COLS])
        y_test = build_label_for_next(usable, t-1, DEVICE)
        y_true_full = y_test.cpu().numpy()
        alive_mask = (y_true_full != -100)

        run_rows, cms_norm = [], []

        for r in range(runs):
            seed = base_seed + r
            _set_seed(seed)
            print(f"  -- run {r+1}/{runs} (seed={seed})")

            # Rolling window: last 6 quarters before t
            pool = []
            val_X, val_y = None, None
            start_e = max(SEQ_LEN, t - SEQ_LEN)
            for e in range(start_e, t):
                df_e, _, _ = usable[e]
                Xe = pre.transform(df_e[FEATURE_COLS])
                y_e = build_label_for_next(usable, e-1, DEVICE)

                if e == t-1:
                    # validation 10% of last train quarter
                    idx_train, idx_val = train_test_split(np.arange(len(y_e)), test_size=0.1, random_state=seed)
                    pool.append((torch.tensor(Xe[idx_train], dtype=torch.float32, device=DEVICE), y_e[idx_train]))
                    val_X, val_y = torch.tensor(Xe[idx_val], dtype=torch.float32, device=DEVICE), y_e[idx_val]
                else:
                    pool.append((torch.tensor(Xe, dtype=torch.float32, device=DEVICE), y_e))

            # Class weights
            all_y = np.concatenate([y.cpu().numpy() for _, y in pool])
            alive = (all_y != -100)
            classes = np.unique(all_y[alive])
            weight = torch.ones(OUT_CLASSES, dtype=torch.float32, device=DEVICE)
            if classes.size > 0:
                cw = compute_class_weight('balanced', classes=classes, y=all_y[alive])
                weight[classes.astype(int)] = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

            # Model / opt / loss
            model = SimpleMLP(len(FEATURE_COLS), mlp_hidden=D_MLP_HID,
                              num_classes=OUT_CLASSES, dropout=DROP).to(DEVICE)
            opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            crit = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)

            # Train
            for ep in range(1, EPOCHS+1):
                model.train()
                total_loss = 0.0
                for Xe, y in pool:
                    opt.zero_grad(set_to_none=True)
                    logits = model(Xe)
                    loss = crit(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                    total_loss += float(loss.detach().cpu())

                if ep==1 or ep%10==0 or ep==EPOCHS:
                    model.eval()
                    if val_X is not None:
                        with torch.no_grad():
                            val_preds = model(val_X).argmax(dim=1).cpu().numpy()
                            val_true = val_y.cpu().numpy()
                            m_val = (val_true != -100)
                            if m_val.sum():
                                val_acc = accuracy_score(val_true[m_val], val_preds[m_val])
                                val_f1  = f1_score(val_true[m_val], val_preds[m_val], average='macro')
                                print(f"      [E{ep:02d}/{EPOCHS}] VAL ACC={val_acc:.3f} F1={val_f1:.3f}")

            # Final eval
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_te, dtype=torch.float32, device=DEVICE)).argmax(dim=1).cpu().numpy()

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

            # Save report
            txt_path = os.path.join(out_dir, f"classification_report_{qn}_run{r+1}.txt")
            with open(txt_path,'w') as f:
                f.write(f"MLP | {qn} | seed={seed} | epochs={EPOCHS}\n")
                f.write(f"ACC={acc:.6f} | PREC_macro={prec:.6f} | RECALL_macro={rec:.6f} | F1_macro={f1:.6f}\n\n")
                f.write(report)
            print(f"  Wrote TXT: {txt_path}")

            if m.sum():
                cm = confusion_matrix(y_true_full[m], preds[m], labels=np.arange(OUT_CLASSES), normalize='true')
                disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(OUT_CLASSES)])
                fig = disp.plot(include_values=True, cmap='Blues', values_format=".2f", colorbar=False).figure_
                plt.title(f"{qn} — MLP — Confusion Matrix")
                cm_path = os.path.join(out_dir, f"cm_{qn}_MLP_run{r+1}.png")
                fig.savefig(cm_path, dpi=220, bbox_inches='tight'); plt.close(fig)
                cms_norm.append(cm)
                print(f"  Wrote CM: {cm_path}")

            run_rows.append({"quarter":qn,"seed":seed,"epochs":EPOCHS,
                             "acc":acc,"precision_macro":prec,"recall_macro":rec,"f1_macro":f1})
            global_rows.append({"quarter":qn,"model":"MLP","seed":seed,"epochs":EPOCHS,
                                "acc":acc,"precision_macro":prec,"recall_macro":rec,"f1_macro":f1})

            del model, opt, crit, pool
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        runs_df = pd.DataFrame(run_rows)
        runs_df.to_csv(os.path.join(out_dir, f"runs_{qn}.csv"), index=False)
        if len(runs_df):
            metrics = ["acc","precision_macro","recall_macro","f1_macro"]
            mean_vals = runs_df[metrics].mean().to_frame().T
            std_vals  = runs_df[metrics].std().to_frame().T
            mean_vals.index = ["mean"]
            std_vals.index  = ["std"]
            agg = pd.concat([mean_vals, std_vals])
            agg_csv = os.path.join(out_dir, f"aggregate_metrics_{qn}.csv")
            agg.to_csv(agg_csv)
            print(f"Saved aggregate mean/std: {agg_csv}")
        if len(cms_norm)>0:
            mean_cm = np.mean(np.stack(cms_norm, axis=0), axis=0)
            disp = ConfusionMatrixDisplay(mean_cm, display_labels=[str(i) for i in range(OUT_CLASSES)])
            fig = disp.plot(include_values=True, cmap='Blues', values_format=".2f", colorbar=False).figure_
            plt.title(f"{qn} — MLP — Mean Confusion Matrix")
            fig.savefig(os.path.join(out_dir, f"cm_{qn}_MLP_mean.png"), dpi=220, bbox_inches='tight'); plt.close(fig)

        shutil.make_archive(out_dir, 'zip', os.path.dirname(out_dir), os.path.basename(out_dir))

    if len(global_rows):
        pd.DataFrame(global_rows).to_csv(os.path.join(ROOT_DIR,"MLP_walkforward_runs_all_quarters.csv"), index=False)
