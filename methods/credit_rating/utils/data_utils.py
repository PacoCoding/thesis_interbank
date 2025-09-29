import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from sklearn.preprocessing import FunctionTransformer, RobustScaler
TARGET = "rank_next_quarter"

def _set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ───────────── helpers ─────────────
def _clean_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return s

def enrich_nodes(nodes_df, edges_df):
    df = nodes_df.copy()
    if 'BankID' not in df.columns:
        df.insert(0, 'BankID', df.index.astype(str))
    else:
        df['BankID'] = df['BankID'].astype(str)

    e = edges_df.copy()
    e['Sourceid'] = _clean_id(e['Sourceid'])
    e['Targetid'] = _clean_id(e['Targetid'])

    in_deg   = e.groupby('Targetid').size().rename('in_degree')
    out_deg  = e.groupby('Sourceid').size().rename('out_degree')
    in_wdeg  = e.groupby('Targetid')['Weights'].sum().rename('in_wdeg')
    out_wdeg = e.groupby('Sourceid')['Weights'].sum().rename('out_wdeg')

    nf = (pd.concat([in_deg, out_deg, in_wdeg, out_wdeg], axis=1)
           .fillna(0).reset_index().rename(columns={'index':'BankID'}))
    nf['BankID'] = nf['BankID'].astype(str)

    merged = pd.merge(df, nf, on='BankID', how='left').fillna(0)
    if TARGET in df.columns:
        merged[TARGET] = df[TARGET].values
    return merged

def make_feature_cols(sample_df: pd.DataFrame) -> list[str]:
    num = sample_df.select_dtypes(include='number').columns
    drop = set([TARGET, "index"])
    drop |= {c for c in num if "srisk" in c.lower()}  # drop any srisk* columns
    feats = [c for c in num if c not in drop]
    return feats

# ───────────── preprocessing (data-driven log1p router) ─────────────
def pick_log1p_columns(df, skew_thr=1.0, q3_q1_ratio=10.0, max_med_ratio=100.0):
    num = df.select_dtypes(include='number')
    q1   = num.quantile(0.25)
    q3   = num.quantile(0.75)
    med  = num.median().replace(0, np.nan)
    mn   = num.min()
    mx   = num.max()
    skew = num.skew(numeric_only=True)

    nonneg     = (mn >= 0)
    big_spread = ((q3 / q1.replace(0, np.nan)) > q3_q1_ratio) | ((mx / med) > max_med_ratio)
    right_skew = (skew > skew_thr)
    many_zeros = (q1.eq(0) & med.fillna(0).eq(0))

    mask = nonneg & (right_skew | big_spread | many_zeros)
    return list(num.columns[mask])

def make_preprocessor(df_train: pd.DataFrame):
    log_cols  = pick_log1p_columns(df_train)
    rest_cols = [c for c in df_train.select_dtypes(include='number').columns if c not in log_cols]

    log_branch = Pipeline([
        ("clip_nonneg", FunctionTransformer(lambda X: np.clip(X, 0, None), validate=False)),
        ("log1p",       FunctionTransformer(np.log1p, validate=False)),
        ("robust",      RobustScaler()),
    ])
    rest_branch = Pipeline([("robust", RobustScaler())])

    pre = ColumnTransformer(
        transformers=[
            ("log",  log_branch,  log_cols),
            ("rest", rest_branch, rest_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, {"log1p": log_cols, "robust_only": rest_cols}

# ───────────── dead masks & labels (no leakage) ─────────────
def build_dead_masks_up_to(usable, end_idx):
    masks = []
    persistent = set()
    for i in range(end_idx+1):
        df_i, _, _ = usable[i]
        newly = list(df_i.loc[df_i['Equity'] < 0, 'BankID'])
        persistent |= set(newly)
        dead_i = np.isin(df_i['BankID'], list(persistent))
        masks.append(dead_i)
    return masks

def build_label_for_next(usable, k, device):
    df_k, _, _  = usable[k]
    df_n, _, _  = usable[k+1]
    newly_k  = list(df_k.loc[df_k['Equity'] < 0, 'BankID'])
    newly_n  = list(df_n.loc[df_n['Equity'] < 0, 'BankID'])
    pf = set(newly_k) | set(newly_n)
    y = df_n[TARGET].values - 1
    dead = np.isin(df_n['BankID'], list(pf))
    y[dead] = -100
    return torch.tensor(y, dtype=torch.long, device=device)
