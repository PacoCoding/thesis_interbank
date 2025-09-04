import torch
import numpy as np


def load_data(year,Q,attack_rate=0):
    if Q==1:
        if attack_rate!=0:
            cites1 = "datasets/edges"+str(attack_rate)+"/edge_"+str(int(year)-1)+"Q3.csv"
            content1 = "datasets/nodes"+str(attack_rate)+"/"+str(int(year)-1)+"Q3.csv"
        else:
            cites1 = "datasets/edges/edge_"+str(year-1)+"Q3.csv"
            content1 = "datasets/nodes/"+str(year-1)+"Q3.csv"

        cites2 = "datasets/edges/edge_"+str(int(year)-1)+"Q4.csv"
        content2 = "datasets/nodes/"+str(int(year)-1)+"Q4.csv"
    elif Q==2:
        if attack_rate!=0:
            cites1 = "datasets/edges"+str(attack_rate)+"/edge_"+str(int(year)-1)+"Q4.csv"
            content1 = "datasets/nodes"+str(attack_rate)+"/"+str(int(year)-1)+"Q4.csv"
        else:
            cites1 = "datasets/edges/edge_"+str(int(year)-1)+"Q4.csv"
            content1 = "datasets/nodes/"+str(int(year)-1)+"Q4.csv"

        cites2 = "datasets/edges/edge_"+str(year)+"Q1.csv"
        content2 = "datasets/nodes/"+str(year)+"Q1.csv"
    else:
        if attack_rate!=0:
            cites1 = "datasets/edges"+str(attack_rate)+"/edge_"+str(year)+"Q"+str(int(Q)-2)+".csv"
            content1 = "datasets/nodes"+str(attack_rate)+"/"+str(year)+"Q"+str(int(Q)-2)+".csv"
        else:
            cites1 = "datasets/edges/edge_"+str(year)+"Q"+str(int(Q)-2)+".csv"
            content1 = "datasets/nodes/"+str(year)+"Q"+str(int(Q)-2)+".csv"

        cites2 = "datasets/edges/edge_"+str(year)+"Q"+str(int(Q)-1)+".csv"
        content2 = "datasets/nodes/"+str(year)+"Q"+str(int(Q)-1)+".csv"

    
    index_dict = dict()
    label_to_index = dict()

    features = []
    labels = []
    dead_mask = []          # <-- NEW: will become a bool mask over nodes
    n1 = 0                  # number of nodes in first quarter (content1)

    with open(content1, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        # robustly locate Equity and label column
        # Equity name may vary in case/spaces; adjust if your header differs
        eq_idx = next(i for i, h in enumerate(header) if h.strip().lower() == "equity")
        # your label can be last or third from last; keep your logic:
        for j, node in enumerate(lines[1:]):
            node_info = node.strip('\n').split(',')
            node_id = node_info[0]
            index_dict[node_id] = len(index_dict)

            # features (same as yours)
            if len(node_info) > 72:
                feats = [float(i) for i in node_info[1:-3]]
                label_str = node_info[-3]
            else:
                feats = [float(i) for i in node_info[1:-1]]
                label_str = node_info[-1]
            features.append(feats)

            # dead if Equity < 0
            eq_val = float(node_info[eq_idx])
            dead_mask.append(eq_val < 0.0)

            if label_str not in label_to_index:
                label_to_index[label_str] = len(label_to_index)
            labels.append(label_to_index[label_str])

    n1 = len(index_dict)              # <-- dynamic offset
    # ---------- content2 ----------
    with open(content2, "r") as f:
        lines = f.readlines()
        header2 = lines[0].strip().split(',')
        eq_idx2 = next(i for i, h in enumerate(header2) if h.strip().lower() == "equity")
        for node in lines[1:]:
            node_info = node.strip('\n').split(',')
            node_id = str(int(node_info[0]) + n1)   # <-- offset by n1 instead of 24271
            index_dict[node_id] = len(index_dict)

            if len(node_info) > 72:
                feats = [float(i) for i in node_info[1:-3]]
                label_str = node_info[-3]
            else:
                feats = [float(i) for i in node_info[1:-1]]
                label_str = node_info[-1]
            features.append(feats)

            eq_val = float(node_info[eq_idx2])
            dead_mask.append(eq_val < 0.0)

            if label_str not in label_to_index:
                label_to_index[label_str] = len(label_to_index)
            labels.append(label_to_index[label_str])

    # ---------- edges (drop any incident to dead nodes) ----------
    edge_index = []
    dead_arr = np.array(dead_mask, dtype=bool)

    def _add_edges(path, offset=0, has_weight=True):
        with open(path, "r") as f:
            lines = f.readlines()
            for k, line in enumerate(lines[1:]):
                parts = line.strip('\n').split(',')
                if attack_rate != 0 or not has_weight:
                    start, end = parts[0], parts[1]
                else:
                    start, end = parts[0], parts[1]  # weight ignored here
                sid = str(int(start) + offset)
                tid = str(int(end) + offset)
                if sid in index_dict and tid in index_dict:
                    s = index_dict[sid]
                    t = index_dict[tid]
                    # skip if either endpoint is dead
                    if dead_arr[s] or dead_arr[t]:
                        continue
                    edge_index.append([s, t])
                    edge_index.append([t, s])

    # edges for content1 (offset 0)
    _add_edges(cites1, offset=0, has_weight=(attack_rate == 0))
    # edges for content2 (offset n1)
    _add_edges(cites2, offset=n1, has_weight=True)

    labels   = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edge_index = torch.LongTensor(edge_index)
    dead_mask = torch.tensor(dead_arr, dtype=torch.bool)     # <-- NEW

    return label_to_index, labels, features, edge_index, dead_mask

def preprocess():
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mask = torch.arange(9096)
    train_mask = mask[:4548]
    test_mask = mask[4548:]
    return train_mask,test_mask
