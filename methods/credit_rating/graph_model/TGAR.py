import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv,GATv2Conv # GCN
from tqdm import tqdm

 
class Model():
    def __init__(self, model, args, device, class_weight=None):
        self.device = device
        self.args = args
        self.model = model.to(device)
        # store weights on the correct device
        self.class_weight = None if class_weight is None else class_weight.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=5e-4
        )

    def fit(self, batch):
        self.optimizer.zero_grad()
        out = self.model(batch)  # IMPORTANT: nets return log-probs (log_softmax)
        loss = F.nll_loss(
            out[:batch.batch_size],
            batch.y[:batch.batch_size],
            weight=self.class_weight  # <-- weighted loss here
        )
        loss.backward()
        self.optimizer.step()

    def test(self, testbatch):
        self.model.eval()
        out = self.model(testbatch)
        predb = out.cpu().max(dim=1).indices[:testbatch.batch_size]
        trub = torch.Tensor.cpu(testbatch.y)[:testbatch.batch_size]
        self.model.train()
        return trub, predb

class TGAR(torch.nn.Module):
    def __init__(self, batch_size, num_feature, num_label, hiddim, droprate,hidlayers,p, hyper_k):
        super(TGAR, self).__init__()
        self.model_name = "TGAR" 
        # hyper parameters
        self.batch_size = batch_size
        self.num_feature = num_feature
        self.num_label = num_label
        self.hiddim = hiddim 
        self.hidlayers = hidlayers
        self.dropout = torch.nn.Dropout(p=droprate)
        self.hyper_k = hyper_k

        # layers

        self.fcs = nn.ModuleList()

        # hyper-feature transition
        self.GATConv11 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv12 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv13 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv21 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv22 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        self.GATConv23 = GATConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k)
        
        self.TransConv1 = TransformerConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k, edge_dim=1)
        self.TransConv2 = TransformerConv(in_channels = hiddim, out_channels = hiddim // hyper_k, heads = hyper_k, edge_dim=1)


        # fcnn layer 4 * hiddim    hiddim represents m in paper
        for i in range(4):
            self.fcs.append(nn.Linear(self.num_feature, self.hiddim))

        self.fcs.append(nn.Linear(self.hiddim * 3, 1))

        self.fcsK = nn.Linear(self.hiddim, self.num_feature)

        # fcnn layer 4 * hiddim    hiddim represents m in paper
        for i in range(4):
            self.fcs.append(nn.Linear(self.num_feature, self.hiddim))

        # Gain todo: dim?
        self.fcs.append(nn.Linear(self.hiddim * 3, 1))

        # todo: output layer
        self.head = nn.Sequential(
            nn.Linear(self.hiddim, self.hiddim // 2),
            nn.ReLU(),
            nn.Dropout(p=droprate),
            nn.Linear(self.hiddim // 2, self.num_label),
        )
    def forward(self, data):
        # NeighborLoader gives edge_attr for the *batch* edges with shape [E_batch, 1]
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            edge_attr = edge_attr.clamp(min=0)  # safety; already clipped upstream
        K = self.encode(data.x, data.edge_index, edge_attr)
        logits = self.head(K)
        return F.log_softmax(logits, dim=1)

    def encode(self, x, edge_index, edge_attr):
        x = x.view(-1, x.size(-1))
        K = self.contextAttentionLayer(x, edge_index, edge_attr, 0)
        K = self.fcsK(K)
        K = self.contextAttentionLayer(K, edge_index, edge_attr, 1)
        return K
    # differential aggregation operator
    def diffAggr(self, X1, X2):
        concatenated = torch.cat([X1, X2, X1 - X2], dim=1) 
        return concatenated
    

    def contextAttentionLayer(self, x, edge_index, edge_attr, layer_num):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)

        innerLayerNum = 5
        x1 = self.dropout(F.relu(self.fcs[0 + layer_num * innerLayerNum](x)))
        x2 = self.dropout(F.relu(self.fcs[1 + layer_num * innerLayerNum](x)))
        x3 = self.dropout(F.relu(self.fcs[2 + layer_num * innerLayerNum](x)))
        x4 = self.dropout(F.relu(self.fcs[3 + layer_num * innerLayerNum](x)))

        # GATConv — no edge weights
        if layer_num == 0:
            x1 = self.GATConv11(x1, edge_index)
            x2 = self.GATConv12(x2, edge_index)
            x3 = self.GATConv13(x3, edge_index)
        else:
            x1 = self.GATConv21(x1, edge_index)
            x2 = self.GATConv22(x2, edge_index)
            x3 = self.GATConv23(x3, edge_index)

        x1 = self.dropout(F.relu(x1))
        x2 = self.dropout(F.relu(x2))
        x3 = self.dropout(F.relu(x3))

        output_Fc = torch.mul(x1, x2)
        output_F  = x3

        # TransformerConv — pass edge_attr directly (shape [E_batch, 1])
        if edge_attr is not None:
            ea = edge_attr.to(device)
            # (optional) sanity checks:
            # assert edge_index.size(1) == ea.size(0)
            # assert ea.dim() == 2 and ea.size(1) == 1
        else:
            ea = None

        if layer_num == 0:
            m = self.TransConv1(output_Fc, edge_index, edge_attr=ea)
        else:
            m = self.TransConv2(output_Fc, edge_index, edge_attr=ea)

        m = self.dropout(F.relu(m))
        m = torch.softmax(m, dim=1)

        z = torch.mul(m, output_F)
        G = self.fcs[4 + layer_num * innerLayerNum](self.diffAggr(z, x4)).to(device)
        G = self.dropout(torch.sigmoid(G))
        Y = F.leaky_relu(G * z + (1 - G) * x4, 0.25)
        return Y
