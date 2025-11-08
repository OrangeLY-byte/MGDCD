import torch
import torch.nn as nn
import torch.nn.functional as F


class  Spat_Cau_Encoder(nn.Module):
    def __init__(self,n_sub, n_nodes,in_dim,hid_dim):
        super().__init__()
        self.num_nodes = n_nodes
        self.num_sub = n_sub
        self.A_c = nn.Parameter(torch.empty(n_sub,n_nodes, n_nodes))
        torch.nn.init.xavier_uniform_(self.A_c)
        self.A_s = nn.Parameter(torch.empty(n_sub,n_nodes, n_nodes))
        torch.nn.init.xavier_uniform_(self.A_s)
        self.common= nn.Conv2d(in_channels=n_sub, out_channels=n_sub*3, kernel_size=1, stride=1, padding=0)
        self.fc_sin = nn.ModuleList([nn.Linear(in_dim, hid_dim) for _ in range(n_sub)])
        self.fc_sout = nn.ModuleList([nn.Linear(in_dim, hid_dim) for _ in range(n_sub)])
        self.fc_cin = nn.ModuleList([nn.Linear(in_dim, hid_dim) for _ in range(n_sub)])
        self.fc_cout = nn.ModuleList([nn.Linear(in_dim, hid_dim) for _ in range(n_sub)])

    def forward(self, inputs, i):
        all_c = []
        all_s = []
        A_c = F.tanh(self.A_c)
        A_s = F.relu(self.A_s)
        for j in range(self.num_sub):
            batch_data_j = inputs[j][i]
            x, y, t = batch_data_j
            x = x.squeeze(-1)
            x_j_c = A_c[j,:,:]@x
            x_j_s = A_s[j,:,:]@x
            causal_j_c = F.tanh(self.fc_cout[j](x_j_c)) @ F.tanh(self.fc_cin[j](x_j_c)).permute(0, 2, 1)
            causal_j_s = F.tanh(self.fc_sout[j](x_j_s)) @ F.tanh(self.fc_sin[j](x_j_s)).permute(0, 2, 1)
            all_c.append(causal_j_c)
            all_s.append(causal_j_s)
        causal_c_all = torch.stack(all_c, dim=1)
        causal_c = self.common(causal_c_all)
        causal_s = torch.stack(all_s, dim=1)
        return causal_s, causal_c.mean(dim=1).squeeze(1), causal_c_all

class  SCE_single(nn.Module):
    def __init__(self,n_nodes,in_dim,hid_dim):
        super().__init__()
        self.num_nodes = n_nodes
        self.A = nn.Parameter(torch.empty(n_nodes, n_nodes))
        torch.nn.init.xavier_uniform_(self.A)
        self.fc_in =nn.Linear(in_dim, hid_dim)
        self.fc_out = nn.Linear(in_dim, hid_dim)

    def forward(self, inputs, i):
        A = F.tanh(self.A)
        batch_data = inputs[i]
        x, y, t = batch_data
        x = x.squeeze(-1)
        x = A @ x
        causal = F.tanh(self.fc_in(x)) @ F.tanh(self.fc_out(x)).permute(0, 2, 1)
        return torch.sigmoid(causal)