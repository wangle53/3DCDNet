from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
import numpy as np
import configs as cfg
tcfg = cfg.CONFIGS['Train']

def gather_neighbour(pc, neighbor_idx):
    """
    gather the coordinates or features of neighboring points
    pc: [B, C, N, 1]
    neighbor_idx: [B, N, K]
    """
    pc = pc.transpose(2, 1).squeeze(-1)
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  #b* n *k *d
    features = features.permute(0, 3, 1, 2) # b*c*n*k
    return features
 
class SPE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        
        
    def forward(self, feature, neigh_idx):
        f_neigh = gather_neighbour(feature, neigh_idx)
        f_neigh = torch.cat((feature, f_neigh), -1)
        f_agg2 = self.mlp2(f_neigh)
        f_agg2 = torch.sum(f_agg2, -1, keepdim=True)
        return f_agg2

class LFE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp3 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
   
    def forward(self, feature, neigh_idx):
        f_neigh = gather_neighbour(feature, neigh_idx)
        f_neigh = self.mlp1(f_neigh)
        f_neigh = torch.sum(f_neigh, dim=-1, keepdim=True)
        f_neigh = self.mlp2(f_neigh)
        feature = self.mlp3(feature)
        f_agg = f_neigh + feature
        return f_agg

class LFA(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.spe = SPE(d_in, d_out)
        self.lfe = LFE(d_in, d_out)
        self.mlp = pt_utils.Conv2d(d_out, d_out, kernel_size=(1, 1), bn=True)
     
    def forward(self, feature, neigh_idx):
        spe = self.spe(feature, neigh_idx)
        lfe = self.lfe(feature, neigh_idx)
        f_agg = spe + lfe
        f_agg = self.mlp(f_agg)
        return f_agg

    
if __name__ == '__main__':

    xyz = Variable(torch.rand(4,1024,3))
    feature = Variable(torch.rand(4,64,1024,1))
    nidx = np.random.randint(0,1024,size=[4,1024,16])
    nidx = torch.Tensor(nidx).type(torch.int64)
    net = LFA(64, 128)
    y = net(feature, nidx)
    print('out', y.shape)

    