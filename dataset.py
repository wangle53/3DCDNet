import os
import torch
import os.path as osp
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import configs as cfg
import utils
from tqdm import tqdm
from process_data2accelerate import prepare_data, norm_data
from RandomSampler import generate_removed_plane_dataset
tcfg = cfg.CONFIGS['Train']
import time

class CDDataset(Dataset):
    def __init__(self, data_path, txt_path, n_samples, flag, ppdata_path,):
        super(CDDataset, self).__init__()
        self.data_path = data_path
        self.txt_path = txt_path
        self.n_samples = n_samples
        self.flag = flag
        self.ppdata_path = ppdata_path
        if self.flag=='train' or self.flag=='test':
            if tcfg.remove_plane:
                generate_removed_plane_dataset(tcfg.path.data_root)
            if tcfg.if_prepare_data:
                prepare_data()
        with open(self.txt_path, 'r') as f:
            self.list = f.readlines()
            self.file_size = len(self.list)
            
    def __getitem__(self, idx):
        if not tcfg.if_prepare_data:
            p16_path = osp.join(self.data_path, self.list[idx].split(' ')[0])
            p20_path = osp.join(self.data_path, self.list[idx].split(' ')[1].strip())
            dir_name = self.list[idx].split(' ')[0].split('\\')[0]
            pc0_name = self.list[idx].split(' ')[0].split('\\')[-1]
            pc1_name = self.list[idx].split(' ')[1].split('\\')[-1].strip()
            p16, p20, p16_raw_length, p20_raw_length = utils.align_length(p16_path, p20_path, self.n_samples)
            p16_data = p16[:, :-1]; p20_data = p20[:, :-1];
            if tcfg.norm_data:
               p16_data, p20_data, p16_raw, p20_raw = norm_data(p16_data, p20_data)
            label16, label20 = self.generate_label(p16, p20)  

            batch_data16 = self.process_data(p16_data)   
            batch_data20 = self.process_data(p20_data) 
            
            p16ofp20 = utils.search_k_neighbors(p20_data[:, :3], p16_data[:, :3], tcfg.k_neighbors)
            p20ofp16 = utils.search_k_neighbors(p16_data[:, :3], p20_data[:, :3], tcfg.k_neighbors)
                 
            inputs16 = {}; inputs20 = {};
            inputs16['xyz'] = [torch.from_numpy(data).float() for data in batch_data16[0]]
            inputs16['neighbors_idx'] = [torch.from_numpy(data).long() for data in batch_data16[1]]
            inputs16['pool_idx'] = [torch.from_numpy(data).long() for data in batch_data16[2]]
            inputs16['unsam_idx'] = [torch.from_numpy(data).long() for data in batch_data16[3]]
            inputs16['label'] = torch.from_numpy(label16).long()
            inputs16['knearst_idx_in_another_pc'] = torch.from_numpy(p16ofp20).long()
            inputs16['raw_length'] = p16_raw_length
            inputs20['xyz'] = [torch.from_numpy(data).float() for data in batch_data20[0]]
            inputs20['neighbors_idx'] = [torch.from_numpy(data).long() for data in batch_data20[1]]
            inputs20['pool_idx'] = [torch.from_numpy(data).long() for data in batch_data20[2]]
            inputs20['unsam_idx'] = [torch.from_numpy(data).long() for data in batch_data20[3]]
            inputs20['label'] = torch.from_numpy(label20).long()
            inputs20['knearst_idx_in_another_pc'] = torch.from_numpy(p20ofp16).long()
            inputs20['raw_length'] = p20_raw_length
        
            return inputs16, inputs20, dir_name, pc0_name, pc1_name, [p16_raw, p20_raw]
        
        else:
            ppdata = np.load(os.path.join(self.ppdata_path, self.flag, str(idx)+'.npy'), allow_pickle=True)
            inputs16, inputs20, dir_name, pc0_name, pc1_name, [p16_raw, p20_raw] = ppdata
            if not tcfg.use_rgb:
                inputs16['xyz'] = [i[:, :3] for i in inputs16['xyz']]
                inputs20['xyz'] = [i[:, :3] for i in inputs20['xyz']]
            return inputs16, inputs20, dir_name, pc0_name, pc1_name, [p16_raw, p20_raw]
        
    def __len__(self):
        return self.file_size
        
    def process_data(self, pc_data, subsam_rate=tcfg.sub_sampling_ratio):
        if pc_data.shape[1] == 3:
            xyz = pc_data
        else:
            xyz = pc_data[:, :3]
        input_points = []
        input_points.append(pc_data)
        neighbors_idx = []
        pool_idx = []
        upsam_idx = []
        for i in range(tcfg.num_layers):
            k_neigh_idx = utils.search_k_neighbors(xyz, xyz, tcfg.k_neighbors)
            sub_pc_data = pc_data[:pc_data.shape[0]//subsam_rate[i], :]
            sub_xyz = xyz[:xyz.shape[0]//subsam_rate[i], :]
            sub_idx = k_neigh_idx[:pc_data.shape[0]//subsam_rate[i], :]
            up_idx = utils.search_k_neighbors(sub_xyz, xyz,  1)
            input_points.append(sub_pc_data)
            neighbors_idx.append(k_neigh_idx)
            pool_idx.append(sub_idx)
            upsam_idx.append(up_idx)
            pc_data = sub_pc_data
            xyz = sub_xyz
        inputs_list = [input_points, neighbors_idx, pool_idx, upsam_idx]
        
        return inputs_list
    
    def generate_label(self, p16, p20):        
        label16 = np.expand_dims(p16[:, -1], 1)
        label20 = np.expand_dims(p20[:, -1], 1)
        return label16, label20
    
if __name__ == '__main__':
    
    train_data = CDDataset(tcfg.path['train_dataset'], tcfg.path['train_txt'], tcfg.n_samples, 'train', tcfg.path.prepare_data)
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=False)
    for i, data in enumerate(train_dataloader):
        data0, data1, _, _, _, _ = data

        print(data0['xyz'][0].shape, 
              data0['neighbors_idx'][0].shape, 
              data0['pool_idx'][0].shape, 
              data0['unsam_idx'][0].shape, 
              data0['label'].shape, 
              data0['raw_length'])
        xyz = data0['xyz']
        for j in range(5):
            print(xyz[j].shape)
        break
"""
output:
torch.Size([2, 8192, 3]) torch.Size([2, 8192, 16]) torch.Size([2, 2048, 16]) torch.Size([2, 8192, 1]) torch.Size([2, 8192, 1]) tensor([8192, 8192])
torch.Size([2, 8192, 3])
torch.Size([2, 2048, 3])
torch.Size([2, 512, 3])
torch.Size([2, 128, 3])
torch.Size([2, 32, 3])

"""