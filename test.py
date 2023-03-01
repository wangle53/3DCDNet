#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

import utils as utl
import configs as cfg
import metrics as mc
from net import Siam3DCDNet
from dataset import CDDataset
from tqdm import tqdm


def test_network(tcfg):
    test_txt = tcfg.path['test_txt']
    test_data = CDDataset(tcfg.path['test_dataset'], tcfg.path['test_txt'], tcfg.n_samples, 'test', tcfg.path.prepare_data)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    best_model_path = os.path.join(tcfg.path['best_weights_save_dir'], 'best_net.pth')
    pretrained_dict = torch.load(best_model_path)['model_state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Siam3DCDNet(tcfg.in_dim, tcfg.out_dim).to(device=device)
    net.load_state_dict(pretrained_dict,False)
    torch.no_grad()
    net.eval()
    dur = 0
    iou_calc = mc.IoUCalculator()
    tqdm_loader = tqdm(test_dataloader, total=len(test_dataloader))
    for _, data in enumerate(tqdm_loader):
        batch_data0, batch_data1, dir_name, pc0_name, pc1_name, raw_data = data   
        p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx, lb0, knearest_idx0, raw_length0 = [i for i in batch_data0.values()]
        p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx, lb1, knearest_idx1, raw_length1 = [i for i in batch_data1.values()]
        p0 = [i.to(device, dtype=torch.float) for i in p0]
        p0_neighbors_idx = [i.to(device, dtype=torch.long) for i in p0_neighbors_idx]
        p0_neighbors_idx = [i.to(device, dtype=torch.long) for i in p0_neighbors_idx]
        p0_pool_idx = [i.to(device, dtype=torch.long) for i in p0_pool_idx]
        p0_unsam_idx = [i.to(device, dtype=torch.long) for i in p0_unsam_idx]
        p1 = [i.to(device, dtype=torch.float) for i in p1]
        p1_neighbors_idx = [i.to(device, dtype=torch.long) for i in p1_neighbors_idx]
        p1_pool_idx = [i.to(device, dtype=torch.long) for i in p1_pool_idx]
        p1_unsam_idx = [i.to(device, dtype=torch.long) for i in p1_unsam_idx]
        knearest_idx = [knearest_idx0.to(device, dtype=torch.long), knearest_idx1.to(device, dtype=torch.long)]
        
        lb0 = lb0.squeeze(-1).to(device, dtype=torch.long)
        lb1 = lb1.squeeze(-1).to(device, dtype=torch.long)
        t0 = time.time()
        out0, out1 = net([p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx],
                              [p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx], 
                              knearest_idx)
        dur += time.time()-t0
        out0 = out0.max(dim=-1)[1]; out1 = out1.max(dim=-1)[1];
        
        iou_calc.add_data(out0.squeeze(0), out1.squeeze(0), lb0.squeeze(0), lb1.squeeze(0))
        if tcfg.save_prediction:
            utl.save_prediction3(raw_data[0], raw_data[1], 
                                 lb0, lb1, 
                                 out0.squeeze(-1), out1.squeeze(-1), 
                                 os.path.join(tcfg.path['test_prediction_PCs'], str(dir_name[0])),
                                 pc0_name, pc1_name,
                                 tcfg.path['test_dataset'])
            
    iou = iou_calc.metrics()
    for k, v in iou.items():
        print(k, v)    
    with open(os.path.join(tcfg.path['outputs'], 'test_IoU.txt'),'a') as f:
        f.write('Time:{},miou:{:.6f},oa:{:.6f},iou_list:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
                iou['miou'], iou['oa'], iou['iou_list']))
        f.write('\n')   
    print('FPS: ', len(test_dataloader)/dur)
    
    
if __name__ == '__main__':
    
    tcfg = cfg.CONFIGS['Train']
    test_network(tcfg)
    
    
    
    
    