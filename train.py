#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import time
import visdom
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
 
import utils as utl
import configs as cfg
import metrics as mc
from net import Siam3DCDNet
from dataset import CDDataset
import test


def train_network(tcfg, vis):
    
    utl.save_cfg(tcfg, tcfg.path['outputs'])
    weights_save_dir = tcfg.path['weights_save_dir']
    
    init_epoch = 0
    best_metric = -0.001
    total_steps = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    train_txt = tcfg.path['train_txt']
    val_txt = tcfg.path['val_txt']
    train_data = CDDataset(tcfg.path['train_dataset'], tcfg.path['train_txt'], tcfg.n_samples, 'train', tcfg.path.prepare_data)
    train_dataloader = DataLoader(train_data, batch_size=tcfg.batch_size, shuffle=True)
    val_data = CDDataset(tcfg.path['val_dataset'], tcfg.path['val_txt'], tcfg.n_samples, 'val', tcfg.path.prepare_data)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    if not tcfg.resume:
        if  torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(Siam3DCDNet(tcfg.in_dim, tcfg.out_dim)).to(device)
        else:
            net = Siam3DCDNet(tcfg.in_dim, tcfg.out_dim).to(device)
    
    if tcfg.resume:
        net = Siam3DCDNet(tcfg.in_dim, tcfg.out_dim).to(device)
        assert os.path.exists(os.path.join(weights_save_dir, 'current_net.pth')), 'There is not found any saved weights'
        print("\nLoading pre-trained networks...")
        init_epoch = torch.load(os.path.join(weights_save_dir, 'current_net.pth'))['epoch']
        net.load_state_dict(torch.load(os.path.join(weights_save_dir, 'current_net.pth'))['model_state_dict'])
        with open(os.path.join(tcfg.path['outputs'], 'val_metric.txt')) as f:
            lines = f.readlines()
            best_metric = float(lines[-1].strip().split(':')[-1])
        print("\tDone.\n")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net).to(device)
        
    weight = torch.from_numpy(np.array(2.0)).to(device)
    bce = nn.BCELoss(weight=weight)
    optimizer = optim.Adam(net.parameters(), lr=tcfg.optimizer['lr'], betas=(0.5, 0.999))
#     optimizer = optim.SGD(net.parameters(), lr=tcfg.optimizer['lr'], momentum=tcfg.optimizer['momentum'], weight_decay=tcfg.optimizer['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=tcfg.optimizer['lr_step_size'], gamma=tcfg.optimizer['gamma'])
    start_time = time.time()
    for epoch in range(init_epoch, tcfg.epoch):
        loss = []
        net.train()
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):
            
            batch_data0, batch_data1, _, _, _, _ = data
            
            p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx, lb0, knearest_idx0, _ = [i for i in batch_data0.values()]
            p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx, lb1, knearest_idx1, _ = [i for i in batch_data1.values()]
            p0 = [i.to(device) for i in p0]
            p0_neighbors_idx = [i.to(device) for i in p0_neighbors_idx]
            p0_pool_idx = [i.to(device) for i in p0_pool_idx]
            p0_unsam_idx = [i.to(device) for i in p0_unsam_idx]
            p1 = [i.to(device) for i in p1]
            p1_neighbors_idx = [i.to(device) for i in p1_neighbors_idx]
            p1_pool_idx = [i.to(device) for i in p1_pool_idx]
            p1_unsam_idx = [i.to(device) for i in p1_unsam_idx]
            knearest_idx0 = knearest_idx0.to(device)
            knearest_idx1 = knearest_idx1.to(device)
            knearest_idx = [knearest_idx0, knearest_idx1]
                
            lb0 = lb0.squeeze(-1).to(device, dtype=torch.long)
            lb1 = lb1.squeeze(-1).to(device, dtype=torch.long)
            
            fused_lb0 = torch.max(lb0, torch.gather(lb1, 1, knearest_idx0[:,:,0].squeeze(-1)))
            fused_lb1 = torch.max(lb1, torch.gather(lb0, 1, knearest_idx1[:,:,0].squeeze(-1)))
            
            epoch_iter += tcfg.batch_size
            total_steps += tcfg.batch_size
            # forward
            out0, out1 = net([p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx],
                              [p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx],
                              knearest_idx)
            err = F.nll_loss(out0.reshape(-1, 2), lb0.reshape(-1)) + F.nll_loss(out1.reshape(-1, 2), lb1.reshape(-1))
            # backward
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            
            errors = utl.get_errors(err)
            loss.append(err.item())
              
            counter_ratio = float(epoch_iter) / len(train_dataloader.dataset)
            if (i % tcfg.print_rate == 0 and i>0):
                print('Time:{},epoch:{},iteration:{},loss:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i, np.mean(loss)))
                with open(os.path.join(tcfg.path['outputs'],'train_loss.txt'),'a') as f:
                    f.write('Time:{},epoch:{}, iteration:{}, loss:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i, np.mean(loss)))
                    f.write('\n')      
                if tcfg.display:
                    utl.plot_current_errors(epoch, counter_ratio, errors, vis)
            utl.mkdir(weights_save_dir)
            utl.save_weights(epoch, net, optimizer, weights_save_dir, 'net')
        scheduler.step()
        duration = time.time() - start_time
        print('training duration: {}, lr: {}'.format(duration, optimizer.state_dict()['param_groups'][0]['lr']))
            
            
        # val_phase
        print('Validationg................')
        with net.eval() and torch.no_grad():      
            TP = 0
            FN = 0
            FP = 0
            TN = 0     
            iou_calc = mc.IoUCalculator()  
            for k, data in enumerate(val_dataloader):
                
                batch_data0, batch_data1, dir_name, pc0_name, pc1_name, raw_data = data
                p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx, lb0, knearest_idx0, raw_length0 = [i for i in batch_data0.values()]
                p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx, lb1, knearest_idx1, raw_length1 = [i for i in batch_data1.values()]
                p0 = [i.to(device) for i in p0]
                p0_neighbors_idx = [i.to(device) for i in p0_neighbors_idx]
                p0_pool_idx = [i.to(device) for i in p0_pool_idx]
                p0_unsam_idx = [i.to(device) for i in p0_unsam_idx]
                p1 = [i.to(device) for i in p1]
                p1_neighbors_idx = [i.to(device) for i in p1_neighbors_idx]
                p1_pool_idx = [i.to(device) for i in p1_pool_idx]
                p1_unsam_idx = [i.to(device) for i in p1_unsam_idx]
                knearest_idx = [knearest_idx0.to(device), knearest_idx1.to(device)]
                    
                lb0 = lb0.squeeze(-1).to(device)
                lb1 = lb1.squeeze(-1).to(device)
                
                time_i = time.time()
                v_out0, v_out1 = net([p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx],
                              [p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx],
                              knearest_idx)
                v_out0 = v_out0.max(dim=-1)[1]; v_out1 = v_out1.max(dim=-1)[1];
                iou_calc.add_data(v_out0.squeeze(0), v_out1.squeeze(0), lb0.squeeze(0), lb1.squeeze(0))
#                 if tcfg.save_prediction: # if save validation prediction
                if False:
                    utl.save_prediction2(raw_data[0], raw_data[1], 
                                         lb0, lb1, 
                                         v_out0.squeeze(-1), v_out1.squeeze(-1), 
                                         tcfg.plane_threshold, os.path.join(tcfg.path['val_prediction'], str(dir_name[0])), 
                                         pc0_name, pc1_name,
                                         tcfg.path['train_dataset'])
            metrics = iou_calc.metrics()  
            criterion = tcfg.criterion
        
        cur_metric = metrics[criterion]
        utl.mkdir(tcfg.path['best_weights_save_dir'])
        if cur_metric > best_metric: 
            best_metric = cur_metric
            shutil.copy(os.path.join(tcfg.path['weights_save_dir'],'current_net.pth'),os.path.join(tcfg.path['best_weights_save_dir'], 'best_net.pth'))           
        with open(os.path.join(tcfg.path['outputs'],'val_metric.txt'),'a') as f:
            f.write('Time:{},current_epoch:{},criterion: {}, current_metric:{},best_metrci:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
                                                                                                      epoch, criterion, cur_metric, best_metric))
            f.write('\n')    
        with open(os.path.join(tcfg.path['outputs'], 'val_performance.txt'),'a') as f:
            f.write('Time:{},current_epoch:{},miou:{:.6f},oa:{:.6f},iou_list:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
                    epoch,metrics['miou'], metrics['oa'], metrics['iou_list']))
            f.write('\n') 
        print('{}:  current metric {}, best metric {}'.format(criterion, cur_metric, best_metric))
     
if __name__ == '__main__':
    
    tcfg = cfg.CONFIGS['Train']
    if tcfg.display:
        vis = visdom.Visdom(server="http://localhost", port=8097)
    else:
        vis = None
    
    train_network(tcfg, vis)
    test.test_network(tcfg)
    
    
    