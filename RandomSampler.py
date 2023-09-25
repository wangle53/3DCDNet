#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from tqdm import tqdm
import configs as cfg
tcfg = cfg.CONFIGS['Train']

def random_subsample(points, n_samples):
    if tcfg.remove_plane:
        zmin = np.min(points[:, 2])
        nthr = zmin + tcfg.plane_threshold
        plane = points[points[:, 2] < nthr]
        points = points[points[:, 2] >= nthr] #remove the ground in each PC
    
    if points.shape[0]==0:
        points = np.zeros((n_samples,points.shape[1]))
    if n_samples < points.shape[0]:
        random_indices = np.random.choice(points.shape[0], n_samples, replace=False)
        points = points[random_indices, :]
    if n_samples > points.shape[0]:
        np.random.shuffle(points)
        apd = np.zeros((n_samples-points.shape[0], points.shape[1]))
        points = np.vstack((points, apd))
    return points, plane

def process_data(root, name, n=tcfg.n_samples):
    path = os.path.join(root, name)
    sname = name + '_split_plane_' + str(n) + '_thr_' + str(tcfg.plane_threshold)
    
    for dir in tqdm(os.listdir(path)):
        for file in os.listdir(os.path.join(path, dir)):
            
            sp = os.path.join(root, sname, dir, file)
            spp = os.path.join(root, sname, dir, file.replace('.txt', '_plane.txt'))
            
            if os.path.exists(sp) and os.path.exists(spp):
                continue
            else:
                pc = np.loadtxt(os.path.join(path, dir, file), skiprows=2)
                pc, plane = random_subsample(pc, n)
                
                if not os.path.exists(os.path.join(root, sname, dir)):
                    os.makedirs(os.path.join(root, sname, dir))
                    
                np.savetxt(sp, pc, fmt="%.8f %.8f %.8f %.8f %.8f %.8f %.0f")
                head = '//X Y Z Rf Gf Bf label\n'
                with open(sp, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(head + (str(len(pc))+'\n') + content)
                
                np.savetxt(spp, plane, fmt="%.8f %.8f %.8f %.8f %.8f %.8f %.0f")
            

def generate_removed_plane_dataset_train(root):
    print('removing planes of train and val datasets...')
    process_data(root, 'train_seg', n=tcfg.n_samples)
    print('all point clouds have been removed planes')
    
def generate_removed_plane_dataset_test(root):
    print('removing planes of test datasets...')
    process_data(root, 'test_seg', n=tcfg.n_samples)
    print('all point clouds have been removed planes')
    
if __name__ == '__main__':
    generate_removed_plane_dataset(tcfg.path.data_root)
    
    
    
    
    