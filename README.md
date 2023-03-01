# 3DCDNet: An End-to-end Point-based Method and A New Dataset for Street Level Point Cloud Change Detection  
The source code and dataset are coming soon!!!
## Requirement
```
python 3.7.4
torch 1.8.10
visdom 0.1.8.9
torchvision 0.9.0
```
## PCCD Datasets
This dataset is developed from SHREC2O21 (T. Ku, S. Galanakis, B. Boom et al., SHREC 2021: 3D Point cloud change detection for street scenes, Computers & Graphics, https://doi.org/10.1016/j.cag.2021.07.004). It is a new 3D change detection benchmark dataset and aims to provide opportunities for researchers to develop novel 3D change detection algorithms. The dataset is available at [[Google Drive]](https://drive.google.com/drive/folders/1iwiKVBSFmUdSVhXlix2uVBVpTRzsjkLF?usp=sharing) and [[Baiduyun]](https://pan.baidu.com/s/1XkLOHYKZJj0nYWzCBkIsJg) (the password is: quid). 
## Pretrainde Model
The pretrained model for PCCD is available at [[Google Drive]](https://drive.google.com/drive/folders/1ehQbfsGvOv4syc98r5PlhJDV88Q3bQlg?usp=sharing) and [[Baiduyun]](https://pan.baidu.com/s/1IUy8WFIggkIsHNyR8rTG-w) (the password is: qjmf).
## Test
Before test, please download datasets and pretrained models. Change path to your data path in configs.py. Copy pretrained models to folder './outputs/best_weights', and run the following command: 
```
cd 3DCDNet_ROOT
python test.py
```
## Training
Before training, please download datasets and revise dataset path in configs.py to your path.
```
cd 3DCDNet_ROOT
python -m visdom.server
python train.py
```
To display training processing, open 'http://localhost:8097' in your browser.
## Citing TransCD
If you use this repository or would like to refer the paper, please use the following BibTex entry.
## Reference
-T. Ku, S. Galanakis, B. Boom et al., SHREC 2021: 3D Point cloud change detection for street scenes, Computers & Graphics, https://doi.org/10.1016/j.cag.2021.07.004  
-HU, Qingyong, et al. Randla-net: Efficient semantic segmentation of large-scale point clouds. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. p. 11108-11117.
