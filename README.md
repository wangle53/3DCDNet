# 3DCDNet: An End-to-end Point-based Method and A New Dataset for Street Level Point Cloud Change Detection
![image](https://user-images.githubusercontent.com/79884379/222049705-65fe8a0f-a9f3-47fd-acd0-868a7cebc0be.png)
![image](https://github.com/wangle53/3DCDNet/assets/79884379/e406a78a-ea4d-44c7-a9a2-4dd1a89acfea)


## Requirement
```
python 3.7.4
torch 1.8.10
visdom 0.1.8.9
torchvision 0.9.0
```
## SLPCCD Dataset
This dataset is developed from SHREC2O21 (T. Ku, S. Galanakis, B. Boom et al., SHREC 2021: 3D Point cloud change detection for street scenes, Computers & Graphics, https://doi.org/10.1016/j.cag.2021.07.004). It is a new 3D change detection benchmark dataset and aims to provide opportunities for researchers to develop novel 3D change detection algorithms. The dataset is available at [[Google Drive]](https://drive.google.com/drive/folders/15Wom0FQ6K6RcGxfLAnS-ELDrpZq-xYH1?usp=sharing) and [[Baiduyun]](https://pan.baidu.com/s/1onEEmQKkt7aXTLKJVB7agQ?pwd=8epz) (the password is: 8epz). 
## Pretrained Model
The pretrained model for SLPCCD is available at [[Google Drive]](https://drive.google.com/drive/folders/15Wom0FQ6K6RcGxfLAnS-ELDrpZq-xYH1?usp=sharing) and [[Baiduyun]](https://pan.baidu.com/s/1onEEmQKkt7aXTLKJVB7agQ?pwd=8epz) (the password is: 8epz).
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
## Experiments on Urb3DCD dataset
The experiments on Urb3DCD dataset can be found from [this link](https://github.com/wangle53/3DCDNet-Urb3DCD).
## Citing 3DCDNet
If you use this repository or would like to refer the paper, please use the following BibTex entry.
## Reference
-T. Ku, S. Galanakis, B. Boom et al., SHREC 2021: 3D Point cloud change detection for street scenes, Computers & Graphics, https://doi.org/10.1016/j.cag.2021.07.004  
-HU, Qingyong, et al. Randla-net: Efficient semantic segmentation of large-scale point clouds. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. p. 11108-11117.
## More
[My personal google web](https://scholar.google.com/citations?user=qdkY0jcAAAAJ&hl=zh-TW)
<p> 
  <a href="https://scholar.google.com/citations?user=qdkY0jcAAAAJ&hl=zh-TW"><img src="https://img.shields.io/badge/scholar-4385FE.svg?&style=plastic&logo=google-scholar&logoColor=white" alt="Google Scholar" height="25px"> </a>
</p> 
