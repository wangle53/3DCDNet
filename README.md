# [3DCDNet: An End-to-end Point-based Method and A New Dataset for Street Level Point Cloud Change Detection](https://ieeexplore.ieee.org/document/10184135?source=authoralert)

![image](https://github.com/wangle53/3DCDNet/assets/79884379/5a5efd38-c2e4-4f60-b670-28b6b51adc08)
![image](https://github.com/wangle53/3DCDNet/assets/79884379/7f28673f-579c-43f8-9082-643ea74e6045)



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
If you use this repository or would like to refer the paper, please use the following BibTex entry.## Citing TransCD
```
@ARTICLE{10184135,
  author={Wang, Zhixue and Zhang, Yu and Luo, Lin and Yang, Kai and Xie, Liming},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={An End-to-End Point-Based Method and a New Dataset for Street-Level Point Cloud Change Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3295386}}
```
## Reference
-T. Ku, S. Galanakis, B. Boom et al., SHREC 2021: 3D Point cloud change detection for street scenes, Computers & Graphics, https://doi.org/10.1016/j.cag.2021.07.004  
-HU, Qingyong, et al. Randla-net: Efficient semantic segmentation of large-scale point clouds. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. p. 11108-11117.
## More
[My personal google web](https://scholar.google.com/citations?user=qdkY0jcAAAAJ&hl=zh-TW)
<p> 
  <a href="https://scholar.google.com/citations?user=qdkY0jcAAAAJ&hl=zh-TW"><img src="https://img.shields.io/badge/scholar-4385FE.svg?&style=plastic&logo=google-scholar&logoColor=white" alt="Google Scholar" height="25px"> </a>
</p> 
