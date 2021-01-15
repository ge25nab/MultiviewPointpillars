# MultiviewPointpillars

MultiviewPointpillars is a variant of Pointpillars with both front view and BEV.  
The code structure is based on OpenPCDet. The main idea also refers to the paper [Pillar-based Object Detection for Autonomous Driving](https://github.com/WangYueFt/pillar-od), and could be viewed as its variant re-implementation from Tensorflow to PyTorch Version. 

# Dependencices
1. Python 3.5+
2. pytorch_scatter
3. Pytorch (test on 1.4)
4. OpenPCDet(spconv will not be used)
5. mmdet3d(if use dynamic voxelization)

# Intallation
1. clone this repository
```
git clone git@github.com:ge25nab/MultiviewPointpillars.git
```
2. install the dependet libraries
```
pip install -r requirements.txt 
```
3. install the library 
```
python setup.py develop
```

# Data Preparation
Follow the guidence from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md). Only support KITTI dataset so far.


# Train and Evaluation
## Train Multiview Pointpillars
```
cd ./MultiviewPointpillars
python tools/train.py --cfg_file tools/cfgs/kitti_models/multiview_2conv.yaml
```

## Evaluate Multiview Pointpillars
```
cd ./MultiviewPointpillars
python tools/test.py --cfg_file tools/cfgs/kitti_models/multiview_2conv.yaml --ckpt /path/to/your/weight
```

# Pretrained Weight 

To be uploaded


