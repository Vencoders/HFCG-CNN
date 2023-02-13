# HFCG-CNN
code for HFCF-CNN
## Pre-requisites

python3.6

Pytorch 1.6

安装requirements.txt依赖

```
$ pip install -r requirements.txt
```



# Dataset

11,871 pairs (size 160x160) RGB images extracted from [CUFED](http://acsweb.ucsd.edu/~yuw176/event-curation.html).

RGB images are then converted to YUV420 by FFmpeg and encoded by HEVC HM16.22 with random Access configuration. 

We selected the corresponding four quantitative parameters (QP) (22,27,32,37) following the requirement of the challenge and constructed four dataset sets.



# Training

```
$ python train.py
```



# Eval

```
$ python ./eval/*.py
```

最终BD-rate结果由赛方的template.XLSM进行计算
