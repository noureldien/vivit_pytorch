# Implementation of ViViT: A Video Vision Transformer

#### Model 4
This is the implementation of only model 4 in the paper: factorized dot-product attention.
There is a small error in the implementation, I'll try to fix it when I have time.
The error is I use different query for space and time attentions.
But in the paper, they use the same query for space and time attention.
Please see 2nd paragraph in page 5.

#### Python
This project uses `python 3.7.3`. I use the `anaconda` distribution,
but please use whatever suits you. Just make sure all  the depending packages are installed.
If you want to use anaconda, then you can install it and the depending packages
using the following script
```
# Anaconda 4.7.12, Python 3.7.3, download anaconda then install it
https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Create a conda env and work within it
conda create -n video_tfms python=3.7.3 pip conda

# Python Libaries
pip install moviepy pytube3 multicoretsne dill thop seaborn natsort paramiko pydot joblib h5py sklearn pyyaml torchsummary torchviz tensorboardx ipdb 
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# for opencv, use
pip install opencv-python

# or install it from one of the anaconda distributions
conda install -c loopbio opencv
conda install -c anaconda opencv
conda install -c menpo opencv3
conda install -c jjhelmus opencv

```

#### Pacakge Requirements
Please make sure you check list of required pacakges in this file [`requirements.txt`](requirements.txt)
