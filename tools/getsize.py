import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2

import numpy as np
import wandb


names = ['sport1_exp', 'sport2_exp','sport3_exp']

for name in names:
    mypath = f'/logs2/{name}/'

    datasizes = []
    psnrs = []

    numframe = 180
    step = 10
    qps = [15,20,25,28,33]

    wandbrun = wandb.init(
        # set the wandb project where this run will be logged
        project="NeRFVideo_Joint",

        resume = "allow",
        id = 'compressionV3_'+name,
    )

    for qp in qps:
    
        os.system(f"python tools/planes_to_videos.py --logdir {mypath}  --numframe {numframe} --qp {qp}")

        ss = 0
        for p in ['xy','xz','yz']:
            stats = os.stat(os.path.join(mypath,'raw',f"{p}_planes.mp4"))
            ss +=stats.st_size/1024
        stats = os.stat(os.path.join(mypath,'raw',f"density_planes.mp4"))
        ss +=stats.st_size/1024
        datasizes.append(ss/numframe)
        print('datasize per frame:',ss/numframe,'KB')
        
    wandbrun.log({'data_size': datasizes },commit=False)
    wandbrun.log({'qps': qps })

    wandbrun.finish()


 
