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




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', required=True,
                    help='config file path')
parser.add_argument("--codec", type=str, default='h265',
                    help='h265 or mpg2')

args = parser.parse_args()



name = args.name
# !!!!! modify the path to the output folder
mypath = f'/logs2/{name}/'



filename = '/dev/shm/distortion.sh'
with open(filename,'w') as f:
    f.write(f'cd {mypath}\n')
    f.write(f'rm -rf raw\n')
  
#os.system(f"bash {filename}")




datasizes = []
psnrs = []
ssims=[]
lpipss=[]

numframe = 180
step = 10
qps = [18,20,24,28,33,38]

if args.codec == 'mpg2':
    qps = [2,6,10,14,18]  #mpeg

wandbrun = wandb.init(
    # set the wandb project where this run will be logged
    project="NeRFVideo_Joint",

    resume = "allow",
    id = 'compressionV7_'+name+'_'+args.codec,
)



for qp in qps:
    os.system(f"python tools/planes_to_videos.py --logdir {mypath}  --numframe {numframe} --qp {qp} --codec {args.codec}")
    os.system(f"python tools/videos_to_planes.py --dir {mypath}raw  --numframe {numframe} --codec {args.codec}")
    ss = 0

    if args.codec == 'mpg2':
        for p in ['xy','xz','yz']:
            stats = os.stat(os.path.join(mypath,'raw',f"{p}_planes.mpg"))
            ss +=stats.st_size/1024
        stats = os.stat(os.path.join(mypath,'raw',f"density_planes.mpg"))
        ss +=stats.st_size/1024
    else:
        for p in ['xy','xz','yz']:
            stats = os.stat(os.path.join(mypath,'raw',f"{p}_planes.mp4"))
            ss +=stats.st_size/1024
        stats = os.stat(os.path.join(mypath,'raw',f"density_planes.mp4"))
        ss +=stats.st_size/1024
    print('datasize per frame:',ss/numframe,'KB')

    tmp_psnr = []
    tmp_ssim=[]
    tmp_lpips=[]
 
    tmp = " ".join([str(j) for j in range(0,numframe,step)])
    os.system(f"python render.py --config {mypath}config.py --render_only --render_test --frame_ids {tmp} --qp {qp} --codec {args.codec}")


    
    for i in range(0,numframe,step):
        p = np.loadtxt(f'{mypath}render_test/{i}_psnr.txt')
        tmp_psnr.append(p)
        print(p)
        p = np.loadtxt(f'{mypath}render_test/{i}_ssim.txt')
        tmp_ssim.append(p)
        p = np.loadtxt(f'{mypath}render_test/{i}_lpips.txt')
        tmp_lpips.append(p)
    psnrs.append(np.mean(tmp_psnr))
    ssims.append(np.mean(tmp_ssim))
    lpipss.append(np.mean(tmp_lpips))
    datasizes.append(ss/numframe)




print(datasizes)
print(psnrs)


data = [[x, y] for (x, y) in zip(datasizes, psnrs)]
table = wandb.Table(data=data, columns = ["size", "psnr"])
wandbrun.log({"Distortion Curve" : wandb.plot.line(table,
                                 "size", "psnr", title="Distortion Curve")}, commit = False)

data = [[x, y] for (x, y) in zip(datasizes, ssims)]
table = wandb.Table(data=data, columns = ["size", "ssim"])
wandbrun.log({"Distortion Curve SSIM" : wandb.plot.line(table,
                                 "size", "ssim", title="Distortion Curve SSIM")}, commit = False)

data = [[x, y] for (x, y) in zip(datasizes, lpipss)]
table = wandb.Table(data=data, columns = ["size", "lpips"])
wandbrun.log({"Distortion Curve LPIPS" : wandb.plot.line(table,
                                 "size", "lpips", title="Distortion Curve LPIPS")})


wandbrun.finish()