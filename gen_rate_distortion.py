import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange



# experiment names in the output folder. Do not forget to specify the output folder in 'tools/distortion_curve.py'
names = ['sport1']

filename = '/dev/shm/gen_rate.sh'
with open(filename,'w') as f:
    for name in names:
        f.write(f'python tools/distortion_curve.py --name {name}\n')

os.system(f"bash {filename}")