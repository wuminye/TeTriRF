import numpy as np
import os
import json


# change the config file here
config_file = "configs/NHR/sport1.py"
step = 20
overlap = True

filename = './run.sh'
with open(filename,'w') as f:
    

    for i in range(0,200,step):
        if i!=0 and overlap:
            tmp = " ".join([str(j) for j in range(i-1,i+step)])
        else:
            tmp = " ".join([str(j) for j in range(i,i+step)])
        mode = 0
        if i!=0:
            mode = 1
        f.write(f"python run_multiframe.py --config {config_file} --frame_ids {tmp} --training_mode {mode}\n" )

print(filename)

os.system(f"sh {filename}")






