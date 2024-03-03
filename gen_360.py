import numpy as np
import os
import json


#config file
cfgfile = 'configs/NHR/sport1.py'
#Number of frames
N_frame = 200

tmp = " ".join([str(j) for j in range(0, N_frame)])


filename = f'/dev/shm/run.sh'
with open(filename,'w') as f:
    f.write(f"python render360.py --config {cfgfile} --frame_ids {tmp} --render_only --render_360 0 --reald\n" )


print(filename)

os.system(f"bash {filename}")