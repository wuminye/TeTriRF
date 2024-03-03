import numpy as np
import os
import json




step = 20

names = ['sport1']

for name in names:

    filename = f'./test_{name}.sh'
    with open(filename,'w') as f:
        

        for i in range(0,200,step):
            tmp = " ".join([str(j) for j in range(i,i+step)])
            f.write(f"python render.py --config configs/NHR/{name}.py  --frame_ids {tmp} --render_only --render_test --reald \n" )

    print(filename)
    os.system(f'sh {filename}')








