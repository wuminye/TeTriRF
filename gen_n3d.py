import numpy as np
import os
import json



step = 20
overlap = True


# add scenes' names that need to be trained
for name in ['sear_steak','cook_spinach','coffee_martini']:


    filename = f'./run_{name}.sh'
    with open(filename,'w') as f:
        f.write(f"python run_multiframe.py --config configs/N3D/{name}.py --frame_ids 0 --training_mode 0\n" )

        for i in range(0,200,step):
            if i!=0 and overlap:
                tmp = " ".join([str(j) for j in range(i-1,i+step)])
            else:
                tmp = " ".join([str(j) for j in range(i,i+step)])

            f.write(f"python run_multiframe.py --config configs/N3D/{name}.py --frame_ids {tmp} --training_mode 1\n" )

    print(filename)


