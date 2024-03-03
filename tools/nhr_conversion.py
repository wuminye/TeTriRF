import numpy as np
import os
import json

# path to the scene folder of NHR dataset
root_path = '/scratch/leuven/346/vsc34668/sport_1_mask'
# number of cameras.  56 for NHR dataset
N_cam = 56  
# number of frames.  56 for NHR dataset
N_frame = 200

def campose_to_extrinsic(camposes):
    if camposes.shape[1]!=12:
        raise Exception(" wrong campose data structure!")
        return
    
    res = np.zeros((camposes.shape[0],4,4))
    
    res[:,0:3,2] = camposes[:,0:3]
    res[:,0:3,0] = camposes[:,3:6]
    res[:,0:3,1] = camposes[:,6:9]
    res[:,0:3,3] = camposes[:,9:12]
    res[:,3,3] = 1.0
    
    return res


def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data= fo.readlines()
    i = 0
    Ks = []
    while i<len(data):
        if len(data[i])>5:
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            a = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            b = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            c = np.array(tmp)
            res = np.vstack([a,b,c])
            Ks.append(res)

        i = i+1
    Ks = np.stack(Ks)
    fo.close()

    return Ks


camposes = np.loadtxt(os.path.join(root_path,'CamPose.inf'))
Ts = campose_to_extrinsic(camposes)
Ks = read_intrinsics(os.path.join(root_path,'Intrinsic.inf'))


# normalize to zeros , range: [-2,2]
m = np.mean(Ts[:,:3,3],axis = 0)
print('OBJ center:',m)
Ts[:,:3,3] = Ts[:,:3,3] - m
print(Ts[:,:3,3].max(),-Ts[:,:3,3].min())
Ts[:,:3,3] = Ts[:,:3,3]*2.0/max(Ts[:,:3,3].max(),-Ts[:,:3,3].min())

for j in range(N_frame):
    frames = []
    for i in range(N_cam):
        frame = {}
        frame['file'] = 'img/%d/img_%04d.jpg' % (j,i)
        frame['mask'] = 'img/%d/mask/img_%04d.jpg' %  (j,i)
        frame['extrinsic'] = Ts[i].tolist()
        frame['intrinsic'] = Ks[i].tolist()
        frames.append(frame)
    with open(os.path.join(root_path,'cams_%d.json' % j), 'w', encoding='utf-8') as f:
        json.dump({'frames':frames}, f, ensure_ascii=False, indent=4)

print('done.')