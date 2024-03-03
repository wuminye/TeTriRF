import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb


# the output folder
outfolder = '<folder path>'


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def feats_pca_projection(feats,pre = None):
    data = np.mat(feats.T)
    if pre is None:
        data_mean = data - data.mean(axis=1)
        data_cov = np.cov(data_mean)
        tzz,tzxl = np.linalg.eig(data_cov)
        xl = tzxl.T[0:3]
        res =  data_mean.T.__mul__(np.mat(xl).T)
    else:
        data_mean = data - pre[0]
        xl = pre[1]
        res =  data_mean.T.__mul__(np.mat(xl).T)

    res = np.array(res)

    return res,(data.mean(axis=1),xl )


def feats_map_pca_projection(feats_map, pre = None):
    feats = feats_map.transpose(1,2,0)
    ori_shape = feats.shape
    feats = np.reshape(feats,(-1,feats.shape[2]))

    res, ss = feats_pca_projection(feats,pre)
    res = res.reshape((ori_shape[0],ori_shape[1],3))
    return res,ss



name = 'flame_steak_exp4'
path =  os.path.join(outfolder,name)

wandbrun = wandb.init(
        # set the wandb project where this run will be logged
        project="TeTriRF",
    
        # track hyperparameters and run metadata
        config={
        "Path": path,
        },
        id = 'visual_'+name
    )

os.makedirs(os.path.join(path,'feats'), exist_ok=True)

pre = None
mmin = None
mmax = None
for frameid in range(0,180):

    state = torch.load(os.path.join(path, f'fine_last_{frameid}.tar'))
    planes = ['k0.xy_plane','k0.xz_plane','k0.yz_plane']
    mean_vals = []

    ret = []
    for plane in planes:
        print(plane)
        data = state['model_state_dict'][plane].cpu().numpy()[0]
        res, ss=feats_map_pca_projection(data, pre)
        if pre is None:
            pre = ss
            mmin = np.min(res)
            res = res - np.min(res)
            mmax = np.max(res)
            res = res/ np.max(res)
        else:
            res = res - mmin
            res = res/ mmax

        cv2.imwrite(os.path.join(path, 'feats', f'{frameid}_{plane}.jpg'), res*255)

    
        ret.append(wandb.Image(to8b(res), caption=f"feature_{plane}_{frameid}"))
        
        mean_vals.append(np.abs(data).mean())
        print(np.abs(data).mean())
        print(np.count_nonzero(data)/np.prod(data.shape)*100,'%')

filename = '/dev/shm/feats_visualization_videos.sh'
save_dir = os.path.join(path, 'feats')
print(save_dir)
with open(filename,'w') as f:
    f.write(f'cd {save_dir}\n')
    for p in ['xy','xz','yz']:
        f.write(f"ffmpeg -y -framerate 5 -start_number 0 -i %d_k0.{p}_plane.jpg -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264  -preset veryslow /dev/shm/{name}_{p}_plane.mp4\n")


os.system(f"bash {filename}")


videos = []
for p in ['xy','xz','yz']:
    videos.append(wandb.Video(f"/dev/shm/{name}_{p}_plane.mp4", fps=5, caption=p))
wandb.log({"video": videos})