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
import wandb
import cv2

from multiprocessing import Pool
from functools import partial



def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_val = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_val

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : ((2**16-1)*np.clip(x,0,1)).astype(np.uint16)


def zero_pads(data, voxel_size =16):
    if data.size(0)==1:
        data = data.squeeze(0)

    size = list(data.size())

    new_size = copy.deepcopy(size)
    for i in range(1,len(size)):
        if new_size[i]%voxel_size==0:
            continue
        new_size[i] = (new_size[i]//voxel_size+1)*voxel_size
    
    res= torch.zeros(new_size, device = data.device)
    res[:,:size[1],:size[2],:size[3]] = data.clone()
    return res

def zero_unpads(data, size):
    
    return data[:,:size[0],:size[1],:size[2]]


def split_volume(data, voxel_size =16):
    size = list(data.size())
    for i in range(1,len(size)):
        size[i] = size[i]//voxel_size

    res = []
    for x in range(size[1]):
        for y in range(size[2]):
            for z in range(size[3]):
                res.append(data[:,x*voxel_size:(x+1)*voxel_size,y*voxel_size:(y+1)*voxel_size,z*voxel_size:(z+1)*voxel_size].clone())

    res = torch.stack(res)

    return res,size[1:]


def merge_volume(data,size):
    voxel_size = data.size(-1)
    
    
    res=torch.zeros(data.size(1), size[0]*voxel_size,size[1]*voxel_size, size[2]*voxel_size,device = data.device)
    cnt = 0
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                res[:,x*voxel_size:(x+1)*voxel_size,y*voxel_size:(y+1)*voxel_size,z*voxel_size:(z+1)*voxel_size] = data[cnt,:,:,:,:]
                cnt = cnt + 1

    return res



def tile_maker(feat_plane, h = 2160, w= 3840):
    image = torch.zeros(h,w)

    h,w = list(feat_plane.size())[-2:]


    x,y = 0,0
    for i in range(feat_plane.size(1)):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "Tile_maker: not enough space, please increase the image resolution"

        image[x:x+h,y:y+w] = feat_plane[0,i,:,:]
        y = y + w

    return image

def density_quantize(density,nbits):

    nbits = 2**nbits-1
    data = density.clone()
    
    data[data<-5] = -5
    data[data>30] = 30

    data = data +5
    data = data /(30+5)
    

    data = torch.round(data *nbits)/nbits

    return data

def density_dequantize(density):
    
    data = density *(30+5)
    data = data-5


    return data




def make_density_image(density_grid, nbits, act_shift=0):


    '''
    data = density_grid +3
    data[data<0] = 0
    data = torch.log10(0.1*data+1.0)
    data = data/1.5
    data[data>1.0] = 1.0
    '''
    data = density_quantize(density_grid, nbits)
    
    #res = tile_maker(data[0],h=4320,w=7680)
    res = tile_maker(data[0])

    '''
    ipdb.set_trace()
    
    data = data.reshape(-1)
    th = int(np.round(data.size(0)/image.size(1))+1)
    padding = th*image.size(1) - data.size(0)
    padded_tensor = torch.cat((data, torch.zeros(padding)))

    image[:th,:] = padded_tensor.view(th,-1)
    '''


    return res



if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True,
                        help='config file path')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames')

    parser.add_argument("--voxel_size", type=int, default=8,
                        help='number of frames')
    
    parser.add_argument("--qp", type=int, default=20,
                        help='qp value for video codec')

    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')

    args = parser.parse_args()

    voxel_size = args.voxel_size

    thresh = 0


    bound_thres = 20
    low_bound = -bound_thres
    high_bound = bound_thres
    nbits = 2**16-1

    if args.logdir[-1] =='/':
        args.logdir = args.logdir[:-1]
    name = args.logdir.split('/')[-1]
    wandbrun = wandb.init(
        # set the wandb project where this run will be logged
        project="TeTriRF",
    
        # track hyperparameters and run metadata
        config={
        "Path": args.logdir,
        "qp": args.qp,
        "numframe": args.numframe,
        "voxel_size": voxel_size
        },
        resume = "allow",
        id = 'compressionV7_'+name+'_'+args.codec,
    )

    save_dir = os.path.join(args.logdir, 'raw')

    for frameid in tqdm(range(0, args.numframe)):


        if os.path.isfile(os.path.join(save_dir, f'density_frame_{frameid+1}.png')):
            continue


        tmp_file = os.path.join(args.logdir, f'fine_last_{frameid}.tar')

        assert os.path.isfile(tmp_file), "Checkpoint not found."

        tqdm.write(f"Loading Checkpoint {tmp_file}")
        ckpt = torch.load(tmp_file, map_location='cpu')
        

        density = ckpt['model_state_dict']['density.grid'].clone()
        volume_size = list(density.size())[-3:]


        voxel_size_ratio = ckpt['model_kwargs']['voxel_size_ratio']



        masks = None
        if 'act_shift' in ckpt['model_state_dict']:

            alpha = 1- (torch.exp(density+ckpt['model_state_dict']['act_shift'])+1)**(-voxel_size_ratio)
            alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1)

            mask = alpha<1e-4

            density[mask] = -5

         

            feature_alpha = F.interpolate(alpha, size=tuple(np.array(volume_size)*3), mode='trilinear', align_corners=True)
            mask_fg = feature_alpha>=1e-4


            # mask projection
            masks = {}
            masks['xy'] = mask_fg.sum(axis=4)
            masks['xz'] = mask_fg.sum(axis=3)
            masks['yz'] = mask_fg.sum(axis=2)


        planes = {}


        for key in ckpt['model_state_dict'].keys():
            if 'k0' in key and 'plane' in key and 'residual' not in key:
                data = ckpt['model_state_dict'][key]
                planes[key.split('.')[-1]]= data

        
        plane_data = []
        ratios = []
        tpsnr = []
        for p in ['xy','xz','yz']:
            plane_size = list(planes[f"{p}_plane"].size())[-1:-3:-1]

            
            if masks is not None:

                cur_mask = masks[p].repeat(1,planes[f"{p}_plane"].size(1),1,1)

                planes[f"{p}_plane"][cur_mask<1] = 0



                #sanity check
                ra = planes[f"{p}_plane"].abs()
                ra = ra[cur_mask>=1].abs()
                assert (ra<bound_thres).sum()/ra.size(0) >0.995, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
                ratios.append(((ra<bound_thres).sum()/ra.size(0)).item())
            else:
                ra = planes[f"{p}_plane"].abs()
                ra = ra.reshape(-1)
                assert (ra<bound_thres).sum()/ra.size(0) >0.995, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
                ratios.append(((ra<bound_thres).sum()/ra.size(0)).item())

            feat = (planes[f"{p}_plane"] - low_bound)/(high_bound-low_bound)
            
            feat = torch.round(feat *nbits)/nbits

            
            feat[feat<0]=0
            feat[feat>1.0] = 1.0

            plane_data.append(feat)

            gt_feat = (planes[f"{p}_plane"]- low_bound)/(high_bound-low_bound)

            tpsnr.append(psnr(gt_feat,feat).item())


        tqdm.write(f"ratio: {np.mean(ratios)*100}%   PSNR:{np.mean(tpsnr)} qp:{args.qp}")
        os.makedirs(save_dir, exist_ok=True)


  
        imgs = {}
        plane_sizes ={}
        for ind,plane in zip(['xy','xz','yz'],plane_data):
            img = tile_maker(plane).half()
            imgs[f'{ind}_plane'] = img
            plane_sizes[f'{ind}_plane'] = plane.size()

            cv2.imwrite(os.path.join(save_dir, f'{ind}_planes_frame_{frameid+1}.png'),to16b(img.cpu().numpy()))

        
 
        density_image = make_density_image(density, 16)
  
        cv2.imwrite(os.path.join(save_dir, f'density_frame_{frameid+1}.png'),to16b(density_image.cpu().numpy()))

        torch.save({'img':imgs, 'plane_size':plane_sizes, 
                    'density': density_dequantize(density_quantize(ckpt['model_state_dict']['density.grid'].clone()[0],16).float()),
                    'bounds': (low_bound,high_bound),
                    'nbits':nbits,
                    'planes':planes,
                    'qp':args.qp}, 
                    os.path.join(save_dir, f'planes_frame_{frameid}.nf'))
    
    #parallel_process(args.numframe, args, thresh, bound_thres, low_bound, high_bound, nbits, args.qp)

    save_dir = os.path.join(args.logdir, 'raw')
    filename = '/dev/shm/planes_to_videos.sh'
    with open(filename,'w') as f:
        f.write(f'cd {save_dir}\n')
        f.write(f'rm -f ./*_out.bmp\n')
        for p in ['xy','xz','yz']:
            if args.codec =='h265':
                f.write(f"ffmpeg -y -framerate 30 -i {p}_planes_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc   -crf {args.qp}  {p}_planes.mp4\n")
            elif args.codec =='mpg2':
                f.write(f"ffmpeg -y -framerate 30 -i {p}_planes_frame_%d.png -c:v mpeg2video  -color_range pc   -qscale:v {args.qp}  {p}_planes.mpg\n")
        if args.codec =='h265':
            f.write(f"ffmpeg -y -framerate 30 -i density_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc   -crf {args.qp}  density_planes.mp4\n")
        elif args.codec =='mpg2':    
            f.write(f"ffmpeg -y -framerate 30 -i density_frame_%d.png -c:v mpeg2video  -color_range pc   -qscale:v {args.qp}  density_planes.mpg\n")
       

    os.system(f"bash {filename}")

    
    if args.codec =='h265':

        filename = '/dev/shm/planes_to_videos.sh'
        with open(filename,'w') as f:
            f.write(f'cd {save_dir}\n')
            for p in ['xy','xz','yz']:
                f.write(f"cp {p}_planes.mp4 {p}_planes_{args.qp}.mp4 \n")

            f.write(f"cp density_planes.mp4 density_planes_{args.qp}.mp4 \n")
        
            #f.write(f'rm -f ./*.bmp\n')
            #for p in ['xy','xz','yz']:
            #    f.write(f"ffmpeg -y -i {p}_planes.mp4  -pix_fmt gray {p}_planes_frame_%d_out.bmp\n")

        os.system(f"bash {filename}")


        videos = []
        for p in ['xy','xz','yz']:
            filename = os.path.join(save_dir, f"{p}_planes.mp4")
            videos.append(wandb.Video(filename, fps=5, caption=p+f" qp: {args.qp}"))
            
        filename = os.path.join(save_dir, f"density_planes.mp4")
        videos.append(wandb.Video(filename, fps=5, caption=p+f" qp: {args.qp}"))


        wandbrun.log({"video": videos})
    #wandb.log({"video": videos, "Reconstruction_PSNR" : wandb.plot.line(table, "frame", "psnr",
    #        title="Reconstruction PSNR")},step=0)
        

        
    

        



