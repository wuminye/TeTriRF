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




def tile_maker(feat_plane, h = 2160, w= 3840):
    image = torch.zeros(h,w)
    #image = torch.zeros(1440,2560)

    h,w = list(feat_plane.size())[-2:]


    x,y = 0,0
    for i in range(feat_plane.size(1)):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "Tile_maker: not enough space"

        #ipdb.set_trace()

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



    data = density_quantize(density_grid, nbits)
    

    res = tile_maker(data[0])

    return res



if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logdir', required=True,
                        help='config file path')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames')
    
    parser.add_argument("--qp", type=int, default=20,
                        help='qp value for video codec')

    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')

    args = parser.parse_args()


    thresh = 0


    bound_thres = 20
    low_bound = -bound_thres
    high_bound = bound_thres
    nbits = 2**16-1

    if args.logdir[-1] =='/':
        args.logdir = args.logdir[:-1]
    name = args.logdir.split('/')[-1]

    save_dir = os.path.join(args.logdir, f'compressed_{args.qp}')

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
            #mask = density.reshape(density.size(0),-1).clone()
            #mask = torch.nn.functional.softplus(mask + ckpt['model_state_dict']['act_shift'].cpu()) >0.4
            #mask = mask.sum(dim=1)
            #mask = mask<=thresh

            density[mask] = -5

         

            feature_alpha = F.interpolate(alpha, size=tuple(np.array(volume_size)*3), mode='trilinear', align_corners=True)
            mask_fg = feature_alpha>=1e-4

            #mask_fg = ~mask
            #mask_fg = mask_fg.reshape(mask_fg.size(0),1,1,1,1).repeat(1,1,voxel_size,voxel_size,voxel_size)
            #mask_fg = zero_unpads(merge_volume(mask_fg,grid_size),volume_size)



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
                #torch.median(planes[f"{p}_plane"])


                #sanity check
                ra = planes[f"{p}_plane"].abs()
                ra = ra[cur_mask>=1].abs()
                assert (ra<bound_thres).sum()/ra.size(0) >0.95, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
                ratios.append(((ra<bound_thres).sum()/ra.size(0)).item())
            else:
                ra = planes[f"{p}_plane"].abs()
                ra = ra.reshape(-1)
                assert (ra<bound_thres).sum()/ra.size(0) >0.95, f" absolute value error {(ra<bound_thres).sum()/ra.size(0)}"
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

        
        #density_image = make_density_image(ckpt['model_state_dict']['density.grid'], 16)
        density_image = make_density_image(density, 16)
        #ipdb.set_trace()
        cv2.imwrite(os.path.join(save_dir, f'density_frame_{frameid+1}.png'),to16b(density_image.cpu().numpy()))

        torch.save({'plane_size':plane_sizes, 
                    'bounds': (low_bound,high_bound),
                    'nbits':nbits}, 
                    os.path.join(save_dir, f'planes_frame_meta.nf'))
    
    #parallel_process(args.numframe, args, thresh, bound_thres, low_bound, high_bound, nbits, args.qp)

    save_dir = os.path.join(args.logdir, f'compressed_{args.qp}')
    filename = '/dev/shm/planes_to_videos.sh'
    with open(filename,'w') as f:
        f.write(f'cd {save_dir}\n')
       
        for p in ['xy','xz','yz']:
            if args.codec =='h265':
                f.write(f"ffmpeg -y -framerate 30 -i {p}_planes_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc   -crf {args.qp}  {p}_planes.mp4\n")
            elif args.codec =='mpg2':
                f.write(f"ffmpeg -y -framerate 30 -i {p}_planes_frame_%d.png -c:v mpeg2video  -color_range pc   -qscale:v {args.qp}  {p}_planes.mpg\n")
        if args.codec =='h265':
            f.write(f"ffmpeg -y -framerate 30 -i density_frame_%d.png -c:v libx265 -pix_fmt gray12le -color_range pc   -crf {args.qp}  density_planes.mp4\n")
        elif args.codec =='mpg2':    
            f.write(f"ffmpeg -y -framerate 30 -i density_frame_%d.png -c:v mpeg2video  -color_range pc   -qscale:v {args.qp}  density_planes.mpg\n")
       
        f.write(f'rm -f ./*.png\n')
        f.write(f'cp ../rgbnet* ./\n')
        f.write(f'cp ../config.py ./\n')
        f.write(f'cd ..\n')
        f.write(f'zip -r compressed.zip compressed_{args.qp}\n')


    os.system(f"bash {filename}")







