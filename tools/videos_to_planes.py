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

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_val = 10 * torch.log10(max_val ** 2 / mse)
    return psnr_val

def untile_image(image,h,w,ndim):

    features = torch.zeros(1,ndim,h,w)

    x,y = 0,0
    for i in range(ndim):
        if y+w>=image.size(1):
            y=0
            x = x+h
        assert x+h<image.size(0), "untile_image: too many feature maps"

        features[0,i,:,:] = image[x:x+h,y:y+w]
        y = y + w

    return features


def tile_maker(feat_plane, h = 2160, w= 3840):
    image = torch.zeros(h,w)

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

def make_density_image(density_grid, nbits, h=4320,w=7680):
    #image = torch.zeros(1080,1920)

    data = density_grid +5
    data[data<0] = 0

    data = data / 30
    data[data>1.0] = 1.0

    data = torch.round(data *nbits)/nbits
    
    #res = tile_maker(data,h=1080,w=1920)
    res = tile_maker(data, h=h,w=w)



    return res

if __name__=='__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', required=True,
                        help='raw files director path')



    parser.add_argument('--model_template', type=str, default='fine_last_0.tar',
                        help='model template')

    parser.add_argument("--numframe", type=int, default=10,
                        help='number of frames')

    parser.add_argument("--codec", type=str, default='h265',
                        help='h265 or mpg2')


    args = parser.parse_args()


    outdir = os.path.join(args.dir,'../raw_out')

    os.makedirs(outdir, exist_ok=True)


    ckpt = torch.load(os.path.join(args.dir, '..', args.model_template))

    name = args.dir.split('/')[-2]
    wandbrun = wandb.init(
        # set the wandb project where this run will be logged
        project="TeTriRF",
    
        # track hyperparameters and run metadata
        resume = "allow",
         id = 'compressionV7_'+name+'_'+args.codec,
    )


    filename = '/dev/shm/videos_to_planes.sh'
    with open(filename,'w') as f:
        f.write(f'cd {args.dir}\n')
        for p in ['xy','xz','yz']:
            if args.codec =='mpg2':
                f.write(f"ffmpeg -y -i {p}_planes.mpg  -pix_fmt gray16be {p}_planes_frame_%d_out.png\n")
            else:
                f.write(f"ffmpeg -y -i {p}_planes.mp4  -pix_fmt gray16be {p}_planes_frame_%d_out.png\n")
                
        if args.codec =='mpg2':
            f.write(f"ffmpeg -y -i density_planes.mpg  -pix_fmt gray16be  density_frame_%d_out.png\n")
        else:
            f.write(f"ffmpeg -y -i density_planes.mp4  -pix_fmt gray16be  density_frame_%d_out.png\n")
    os.system(f"bash {filename}")

    fpsnr = []
    dpsnr = []
    for frameid in tqdm(range(0, args.numframe,10)):
        raw_frame = torch.load(os.path.join(args.dir, f'planes_frame_{frameid}.nf'))

        low_bound,high_bound = raw_frame['bounds']

        
        gt_planes = raw_frame['planes']

        tpsnr = []

        for key in raw_frame['img'].keys():
            #ipdb.set_trace()
            quant_img = cv2.imread(os.path.join(args.dir,f"{key.split('_')[0]}_planes_frame_{frameid+1}_out.png"), -1)

            #ipdb.set_trace()
            plane = untile_image(torch.tensor(quant_img.astype(np.float32))/int(2**16-1), 
                                raw_frame['plane_size'][key][2],
                                raw_frame['plane_size'][key][3],
                                raw_frame['plane_size'][key][1])

            tpsnr.append(psnr(plane, (gt_planes[key].cpu()-low_bound)/(high_bound-low_bound)).item())
            

            plane = plane*(high_bound-low_bound) + low_bound


            assert 'k0.'+key in ckpt['model_state_dict'], ' Wrong plane name'

            ckpt['model_state_dict']['k0.'+key] = plane.clone().cuda()

 

        

        quant_img = cv2.imread(os.path.join(args.dir,f"density_frame_{frameid+1}_out.png"), -1)
        gt_quant_img = make_density_image(raw_frame['density'],int(2**16-1), h = quant_img.shape[0],w = quant_img.shape[1])
        desity_plane = untile_image(torch.tensor(quant_img.astype(np.float32))/int(2**16-1), 
                                raw_frame['density'].size(2),
                                raw_frame['density'].size(3),
                                raw_frame['density'].size(1))

        desity_plane = desity_plane*(30+5) - 5
        
        dpsnr.append(psnr(torch.tensor(quant_img.astype(np.float32))/int(2**16-1), gt_quant_img).item())

 

        fpsnr.append(np.mean(tpsnr))
        tqdm.write(f"Reconstruction PSNR {np.mean(tpsnr)}")

        tqdm.write(f"Reconstruction Density PSNR {psnr(torch.tensor(quant_img.astype(np.float32))/int(2**16-1), gt_quant_img).item()}")


        ckpt['model_state_dict']['density.grid'] = desity_plane.clone().cuda().unsqueeze(0)

        
        torch.save(ckpt, os.path.join(outdir, f'fine_last_{frameid}.tar'))




    data = [[x, y] for (x, y) in zip(range(len(fpsnr)), fpsnr)]
    table = wandb.Table(data=data, columns = ["frame", "psnr"])

    data_d = [[x, y] for (x, y) in zip(range(len(dpsnr)), dpsnr)]
    table_d = wandb.Table(data=data_d, columns = ["frame", "psnr"])

    wandbrun.log({"Reconstruction_Feat_PSNR" : wandb.plot.line(table, "frame", "psnr",
            title=f"Reconstruction Feature PSNR  qp:{raw_frame['qp']}")})