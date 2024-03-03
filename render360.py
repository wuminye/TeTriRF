import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange
import wandb
import mmcv
import imageio
import numpy as np
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo,  dmpigo
from lib.load_data import load_data
from lib.dvgo_video import RGB_Net, RGB_SH_Net

from torch_efficient_distloss import flatten_eff_distloss
import pandas as pd
import time
import math
def excepthook(exc_type, exc_value, exc_traceback):
    ipdb.post_mortem(exc_traceback)

import json

#sys.excepthook = excepthook

wandbrun=None

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    parser.add_argument('--frame_ids', nargs='+', type=int, help='a list of ID')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_360", default=0, type=int)
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--residual_train", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true', default=True)
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true',  default=True)

    parser.add_argument("--codec", type=str, default='h265',
                    help='h265 or mpg2')


    parser.add_argument("--qp",   type=int, default=0)
    parser.add_argument("--reald", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,frame_id = 0, masks = None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    curf = frame_id % len(render_poses)

    for i, c2w in enumerate(tqdm(render_poses)):
        if i != curf and ndc:
            continue
        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        with torch.no_grad():
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(150480, 0), rays_d.split(150480, 0), viewdirs.split(150480, 0))
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
                for k in render_result_chunks[0].keys()
            }

        
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:

            if masks is not None:
                p = -10. * np.log10(np.mean(np.square(rgb[masks[i][...,0]>0.5,:] - gt_imgs[i][masks[i][...,0]>0.5,:])))
            else:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
        


    res_psnr = {}
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        res_psnr = {'psnr':np.mean(psnrs)}
        with open(os.path.join(savedir, f'{frame_id}_psnr.txt'),'w') as f:
            f.write('%f' % np.mean(psnrs))
        if eval_ssim: 
            print('Testing ssim', np.mean(ssims), '(avg)')
            res_psnr['ssim'] = np.mean(ssims)
            with open(os.path.join(savedir, f'{frame_id}_ssim.txt'),'w') as f:
                f.write('%f' % np.mean(ssims))
        if eval_lpips_vgg: 
            print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
            res_psnr['lpips'] = np.mean(lpips_vgg)
            with open(os.path.join(savedir, f'{frame_id}_lpips.txt'),'w') as f:
                f.write('%f' % np.mean(lpips_vgg))

        if eval_lpips_alex: 
            print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')


    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, f'{frame_id}_{i}.png')
            imageio.imwrite(filename, rgb8)

            rgb8 = utils.to8b(1 - depths[i] / np.max(depths[i]))
            filename = os.path.join(savedir, f'{frame_id}_{i}_depth.png')
            if rgb8.shape[-1]<3:
                rgb8 = np.repeat(rgb8, 3, axis=-1)
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps, res_psnr


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''

    cfg.data.spherify = True
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
                'i_train', 'i_val', 'i_test', 'irregular_shape',
                'poses', 'render_poses', 'images', 'frame_ids','masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict






def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, residual_mode=False):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            residual_mode = residual_mode,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path,residual = False):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path,residual=residual).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    start = 0
    if not residual:
        model, optimizer, start = utils.load_checkpoint(
                model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def load_existed_residual_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_residual_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    start = 0
    return model, optimizer, start




if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)


    cfg.data.frame_ids = args.frame_ids[:1]
    args.dump_images = True
    

    print("################################")
    print("--- Frame_ID:", args.frame_ids)
    print("################################")

    #wandb
    wandbrun = wandb.init(
            # set the wandb project where this run will be logged
            project="TeTriRF",
        
            # track hyperparameters and run metadata
            config={
            "configs": cfg,
            "args": args,
            },
            resume = "allow",
            id = 'render360_'+cfg.expname+f'_{args.qp}_{args.codec}',
        )



    if not hasattr(cfg.fine_model_and_render, 'dynamic_rgbnet'):
        cfg.fine_model_and_render.dynamic_rgbnet = True
    if not hasattr(cfg.fine_model_and_render, 'RGB_model'):
        cfg.fine_model_and_render.RGB_model = 'MLP'




    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)


    



    for frame_id in args.frame_ids:
        # load model for rendring
        
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            if args.reald:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, f"fine_last_{frame_id}.tar")
            else:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, f"raw_out/fine_last_{frame_id}.tar")
            #ckpt_path = os.path.join(cfg.basedir, cfg.expname, f"fine_last_{frame_id}.tar")

        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        model.reset_occupancy_cache()

        
        rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet.tar')
        if cfg.fine_model_and_render.dynamic_rgbnet:
            rgbnet_file = None
            rgbnet_files = [f for f in os.listdir(os.path.join(cfg.basedir, cfg.expname)) if f.endswith('.tar') and 'rgbnet' in f]
            if len(rgbnet_files)>0:
                
                for f in rgbnet_files:
                    beg = f.split('_')[1]
                    eend = f.split('_')[2].split('.')[0]
                    if frame_id <= int(eend) and frame_id>int(beg):
                        rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
                        break

                if rgbnet_file is None:
                    for f in rgbnet_files:
                        beg = f.split('_')[1]
                        eend = f.split('_')[2].split('.')[0]
                        if frame_id <= int(eend) and frame_id>=int(beg):
                            rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
                            break
            assert rgbnet_file is not None

                

        checkpoint =torch.load(rgbnet_file)
        model_kwargs = checkpoint['model_kwargs']
        if cfg.fine_model_and_render.RGB_model=='MLP':
            dim0 = model_kwargs['dim0']
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            rgbnet = RGB_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth)
        elif cfg.fine_model_and_render.RGB_model =='SH':
            dim0 = model_kwargs['dim0']
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            rgbnet = RGB_SH_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth, deg=2)

        rgbnet.load_state_dict(checkpoint['model_state_dict'])
        print('load rgbnet:', rgbnet_file )

        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
                'shared_rgbnet': rgbnet,
            },
        }


        if not cfg.data.ndc:
            # prepare for views
            render_poses=data_dict['poses'][data_dict['i_train']]
            render_poses = torch.tensor(render_poses).cpu()

            xyz_min_fine = model.xyz_min
            xyz_max_fine = model.xyz_max

            bbox = torch.stack([xyz_min_fine,xyz_max_fine]).cpu()

            center = torch.mean(bbox,dim=0)
            up = -torch.mean(render_poses[:,0:3,1],dim =0)
            up = up / torch.norm(up)
            
            #radius = torch.norm(render_poses[0,0:3,3] - center) * 0.9
            radius = torch.norm(render_poses[0,0:3,3] - center) * 2
            center = center -  up*radius*0.03

            v = torch.tensor([0,-1,1], dtype=torch.float32).cpu()
            v = v - up.dot(v)*up
            v = v / torch.norm(v)

            #
            #s_pos = center - v * radius + up*radius*0.26
            s_pos = center - v * radius + up*radius*0.04

            center = center.numpy()
            up = up.numpy()
            radius = radius.item()
            s_pos = s_pos.numpy()

            lookat = center - s_pos
            lookat = lookat/np.linalg.norm(lookat)

            xaxis = np.cross(lookat, up)
            xaxis = xaxis / np.linalg.norm(xaxis)

            def rodrigues_rotation_matrix(axis, theta):
                axis = np.asarray(axis)
                theta = np.asarray(theta)
                axis = axis/math.sqrt(np.dot(axis, axis))
                a = math.cos(theta/2.0)
                b, c, d = -axis*math.sin(theta/2.0)
                aa, bb, cc, dd = a*a, b*b, c*c, d*d
                bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
                return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

            num_viewangles = 360
            cntp = 0
            
            cam_poses = []
            sTs = []
            sKs = []
            HWs = []
            for view_id in range(num_viewangles):

   
   
                cntp = cntp +1

                i = view_id

                angle = 3.1415926*2*i/num_viewangles
                pos = (s_pos - center)/(1.05)
                pos = rodrigues_rotation_matrix(up,-angle).dot(pos) 
                pos = pos + center

                lookat = center - pos
                lookat = lookat/np.linalg.norm(lookat)

                xaxis = np.cross(lookat, up)
                xaxis = xaxis / np.linalg.norm(xaxis)

                yaxis = -np.cross(xaxis,lookat)
                yaxis = yaxis/np.linalg.norm(yaxis)

                nR = np.array([xaxis,yaxis,lookat, pos]).T
                nR = np.concatenate([nR,np.array([[0,0,0,1]])])

                sTs.append(nR)
                
                sKs.append(data_dict['Ks'][data_dict['i_train']][0])
                HWs.append(data_dict['HW'][data_dict['i_train']][0])
                    


                cam_poses.append(nR)
                


            sTs = np.stack(sTs)
            sKs = np.stack(sKs)
            HWs = np.stack(HWs)

            num_views = 2000
            view_idx = []
            view_frameid = []
            file_no = []
            start_360 = 600
            fid = 0


            for i in range(start_360):
                view_idx.append(0)
                view_frameid.append(fid%len(args.frame_ids))
                fid = fid + 1
                file_no.append(i)

            for i in range(start_360,start_360+360):
                view_idx.append(i-start_360)
                view_frameid.append(fid%len(args.frame_ids))
                fid = fid + 1
                file_no.append(i)

            for i in range(start_360+360,num_views):
                view_idx.append(0)
                view_frameid.append(fid%len(args.frame_ids))
                fid = fid + 1
                file_no.append(i)
        else:
            #ipdb.set_trace()

            render_poses = data_dict['render_poses']
            render_poses = np.pad(render_poses, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)
            render_poses[:,3,3] = 1


            view_idx = []
            view_frameid = []
            file_no = []

            for i in range(200):
                view_idx.append(i % len(render_poses))
                view_frameid.append(i)
                file_no.append(i)

            sTs = render_poses
            sKs = np.repeat(data_dict['Ks'][0][np.newaxis, :, :], len(render_poses), axis=0) 
            HWs = np.repeat(data_dict['HW'][0][np.newaxis, :], len(render_poses), axis=0)   



        view_frameid = np.array(view_frameid)
        view_idx = np.array(view_idx)
        file_no = np.array(file_no)

        json_file = []
        for i in range(len(file_no)):
            tmp = {'extrinsic': sTs[view_idx[i]].tolist(),'intrinsic': sKs[view_idx[i]].tolist(), 'frame_id': int(view_frameid[i]),'id':int(i)}
            json_file.append(tmp)


        json_string = json.dumps({'frames':json_file, 'HW': HWs[0].tolist()}, indent=4)


        if args.render_360>=0 :


            view_frameid = np.array(view_frameid)
            view_idx = np.array(view_idx)

            mask  = (view_frameid ==frame_id)
            masked_view = view_idx[mask]
            file_no_t = file_no[mask]


            sTs_t = sTs[masked_view]
            sKs_t = sKs[masked_view]
            HWs_t = HWs[masked_view]

            

            savedir = os.path.join(cfg.basedir, cfg.expname, f'render_360_%d' % args.render_360 )
            os.makedirs(savedir, exist_ok=True)


            rgbs, depths, bgmaps,res_psnr = render_viewpoints(
                    render_poses=torch.tensor(sTs_t).float(),
                    HW=HWs_t,
                    Ks=torch.tensor(sKs_t).float(),
                    gt_imgs=None,
                    savedir=None, dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    frame_id = frame_id, masks = None,
                    **render_viewpoints_kwargs)

            for i in trange(len(rgbs)):
                rgb8 = utils.to8b(rgbs[i])
                filename = os.path.join(savedir, f'{file_no_t[i]}.png')
                imageio.imwrite(filename, rgb8)

                rgb8 = utils.to8b(1 - depths[i] / np.max(depths[i]))
                filename = os.path.join(savedir, f'{file_no_t[i]}_depth.png')
                if rgb8.shape[-1]<3:
                    rgb8 = np.repeat(rgb8, 3, axis=-1)
                imageio.imwrite(filename, rgb8)



         
       

    filename = '/dev/shm/render360.sh'
    with open(filename,'w') as f:
        f.write(f'cd {savedir}\n')
        

        f.write(f"ffmpeg -y -framerate 25 -i %d.png -c:v libx264 -color_range pc  -qp 20  rgb.mp4\n")
        f.write(f"ffmpeg -y -framerate 25 -i %d_depth.png -c:v libx264 -color_range pc  -qp 20  depth.mp4\n")
        f.write(f'rm *.png\n')

    os.system(f"bash {filename}")


    with open(os.path.join(savedir,"cameras.json"), "w") as file:
        file.write(json_string)
    
        
    videos = []
    filename = os.path.join(savedir, f"rgb.mp4")
    videos.append(wandb.Video(filename, fps=25, caption="rgb"))

    filename = os.path.join(savedir, f"depth.mp4")
    videos.append(wandb.Video(filename, fps=25, caption="rgb"))


    wandbrun.log({"video": videos})

    wandbrun.save(os.path.join(savedir,"cameras.json"))

    print('Done')

