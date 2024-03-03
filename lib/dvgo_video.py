import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import segment_coo
import ipdb
from . import grid
from torch.utils.cpp_extension import load
import copy
from .dvgo import DirectVoxGO
from .dmpigo import DirectMPIGO
from .sh import eval_sh

class RGB_Net(torch.nn.Module):
    def __init__(self,dim0=None, rgbnet_width=None, rgbnet_depth=None):
        super(RGB_Net, self).__init__()
        self.rgbnet = None

        if dim0 is not None and rgbnet_width is not None and rgbnet_depth is not None:
            self.set_params(dim0,rgbnet_width,rgbnet_depth)

    def set_params(self, dim0, rgbnet_width, rgbnet_depth):
        
        if self.rgbnet is None:
            self.dim0= dim0
            self.rgbnet_width = rgbnet_width
            self.rgbnet_depth = rgbnet_depth
            self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('***** rgb_net_ reset   *******')
        else:
            if self.dim0!=dim0 or self.rgbnet_width!=rgbnet_width or self.rgbnet_depth!=rgbnet_depth:
                ipdb.set_trace()
                raise Exception("Inconsistant parameters!")

        return lambda x: self.forward(x)

    def forward(self,x):
        if self.rgbnet is None:
            raise Exception("call set_params() first!")
        return self.rgbnet(x)

    def get_kwargs(self):
        return {
            'dim0': self.dim0,
            'rgbnet_width': self.rgbnet_width,
            'rgbnet_depth': self.rgbnet_depth
        }

class RGB_SH_Net(torch.nn.Module):
    def __init__(self,dim0=None, rgbnet_width=None, rgbnet_depth=None, deg = 3):
        super(RGB_SH_Net, self).__init__()
        self.rgbnet = None
        self.deg = deg

        if dim0 is not None and rgbnet_width is not None and rgbnet_depth is not None:
            self.set_params(dim0,rgbnet_width,rgbnet_depth,out_dim = 3*(self.deg+1)**2)

    def set_params(self, dim0, rgbnet_width, rgbnet_depth,out_dim):
        
        if self.rgbnet is None:
            self.dim0= dim0
            self.rgbnet_width = rgbnet_width
            self.rgbnet_depth = rgbnet_depth
            self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth-2)
                    ],
                    nn.Linear(rgbnet_width, out_dim),
                )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('***** rgb_net_SH reset   *******')
        else:
            if self.dim0!=dim0 or self.rgbnet_width!=rgbnet_width or self.rgbnet_depth!=rgbnet_depth:
                ipdb.set_trace()
                raise Exception("Inconsistant parameters!")

        return lambda x: self.forward(x)

    def forward(self,x, dirs):
        if self.rgbnet is None:
            raise Exception("call set_params() first!")
        coeffs = self.rgbnet(x)
        coeffs = coeffs.reshape(x.size(0),3,-1)
        #coeffs = x.reshape(x.size(0),3,-1)
        return torch.sigmoid(eval_sh(self.deg, coeffs, dirs))

    def get_kwargs(self):
        return {
            'dim0': self.dim0,
            'rgbnet_width': self.rgbnet_width,
            'rgbnet_depth': self.rgbnet_depth,
            'deg': self.deg,
        }

class DirectVoxGO_Video(torch.nn.Module):
    def __init__(self, frameids,xyz_min,xyz_max,cfg=None):
        super(DirectVoxGO_Video, self).__init__()

        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.frameids = frameids
        self.cfg = cfg
        self.dvgos = nn.ModuleDict()
        self.viewbase_pe = cfg.fine_model_and_render.viewbase_pe
        self.fixed_frame = []

        self.initial_models()


    def get_kwargs(self):
        return {
            'frameids': self.frameids,
            'xyz_min': self.xyz_min,
            'xyz_max': self.xyz_max,
            'viewbase_pe': self.viewbase_pe,
        }

    def initial_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_kwargs = copy.deepcopy(self.cfg.fine_model_and_render)
        num_voxels = model_kwargs.pop('num_voxels')
        cfg_train = self.cfg.fine_train

        coarse_ckpt_path = None
        

        if len(cfg_train.pg_scale):
            num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

        for frameid in self.frameids:
            coarse_ckpt_path = os.path.join(self.cfg.basedir, self.cfg.expname, f'coarse_last_{frameid}.tar')
            if not os.path.isfile(coarse_ckpt_path):
                coarse_ckpt_path = None
            frameid = str(frameid)
            print(f'model create: frame{frameid}')

            #k0_config = {'factor': self.cfg.fine_model_and_render.plane_scale}
            if self.cfg.data.ndc:
                
                self.dvgos[frameid] = DirectMPIGO(
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    num_voxels=num_voxels, 
                    mask_cache_path=coarse_ckpt_path,  
                    **model_kwargs)
            else:
                self.dvgos[frameid] = DirectVoxGO(
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    num_voxels=num_voxels, 
                    mask_cache_path=coarse_ckpt_path, rgb_model = self.cfg.fine_model_and_render.RGB_model,
                    **model_kwargs)
            self.dvgos[frameid] = self.dvgos[frameid].to(device)

        # 使用RGBNet来创建self.rgbnet变量，模型的初始化参数在model_kwargs中获取
        if self.cfg.fine_model_and_render.RGB_model=='MLP':
            dim0 = (3+3*self.viewbase_pe*2) + self.cfg.fine_model_and_render.rgbnet_dim
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            self.rgbnet = RGB_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth)
        elif self.cfg.fine_model_and_render.RGB_model =='SH':
            dim0 = self.cfg.fine_model_and_render.rgbnet_dim
            rgbnet_width = model_kwargs['rgbnet_width']
            rgbnet_depth = model_kwargs['rgbnet_depth']
            self.rgbnet = RGB_SH_Net(dim0=dim0, rgbnet_width=rgbnet_width, rgbnet_depth=rgbnet_depth, deg=2)
            
        
        
        print('*** models creation completed.',self.frameids)

    def load_checkpoints(self):

        cfg = self.cfg
        ret = []
        for frameid in self.frameids:
            frameid = str(frameid)
            last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last_{frameid}.tar')
            if not os.path.isfile(last_ckpt_path):
                print(f"Frame {frameid}'s checkpoint doesn't exist")
                continue
            ckpt = torch.load(last_ckpt_path)

            #根据ckpt中的model_kwargs来创建模型
            model_kwargs = ckpt['model_kwargs']
            if self.cfg.data.ndc:
                self.dvgos[frameid] = DirectMPIGO(**model_kwargs)
            else:
                self.dvgos[frameid] = DirectVoxGO(**model_kwargs)
            self.dvgos[frameid].load_state_dict(ckpt['model_state_dict'], strict=True)
            self.dvgos[frameid] = self.dvgos[frameid].cuda()
            print(f"Frame {frameid}'s checkpoint loaded.")
            ret.append(int(frameid))
            break

        beg = self.frameids[0]
        eend = self.frameids[-1]

        if self.cfg.fine_model_and_render.dynamic_rgbnet:
            rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
        else:
            rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet.tar')
        
        if not os.path.isfile(rgbnet_file):
            #获取目录中所有名字中包含rgbnet的tar类型文件，参照命名规则分析文件名中的beg和eend值，选取beg和eend最大的那个文件为rgbnet_file
            rgbnet_files = [f for f in os.listdir(os.path.join(cfg.basedir, cfg.expname)) if f.endswith('.tar') and 'rgbnet' in f]
            if len(rgbnet_files)>0:
                beg = -1
                eend = -1
                for f in rgbnet_files:
                    beg = max(beg, int(f.split('_')[1]))
                    eend = max(eend, int(f.split('_')[2].split('.')[0]))
                rgbnet_file = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
            
        if os.path.isfile(rgbnet_file):
            checkpoint =torch.load(rgbnet_file)
            self.rgbnet.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
            print('load RGBNet', rgbnet_file)
        return ret

    def save_checkpoints(self):
        cfg = self.cfg
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid = str(frameid)
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'fine_last_{frameid}.tar')
            ckpt = {
                'model_state_dict': self.dvgos[frameid].state_dict(),
                # 加入模型的初始化参数
                'model_kwargs': self.dvgos[frameid].get_kwargs(),
            }
            torch.save(ckpt, ckpt_path)
            print(f"Frame {frameid}'s checkpoint saved to {ckpt_path}")


        beg = self.frameids[0]
        eend = self.frameids[-1]

        if self.cfg.fine_model_and_render.dynamic_rgbnet:
            rgbnet_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'rgbnet_{beg}_{eend}.tar')
        else:
            rgbnet_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'rgbnet.tar')
        rgbnet_ckpt = {
            'model_state_dict': self.rgbnet.state_dict(),
            # Add any other necessary information to the checkpoint dictionary
            'model_kwargs': self.rgbnet.get_kwargs(),
        }
        torch.save(rgbnet_ckpt, rgbnet_ckpt_path)
        print(f"RGBNet checkpoint saved to {rgbnet_ckpt_path}")

    def set_fixedframe(self, ids):
        self.fixed_frame = ids
        
        if len(ids)>0:
            frameid = -1
            for frameid in self.frameids:
                if frameid not in self.fixed_frame:
                    break
            assert frameid!=-1
            source_id = ids[0]
            xy_plane, xz_plane, yz_plane = self.dvgos[str(source_id)].k0.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)
            
            density_grid  = self.dvgos[str(source_id)].density.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)
            for frameid in self.frameids:
                if frameid in self.fixed_frame:
                    continue
                device = self.dvgos[str(frameid)].k0.xy_plane.device
                if self.cfg.fine_train.initialize_feature:
                    self.dvgos[str(frameid)].k0.xy_plane = nn.Parameter(xy_plane.clone()).to(device)
                    self.dvgos[str(frameid)].k0.xz_plane = nn.Parameter(xz_plane.clone()).to(device)
                    self.dvgos[str(frameid)].k0.yz_plane = nn.Parameter(yz_plane.clone()).to(device)
                if self.cfg.fine_train.initialize_density:
                    self.dvgos[str(frameid)].density.grid = nn.Parameter((density_grid.clone()*1.0 + 0*torch.randn_like(density_grid))).to(device) 

                print(f'Initialize  frame:{frameid}')



    #-------
    #rays_o：（N,3） rays_d：（N,3）  viewdirs：（N,3）  frame_ids：（N,1）, pytorch tensor
    # frame_ids indicate the frame id of each ray
    def forward(self, rays_o, rays_d, viewdirs, frame_ids, global_step=None, mode='feat', **render_kwargs):
        

        # find unique frame ids
        frame_ids_unique = torch.unique(frame_ids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1

        frameid = frame_ids_unique[0]


        # seperate rays into different frames according to frame_ids_unique, and feed them into different models in self.dvgos
        # then concat the results for each key in the returned dict

        ret_frame = self.dvgos[str(frameid)](rays_o, rays_d, viewdirs, shared_rgbnet= self.rgbnet, global_step=global_step, mode=mode, **render_kwargs)


        return ret_frame



    def scale_volume_grid(self, scale):
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid = str(frameid)
            self.dvgos[frameid].scale_volume_grid(scale)


    def density_total_variation_add_grad(self,  weight, dense_mode, frameids):
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        frameid = frame_ids_unique[0]
        frameid = str(frameid)
        self.dvgos[frameid].density_total_variation_add_grad(weight, dense_mode)

    def k0_total_variation_add_grad(self,  weight, dense_mode, frameids):
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        frameid = frame_ids_unique[0]
        frameid = str(frameid)
        self.dvgos[frameid].k0_total_variation_add_grad(weight, dense_mode)



    def compute_k0_l1_loss(self, frameids):
        frame_ids_unique = torch.unique(frameids, sorted=True).cpu().int().numpy().tolist()   
        assert len(frame_ids_unique) == 1
        loss = 0
        N =0
        frameid = frame_ids_unique[0]
        if (str(frameid-1) in self.dvgos):
            frameid2 = str(frameid-1)
            if not self.dvgos[str(frameid)].k0.xy_plane.size() == self.dvgos[frameid2].k0.xy_plane.size():
                xy_plane, xz_plane, yz_plane = self.dvgos[frameid2].k0.scale_volume_grid_value(self.dvgos[str(frameid)].world_size)

                loss += 2*F.l1_loss(self.dvgos[str(frameid)].k0.xy_plane, xy_plane)
                loss += 2*F.l1_loss(self.dvgos[str(frameid)].k0.xz_plane, xz_plane)
                loss += 2*F.l1_loss(self.dvgos[str(frameid)].k0.yz_plane, yz_plane)
                N=N+3
            else:
                loss += F.l1_loss(self.dvgos[str(frameid)].k0.xy_plane, self.dvgos[frameid2].k0.xy_plane)
                loss += F.l1_loss(self.dvgos[str(frameid)].k0.xz_plane, self.dvgos[frameid2].k0.xz_plane)
                loss += F.l1_loss(self.dvgos[str(frameid)].k0.yz_plane, self.dvgos[frameid2].k0.yz_plane)
                loss += 5*F.l1_loss(self.dvgos[str(frameid)].density.grid, self.dvgos[frameid2].density.grid)
                N+=4
        if str(frameid+1) in self.dvgos:
            frameid2 = str(frameid+1)
            loss += F.l1_loss(self.dvgos[str(frameid)].k0.xy_plane, self.dvgos[frameid2].k0.xy_plane)
            loss += F.l1_loss(self.dvgos[str(frameid)].k0.xz_plane, self.dvgos[frameid2].k0.xz_plane)
            loss += F.l1_loss(self.dvgos[str(frameid)].k0.yz_plane, self.dvgos[frameid2].k0.yz_plane)
            loss += 5*F.l1_loss(self.dvgos[str(frameid)].density.grid, self.dvgos[frameid2].density.grid)
            N+=4
        if N == 0:
            return loss
        return loss/N


    def update_occupancy_cache(self):
        res = []
        for frameid in self.frameids:
            if frameid in self.fixed_frame:
                continue
            frameid = str(frameid)
            res.append(self.dvgos[frameid].update_occupancy_cache())
        return np.mean(res)

    

       

        