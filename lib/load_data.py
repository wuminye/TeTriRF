import numpy as np

from .load_llff import load_llff_data, LLFF_Dataset
from .load_NHR import NHR_Dataset
from torch.utils.data import DataLoader
import ipdb
import tqdm
import torch

def load_data(args):

    K, depths = None, None
    near_clip = None

    frame_ids = None
    masks = None

    if args.dataset_type == 'llff':

        args.frame_ids.sort()

        if not hasattr(args, 'spherify'):
            args.spherify = False

        dataset = LLFF_Dataset(args.datadir, factor = args.factor, frameids = args.frame_ids, test_views = args.test_frames, spherify = args.spherify)
        def my_collate_fn(batch):
            # Separate the data and labels from each sample in the batch
            assert len(batch) ==1
            item = batch[0]
            data1 = item[0] 
            data2 = item[1] 
            data3 = item[2] 
            data4 = item[3] 

            
            # Return the collated data and labels as a single batch
            return data1, data2, data3, data4
        train_dataloader = DataLoader(dataset, batch_size=1,num_workers = 12, shuffle=False, collate_fn = my_collate_fn)

        frame_ids = []
        res_images = []
        res_poses = []
        res_render_poses = []
   
        for i, data in enumerate(train_dataloader):
            images_t, poses_t, render_poses_t, i_test_t = data
            P = images_t.shape[0]
            res_images.append(images_t)
            res_poses.append(poses_t)
            res_render_poses.append(render_poses_t)
            frame_ids.append(torch.ones(P, device='cpu')*args.frame_ids[i])

        res_images = np.concatenate(res_images,axis=0)
        res_poses = np.concatenate(res_poses,axis=0)
        res_render_poses = np.concatenate(res_render_poses,axis=0)
        frame_ids = torch.cat(frame_ids).long()

        images = res_images
        render_poses = res_render_poses
        poses = res_poses

        hwf = res_poses[0,:3,-1]
        poses = res_poses[:,:3,:4]
        print('Loaded llff', res_images.shape, res_render_poses.shape, hwf, args.datadir)
        
        test_frames = []
        for i in args.test_frames:
            for j in range(i,int(res_images.shape[0]),P):
                test_frames.append(j)
        #i_val = i_test
        i_train = np.array([i for i in np.arange(int(res_images.shape[0])) if i not in test_frames])
        i_test = test_frames
        i_val = i_test

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)

        





    elif args.dataset_type == 'NHR':
        args.frame_ids.sort()
        frame_id = args.frame_ids[-1]
        previous_frame_ids = args.frame_ids[:-1]
        tar_size = (args.height,args.width)
        isNHR = True
        if args.width>960:
            isNHR = False
        dataset = NHR_Dataset(args.datadir, frameids = args.frame_ids,test_views = args.test_frames, tar_size=tar_size, isNHR = isNHR)
        def my_collate_fn(batch):
            # Separate the data and labels from each sample in the batch
            assert len(batch) ==1
            item = batch[0]
            data1 = item[0] 
            data2 = item[1] 
            data3 = item[2] 
            data4 = item[3] 

            
            # Return the collated data and labels as a single batch
            return data1, data2, data3, data4
        train_dataloader = DataLoader(dataset, batch_size=1,num_workers = 12, shuffle=False, collate_fn = my_collate_fn)

        frame_ids = []
        res_images = []
        res_images_ori = []
        res_poses = []
        res_intrinsic = []
   
        for i, data in enumerate(train_dataloader):
            images_t, poses_t, intrinsic_t, images_ori_t = data
            P = images_t.size(0)
            res_images.append(images_t)
            res_images_ori.append(images_ori_t)
            res_poses.append(poses_t)
            res_intrinsic.append(intrinsic_t)
            frame_ids.append(torch.ones(P, device='cpu')*args.frame_ids[i])

        res_images = torch.cat(res_images,dim=0).numpy()
        res_images_ori = torch.cat(res_images_ori,dim=0).numpy()
        res_poses = np.concatenate(res_poses,axis=0)
        res_intrinsic = np.concatenate(res_intrinsic,axis=0)
        frame_ids = torch.cat(frame_ids).long()

        N=P

        #copy the list self.test_views to tmp
        tmp = []
        for i in dataset.test_views:
            for j in range(i,len(res_poses),N):
                tmp.append(j)
        dataset.test_views = tmp

        


        if len(dataset.test_views)==0:
            i_split = [np.arange(0, len(res_poses)) for i in range(3)]
        else:
            i_split = [[i for i in np.arange(0, len(res_poses)) if i not in dataset.test_views]]
            i_split.append(dataset.test_views)
            i_split.append(dataset.test_views)





        i_split.append(np.arange(0, P*len(previous_frame_ids)))  # replay data
        i_split.append(np.arange(P*len(previous_frame_ids), P*len(previous_frame_ids)+N))  # current data

        #i_split[0] =  i_split[0][::scale]

        images, poses, render_poses, hwf, K, i_split, frame_ids = res_images, res_poses, res_poses, [res_images.shape[1], res_images.shape[2], intrinsic_t[0,0,0]], res_intrinsic, i_split,frame_ids

        #images, poses, render_poses, hwf, K, i_split, frame_ids = dataset.load_data(frame_id, previous_frame_ids)
        print('Loaded NHR', images.shape, render_poses.shape, hwf, args.datadir, ' frame:', frame_id, ' previous:',previous_frame_ids)

        i_train, i_val, i_test, i_replay, i_current = i_split

        print('@@@@@@ training:', len(i_train), 'test:', len(i_test))

        near, far = inward_nearfar_heuristic(poses[i_current, :3, 3])

        near = near*1.1

        far = far*1.6

        assert images.shape[-1] == 4 or  images.shape[-1] == 3

        masks = images[...,-1:]

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*masks + (1.-masks)
            else:
                images = images[...,:3]
    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        frame_ids = frame_ids, masks= masks,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far

