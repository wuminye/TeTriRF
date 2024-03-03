import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import ipdb
import PIL
from PIL import Image
import collections
import math
import torchvision.transforms as T
import tqdm
from multiprocessing import Pool, Manager, Process


def imread_using_pillow(image_path, flags=cv2.IMREAD_UNCHANGED):
    """
    读取图像使用Pillow库并模仿cv2.imread的行为。

    参数:
    - image_path (str): 图像的路径。
    - flags (int): 读取模式。默认为cv2.IMREAD_UNCHANGED。

    返回:
    - numpy.ndarray: 图像的数组表示。
    """
    # 使用Pillow读取图像
    image = Image.open(image_path)

    if flags == cv2.IMREAD_UNCHANGED:
        # 如果图像有alpha通道，则保留alpha通道
        if image.mode == 'RGBA':
            image_array = np.array(image)
            # 把RGB通道转为BGR
            image_array = image_array[:, :, [2, 1, 0, 3]]
            return image_array
        # 如果图像是RGB，转换为BGR
        elif image.mode == 'RGB':
            return np.array(image)[:, :, ::-1]
        # 如果是灰度或其他模式，直接返回
        else:
            return np.array(image)
    
    # 对于cv2.IMREAD_GRAYSCALE
    if flags == cv2.IMREAD_GRAYSCALE:
        return np.array(image.convert('L'))

    # 如果是其他flags或未被特别指定的情况，返回转换为BGR的图像
    if image.mode == 'RGB':
        return np.array(image)[:, :, ::-1]
    else:
        return np.array(image)

# 例子
# image_data = imread_using_pillo

class Image_Transforms(object):
    def __init__(self, size, interpolation=Image.BICUBIC, is_center = False,  isNHR = True):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.is_center = is_center
        self.isNHR = isNHR
        
    def __call__(self, img, Ks , Ts ,  mask = None, residual =None):

        K = Ks
        Tc = Ts

        

        img = Image.fromarray(img.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.astype('uint8'), 'RGB')
        

        img_np = np.asarray(img)
        

        width, height = img.size

        translation = [0,0]
        ration = 1.0
        if self.is_center:
            translation = [width /2-K[0,2],height/2-K[1,2]]
            translation = list(translation)
            ration = 1.05
            
            if (self.size[1]/2)/(self.size[0]*ration  / height) - K[0,2] != translation[0] :
                ration = 1.2

            if not self.isNHR:
                ration = 1.0
            translation[1] = (self.size[0]/2)/(self.size[0]*ration  / height) - K[1,2]
            translation[0] = (self.size[1]/2)/(self.size[0]*ration  / height) - K[0,2]
            translation = tuple(translation)
        
   
        img = T.functional.affine(img, angle = 0, translate = translation, scale= 1,shear=0)
        img = T.functional.crop(img, 0, 0,  int(height/ration),int(height*self.size[1]/ration/self.size[0]) )

        img_ori = img
        img = T.functional.resize(img, self.size, self.interpolation)
        img = T.functional.to_tensor(img)
        img = img.permute(1,2,0)


        img_ori = T.functional.resize(img_ori, [1080,1920], self.interpolation)
        img_ori = T.functional.to_tensor(img_ori)
        img_ori = img_ori.permute(1,2,0)

        
        ROI = np.ones_like(img_np)*255.0

        ROI = Image.fromarray(np.uint8(ROI))
        ROI = T.functional.affine(ROI, angle = 0, translate = translation, scale= 1,shear=0)
        ROI = T.functional.crop(ROI, 0,0, int(height/ration),int(height*self.size[1]/ration/self.size[0]) )
        ROI = T.functional.resize(ROI, self.size, self.interpolation)
        ROI = T.functional.to_tensor(ROI)
        ROI = ROI[0:1,:,:]
        
        
        
        
        if mask is not None:
            mask = T.functional.affine(mask, angle = 0, translate = translation, scale= 1,shear=0)
            mask = T.functional.crop(mask, 0, 0,  int(height/ration),int(height*self.size[1]/ration/self.size[0]) )
            mask = T.functional.resize(mask, self.size, self.interpolation)
            mask = T.functional.to_tensor(mask)
            mask = mask.permute(1,2,0)
            mask = mask[:,:,0:1]

        if residual is not None:
            residual = T.functional.affine(residual, angle = 0, translate = translation, scale= 1,shear=0)
            residual = T.functional.crop(residual, 0, 0,  int(height/ration),int(height*self.size[1]/ration/self.size[0]) )
            residual = T.functional.resize(residual, self.size, self.interpolation)
            residual = T.functional.to_tensor(residual)
            residual = residual.permute(1,2,0)
            residual = residual[:,:,0:1]


        K[0,2] = K[0,2] + translation[0]
        K[1,2] = K[1,2] + translation[1]

        s = self.size[0] * ration / height

        K = K*s

        K[2,2] = 1  
                
 
        return img, K, Tc, mask, residual, ROI, img_ori
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


def process_frame(f, basedir, transforms):
    f_path = os.path.join(basedir, f['file'])
    f_path_mask = os.path.join(basedir, f['mask'])
    
    view_id = int(f['file'].split('_')[1].split('.')[0])
    frame_id = int(f['file'].split('/')[1])

    # there are non-exist paths in fox...
    if not os.path.exists(f_path) or not os.path.exists(f_path_mask):
        print(f"{f_path} or {f_path_mask} doesn't exist.")
        return None
    
    pose = np.array(f['extrinsic'], dtype=np.float32)  # [4, 4]
    K = np.array(f['intrinsic'], dtype=np.float32)

    mask = cv2.imread(f_path_mask)
    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img, K, Tc, mask, _, ROI = transforms(image, K, pose, mask, None)

    return {'Tc': Tc, 'img': img, 'K': K, 'mask': mask}




def wrapper(args):
    your_class_instance, id, res_images, res_images_ori, res_poses, res_intrinsic, frame_ids = args
    your_class_instance.read_frame_and_append(id, res_images, res_images_ori, res_poses, res_intrinsic, frame_ids)

class NHR_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, frameids=[], test_views = [], tar_size=(720,960), cam_num = -1, isNHR=True):
        super().__init__()
        self.cam_num = cam_num
        self.frameids = frameids
        self.path = path
        self.transforms = Image_Transforms(tar_size, isNHR = isNHR)
        self.test_views = test_views
        self.isNHR = isNHR


    def read_frame(self,frame_id, cam_num = -1):
        transform_path = os.path.join(self.path, 'cams_%d.json' % frame_id)
        with open(transform_path, 'r') as f:
            transform = json.load(f)

        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file'])



        if cam_num<0:
            cameras = [i for i in range(len(frames))]
        else:
            cameras = torch.randperm(len(frames), device='cpu')
            if cam_num>0:
                cameras = cameras[:cam_num]

        poses = []
        images = []
        intrinsic =[]
        masks =[]
        images_ori = []

        
        for id in cameras:
            f = frames[id]
            f_path = os.path.join(self.path, f['file'])
            f_path_mask = os.path.join(self.path, f['mask'])

            #view_id = int(f['file'].split('_')[1].split('.')[0])
            view_id = id
            #frame_id = int(f['file'].split('/')[1] if f['file'].split('/')[1]!='run' else f['file'].split('/')[2])


            # there are non-exist paths in fox...
            if not os.path.exists(f_path):
                print(f_path, "doesn't exist.")
                continue
            if not os.path.exists(f_path_mask):
                print(f_path_mask, "doesn't exist.")
                continue
            
   
            pose = (np.array(f['extrinsic'], dtype=np.float32)) # [4, 4]
            K = np.array(f['intrinsic'], dtype=np.float32)

            #mask = imageio.imread(f_path_mask).astype(np.uint8)
            mask = cv2.imread(f_path_mask)
            
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) 
            #image = imageio.imread(f_path).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img, K, Tc, mask, _, ROI, img_ori = self.transforms(image, K, pose,mask, None)

            images_ori.append(img_ori)
            poses.append(Tc)
            images.append(img)
            intrinsic.append(K)
            masks.append(mask)


        poses = np.stack(poses, axis=0).astype(np.float32)
        intrinsic = np.stack(intrinsic, axis=0).astype(np.float32)

        images = torch.stack(images)
        images_ori = torch.stack(images_ori)
        masks = torch.stack(masks)
        
        
        images = torch.cat([images, masks],dim = -1).float()
        #images = images.permute(0,3,1,2)
        #images = images[:,0:3,:,:] * images[:,3:4,:,:].repeat(1,3,1,1) + torch.ones_like(images[:,0:3,:,:],device = images.device)*(1.0-images[:,3:4,:,:].repeat(1,3,1,1))
        return images,poses,intrinsic, images_ori

    def __len__(self):
        return len(self.frameids)

    def __getitem__(self, idx):

        frame_id = self.frameids[idx]
        #frame_id = 0

        #while frame_id==15:
        #    frame_id = random.randint(0,100)

        images_t, poses_t, intrinsic_t, images_ori_t = self.read_frame(frame_id,cam_num = self.cam_num)
        print('** Finish data loading.', frame_id)

        return images_t, poses_t, intrinsic_t, images_ori_t

    def read_frame_and_append(self, id, res_images, res_images_ori, res_poses, res_intrinsic, frame_ids):
        images_t, poses_t, intrinsic_t, images_ori_t = self.read_frame(id, cam_num=-1)
        res_images.append(images_t)
        res_images_ori.append(images_ori_t)
        res_poses.append(poses_t)
        res_intrinsic.append(intrinsic_t)
        frame_ids.append(torch.ones(self.P, device='cpu') * id)


    def load_data(self, current_id, previous_ids, scale = 1.0):

        scale = int(scale)
        images,poses,intrinsic,images_ori = self.read_frame(current_id,cam_num = -1)

        N = images.size(0)
        self.P = N

        previous_ids.sort()

        frame_ids = []

        res_images = []
        res_images_ori = []
        res_poses = []
        res_intrinsic = []

        
        for id in previous_ids:
            images_t,poses_t,intrinsic_t,images_ori_t = self.read_frame(id,cam_num = -1)
            res_images.append(images_t)
            res_images_ori.append(images_ori_t)
            res_poses.append(poses_t)
            res_intrinsic.append(intrinsic_t)
            frame_ids.append(torch.ones(self.P, device='cpu')*id)




        res_images_ori.append(images_ori)
        res_images.append(images)
        res_poses.append(poses)
        res_intrinsic.append(intrinsic)
        frame_ids.append(torch.ones(N, device='cpu')*current_id)
        


        res_images = torch.cat(res_images,dim=0).numpy()
        res_images_ori = torch.cat(res_images_ori,dim=0).numpy()
        res_poses = np.concatenate(res_poses,axis=0)
        res_intrinsic = np.concatenate(res_intrinsic,axis=0)
        frame_ids = torch.cat(frame_ids).long()

        #copy the list self.test_views to tmp
        tmp = []
        for i in self.test_views:
            for j in range(i,len(res_poses),N):
                tmp.append(j)
        self.test_views = tmp

        


        if len(self.test_views)==0:
            i_split = [np.arange(0, len(res_poses)) for i in range(3)]
        else:
            i_split = [[i for i in np.arange(0, len(res_poses)) if i not in self.test_views]]
            i_split.append(self.test_views)
            i_split.append(self.test_views)





        i_split.append(np.arange(0, self.P*len(previous_ids)))  # replay data
        i_split.append(np.arange(self.P*len(previous_ids),self.P*len(previous_ids)+N))  # current data

        i_split[0] =  i_split[0][::scale]


        return res_images, res_poses, res_poses, [res_images.shape[1], res_images.shape[2], intrinsic[0,0,0]], res_intrinsic, i_split,frame_ids



