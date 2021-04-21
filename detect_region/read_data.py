import torch
import os
import h5py
import random

import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import PIL.Image as Image
import itertools

from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
from itertools import islice
from data_read.meniscus_data import get_ind_no_medial
class MeniscusDataForDetect(Dataset):
    """read menisci data
    """
    def __init__(self, data_dir, num_img=None, in_channels=1, segm_layers=1, if_crop=False, crop_size=512, part=None, 
    second_start=600, no_medial_only=True, if_hdf5=False, is_crop_random=False, if_aug=True, only_cm=False):
        """create image, targets dataset suitable for mask rcnn, faster rcnn training

        Parameters
        ----------
        data_dir : path like
            directory containing image and segm numpy files, or the path of hdf5 file
        num_img : int, optional
            number of images total(if None, then total image in folder), by default None
        in_channels : int, optional
            channel depth of input image to be, by default 3
        segm_layers : int, optional
            layer count, by default 1
        if_crop : bool, optional
            whether to crop, by default False
        crop_size : int, optional
            crop to size, by default 256
        part : str, optional
            chosse among 'first','second' subsets, by default None
        second_start : int, optional
            second subset start index, by default 600
        no_medial_only : bool, optional
            exclude targets with missing masks, by default True
        if_hdf5 : bool, optional
            whether hdf5 file source, by default False
        is_crop_random:
            if image to be cropped for random images
        """
    #         self.hdfile= h5py.File(hd_img_file, 'r')
        self.only_cm = only_cm
        self.data_dir = data_dir # or path of hdf5 file, if_hdf5 is true
        self.part = part
        self.second_start = second_start
        self.resize_shape = crop_size
        self.trans = transforms.Compose([transforms.Resize((crop_size, crop_size)), transforms.ToTensor()])
        self.trans_resize = transforms.Compose([transforms.Resize((crop_size, crop_size), interpolation=Image.NEAREST)])
        self.trans_totens = transforms.Compose([transforms.ToTensor()])

        self.in_channels = in_channels
        self.segm_layers = segm_layers
        self.if_crop = if_crop
        self.if_hdf5 = if_hdf5
        self.is_crop_random = is_crop_random
        self.if_aug = if_aug
        if if_hdf5:
            self.hdfile= h5py.File(data_dir, 'r')
            # keys/slices with "slice" in them
            all_slice_keys = [x for x in list(self.hdfile.keys()) if 'Slice' in x]
            # # keys/slices with central meniscus regions
            # all_slice_keys = [x for x in list(self.hdfile.keys()) if 'Slice' in x and
            # len(np.unique(np.asarray(self.hdfile[x]['exportedSegMask'])))==3]
            # # keys/slices with both foreground labels, not including slices with only bg labels
            # all_slice_keys = [x for x in list(self.hdfile.keys()) if 'Slice' in x and
            # len(np.unique(np.asarray(self.hdfile[x]['exportedSegMask'])))==3 and 
            # len(np.nonzero(np.asarray(self.hdfile[x]['exportedSegMask'])==2)[0]) < 260 and
            # len(np.nonzero(np.asarray(self.hdfile[x]['exportedSegMask'])==1)[0]) < 260]
            self.all_slice_keys = all_slice_keys
            self.num_img = num_img
            if self.num_img is None:
                self.num_img = len(self.all_slice_keys)

            if part=="first":
                self.all_slice_keys = self.all_slice_keys[:second_start]
            elif part=="second":
                self.all_slice_keys = self.all_slice_keys[second_start:self.num_img]
        else:
            self.npy_dir = os.path.join(self.data_dir, "img_npy")
            self.seg_dir = os.path.join(self.data_dir, "seg_npy")
            self.all_npy_files = [f for f in os.listdir(self.npy_dir) if os.path.isfile(os.path.join(self.npy_dir, f)) 
            and "Slice" in os.path.join(self.npy_dir, f) and ".npy" in os.path.join(self.npy_dir, f)]
            self.npy_files = [x for x in self.all_npy_files if x not in get_ind_no_medial()] if no_medial_only else self.all_npy_files

            self.num_img = num_img
            if self.num_img is None:
                path, dirs, files = next(os.walk(self.data_dir+f"/img_npy"))
                self.num_img = len(self.npy_files)

            if part=="first":
                self.npy_files = self.npy_files[:second_start]
            elif part=="second":
                self.npy_files = self.npy_files[second_start:self.num_img]
    def __getitem__(self, ind):       
        # making 1 to 0001 etc
        if self.if_hdf5:
            img = np.asarray(self.hdfile[self.all_slice_keys[ind]]['normalizedImage'])
            segm = np.asarray(self.hdfile[self.all_slice_keys[ind]]['exportedSegMask'])
        else:
            npy_file = os.path.join(self.npy_dir, self.npy_files[ind])
            img = np.load(npy_file)

            seg_file = os.path.join(self.seg_dir, self.npy_files[ind])
            segm = np.load(seg_file)

        # for normalizing
        # img=img-img.min()
        #row min, row max, col min, col max
        #TODO change below to dynamic values
        # if self.only_cm:
        #     if len(np.unique(segm))!=3 or np.count_nonzero(segm==2)>300 or np.count_nonzero(segm==1)>300:
        #         segm = np.zeros_like(segm)
        if len(np.unique(segm))!=1:
            rmin, rmax, cmin, cmax = np.min(np.argwhere(segm!=0)[:,0]), np.max(np.argwhere(segm!=0)[:,0]), np.min(np.argwhere(segm!=0)[:,1]), np.max(np.argwhere(segm!=0)[:,1])
            #113, 425, 90, 402
        else:
            rmin, rmax, cmin, cmax = 0, segm.shape[0]-1, 0, segm.shape[1]-1
        if self.is_crop_random and len(np.unique(segm))!=1:
            self.if_crop = bool(random.getrandbits(1))
            rmin, rmax = np.random.randint(rmin), np.random.randint(rmax, img.shape[0])
            cmin, cmax = np.random.randint(cmin), np.random.randint(cmax, img.shape[1])

        if self.if_crop:
            img =img[rmin:rmax, cmin: cmax]
            segm = segm[rmin:rmax, cmin: cmax]

        mean = np.mean(img)
        std = np.std(img)
        if self.resize_shape!=img.shape[-1] or self.if_aug:
            img_pil=Image.fromarray(img)
            segm_pil=Image.fromarray(segm)

            # appyling input augmentation
            if self.if_aug:
                if random.random()>0.5:
                    rota = np.random.randint(20)
                    scale = round(random.uniform(0.8, 1.2),2)
                    shear = np.round(np.random.uniform(0, 4, size=2), 2)
                    translate = np.random.randint(2,size=2)
                    img_pil = TF.rotate(img_pil, rota)
                    segm_pil =TF.rotate(segm_pil, rota)
                    img_pil = TF.affine(img_pil, angle=rota, scale=scale, shear=list(shear), translate=list(translate))
                    segm_pil = TF.affine(segm_pil, angle=rota, scale=scale, shear=list(shear), translate=list(translate))
                # adding vetical flipping augmentation decreased dice score
                # if random.random()>0.5:
                #     img_pil = TF.vflip(img_pil)
                #     segm_pil = TF.vflip(segm_pil)
                if random.random()>0.5:
                    img_pil = TF.hflip(img_pil)
                    segm_pil = TF.hflip(segm_pil)

            img = self.trans(img_pil.copy()) 
            segm_pil = self.trans_resize(segm_pil.copy()) 
            #  to avoid scaling to [0,1] while converting to tensor, datatype uint8 is changed to float
            segm = np.array(segm_pil, np.float)
        else:
            img = self.trans_totens(img) 
            segm = segm.astype(np.float)

        segm = self.trans_totens(segm).long()

        if self.segm_layers!=1:
            sgm_zer=(torch.zeros(3,*segm.shape[1:3]))
            segm_one=sgm_zer.scatter(0, segm.long(), 1)
            segm = segm_one if self.segm_layers==3 else segm_one[1:]
        else:
            segm = segm.squeeze()

        norm = transforms.Normalize(mean, std)
        img = norm(img)

        if self.in_channels > 1:     
            img = torch.cat([img for _ in range(self.in_channels)], dim=0)

        # TODO why dim=1 not normalizing properly
        # img_norm=F.normalize(img)
        num_objs = torch.unique(segm)
        boxes = []
        labels = []
        masks = []
        for i in range(1,3):
            if i not in num_objs:
                continue
            pos = torch.where(segm==i)
            xmin = torch.min(pos[-1])
            xmax = torch.max(pos[-1])
            ymin = torch.min(pos[-2])
            ymax = torch.max(pos[-2])
            boxes.append(torch.stack([xmin, ymin, xmax, ymax]))
            labels.append(torch.tensor(i))

            #add mask to mask list
            mask1 = torch.zeros_like(segm)
            mask1[segm==i]=1
            masks.append(mask1)

        target = {}
        boxes = torch.stack(boxes)
        labels = torch.stack(labels)
        target['masks'] = torch.stack(masks)
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([ind])#[str(ind) for ind in range(len(labels))]#labels.clone()#torch.tensor([index])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:, 2] - boxes[:,0])
        area = torch.as_tensor(area, dtype=torch.float32)
        target['area'] = area
        target['iscrowd'] = torch.zeros((labels.shape[0],), dtype=torch.int64)
        return img, target, str(ind)

    def __len__(self):
        if self.part is None:
            return self.num_img
        elif self.part=="first":
            return self.second_start
        else:
            return self.num_img-self.second_start