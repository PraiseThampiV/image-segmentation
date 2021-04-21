import numpy as np
from torch import nn
from sklearn.neighbors import NearestNeighbors
import torch
import random
def add_aug_deg(img, segm):
    #add dgeentative appearance
    #0-bumps or small clusters of spots
    #1-fully brightly colored
    #2-bright lines
    #3-bright spots
    which_aug = random.randint(0, 3)
    #everywhere
    meni_label = random.choices(np.unique(segm)[1:],  k=1)#choosing random meniscus for augment
    change_loc = np.argwhere(segm==meni_label)
    #for spots
    if which_aug==3:
        # pos_change = random.choices(range(len(change_loc)), k=len(change_loc)//10)
        # change_loc = np.array(change_loc[pos_change])
        change_loc=random.sample(list(change_loc), k=len(change_loc)//10)
        change_loc=np.array(change_loc)

    #for lines
    if which_aug==2:
        two_pts = random.sample(list(change_loc), k=2)#gettingn just two points
        change_loc = connect(np.array(two_pts))

    #for clusters
    if not which_aug:
        len_pts = len(change_loc)
        #getting n neighbours near to each point, TODO, find only for one random point
        nbrs = NearestNeighbors(n_neighbors=len_pts//10, algorithm='ball_tree').fit(change_loc)
        _, indices = nbrs.kneighbors(change_loc)
        #shape of index: number of points X n (indices of neighbours, first one-the point itself)
        change_loc_inds = indices[random.randint(0, len_pts-1)]
        change_loc = change_loc[change_loc_inds]
    bright_vals = np.random.uniform(low=img.mean(), high=img.max(), size=(len(change_loc),))#getting ranodm values in the range of those in image array
    img[change_loc[:,0], change_loc[:, 1]] = bright_vals
    return img

def dark_fr(img, seg):#create_dark_fringes
    """a type of data augmentation, menisci to have unclear fringes and darker outline almost merging to next regions 

    Parameters
    ----------
    img : `numpy.array`
        image
    seg : `numpy.array`
        segmentation mask
    """
def connect(ends):
    # get all coordinates between two given points
    """

    Parameters
    ----------
    ends : `numpy.array`
        array of shape (2,2)

    Returns
    -------
    list
        coordinates
    """
    d0, d1 = np.diff(ends, axis=0)[0]
    if not d0 or not d1:
        if not d0:
            return np.c_[np.repeat(ends[0,0], abs(d1)+1), np.arange(min(ends[0,1], ends[1,1]), max(ends[0,1], ends[1,1])+1)]
        else:
            return np.c_[np.arange(min(ends[0,0], ends[1,0]), max(ends[0,0], ends[1,0])+1), np.repeat(ends[0,1], abs(d0)+1)]
    if np.abs(d0) > np.abs(d1): 
        return np.c_[np.arange(ends[0, 0], ends[1,0] + np.sign(d0), np.sign(d0), dtype=np.int32),
                     np.arange(ends[0, 1] * np.abs(d0) + np.abs(d0)//2,
                               ends[0, 1] * np.abs(d0) + np.abs(d0)//2 + (np.abs(d0)+1) * d1, d1, dtype=np.int32) // np.abs(d0)]
    else:
        return np.c_[np.arange(ends[0, 0] * np.abs(d1) + np.abs(d1)//2,
                               ends[0, 0] * np.abs(d1) + np.abs(d1)//2 + (np.abs(d1)+1) * d0, d0, dtype=np.int32) // np.abs(d1),
                     np.arange(ends[0, 1], ends[1,1] + np.sign(d1), np.sign(d1), dtype=np.int32)]
def get_roinp(sgm, margin=10):
    """get coordinates of region of foreground labels

    Parameters
    ----------
    sgm : `numpy.array`
        target
    margin : int, optional
        margin, by default 10

    Returns
    -------
    list
        box coordinated
    """
    pos = np.nonzero(sgm)
    xmin = np.min(pos[-1])
    xmax = np.max(pos[-1])
    ymin = np.min(pos[-2])
    ymax = np.max(pos[-2])
    #TODO check if within boundary 
    return [xmin-margin, ymin-margin, xmax+margin, ymax+margin]

def get_roi_tens(sgm, margin=10, label=None):
    """input is tensor

    Parameters
    ----------
    sgm : `torch.tensor`
        mask
    margin : int, optional
        margin, by default 10
    label : int, optional
        label, by default None

    Returns
    -------
    list
        coordinates
    """
    if label is not None:
        pos = torch.where(sgm==label)
    else:
        pos = torch.where(sgm)
    xmin = torch.min(pos[-1])
    xmax = torch.max(pos[-1])
    ymin = torch.min(pos[-2])
    ymax = torch.max(pos[-2])
    #check if within boundary 
    xmin=max(xmin-margin, 0)
    ymin=max(ymin-margin, 0)
    xmax = min(xmax+margin, sgm.shape[-1]-1)
    ymax = min(ymax+margin, sgm.shape[-2]-1)
    return [xmin, ymin, xmax, ymax]

def get_patch(ends, img, crop_size, margin=0, is_long=False, send_ends=False):
    """get patch of roi from given ends, resize to `crop_size` and send it,
    size wrt the width of roi

    Parameters
    ----------
    ends : list
        coordinates
    img : `torch.tensor`
        image
    crop_size : int
        crop to size
    """
    xmin, ymin, xmax, ymax = ends
    halfwidth = (xmax-xmin)//2
    ymid = (ymin+ymax)//2
    ymin = ymid - halfwidth
    ymax = ymid + halfwidth
    ends = [xmin, ymin, xmax, ymax]
    img1 = img[..., ymin: ymax, xmin: xmax]
    # segm1 = segm[..., ymin: ymax, xmin: xmax]
    if is_long:
        upsamlong = nn.Upsample(size=[crop_size, crop_size], mode='nearest')
        patch = upsamlong(img1.float().view(1, 1, img1.shape[-2], img1.shape[-1])).squeeze().long()#to ensure only int values
    else:
        imgupsam = nn.Upsample(size=[crop_size, crop_size], mode='bilinear', align_corners=False)
        patch = imgupsam(img1.unsqueeze(dim=0)).squeeze(dim=0)
    if send_ends:
        return patch, ends
    else:
        return patch