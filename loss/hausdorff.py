import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.filters import sobel
from sklearn.metrics import pairwise_distances
from loss.edge_loss import get_hot_enc, get_edge_img
from loss.dice import dice_loss_sq
def hdistance(X, Y, as_list=False):#hausdorff distance loss
    """hausdorff distance

    Parameters
    ----------
    X : `torch.tensor`
        input
    Y : `torch.tensor`
        target
    as_list : bool, optional
        return list, by default False

    Returns
    -------
    list/torch.tensor
        loss or list of loss
    """
    #not differentiable
    # edgeimgY = get_edg_c(get_hot_enc(Y))
    # sft = nn.Softmax2d()
    # edgeimgX = get_edg_c(sft(X))
    # edge_act_chann = torch.nonzero(edgeimgY, as_tuple=False)#edges per channel, actual
    # pred_bnd = torch.argmax(edgeimgX, dim=1)
    # pred_bnd_he = get_hot_enc(pred_bnd)#hot encoded
    # edge_pred_chann = torch.nonzero(pred_bnd_he, as_tuple=False)
    # pwd = nn.PairwiseDistance()
    # dist = pwd(edge_act_chann, edge_pred_chann)

    yh = get_hot_enc(Y)
    predl = torch.argmax(X, dim=1)
    predhe = get_hot_enc(predl)
    max_dist = []
    for hey, hex in zip(yh, predhe):#hot encoded y, hot encoded x
        y_edge = torch.stack(list(map(get_edge_img, hey)))
        x_edge = torch.stack(list(map(get_edge_img, hex)))
        diff_edge = y_edge - x_edge
        y_edge_not_capt = y_edge.clone()#the boundary pixels in y not captured
        y_edge_not_capt[diff_edge==0]=0
        x_edge_not_capt = x_edge.clone()#the boundary pixels in x not captured
        x_edge_not_capt[diff_edge==0]=0
        dist_max = 0
        #for each label
        for ind in [1, 2]:
            coord_diffx = torch.nonzero(x_edge_not_capt[ind, ...], as_tuple=False).detach().cpu()
            coord_diffy = torch.nonzero(y_edge_not_capt[ind, ...], as_tuple=False).detach().cpu()
            if not len(coord_diffx) or not len(coord_diffy):
                dist_max = np.float16(0)
                continue
            distance_mat = pairwise_distances(coord_diffx, coord_diffy)
            # consider incorrect pixels from pred bnd, take max of min distances
            min_distsx = np.min(distance_mat, axis=0) #min disstances from each pixel to those in actual bnd
            max_dist_fromx = np.max(min_distsx)
            min_distsy = np.min(distance_mat, axis=1) #min disstances from each pixel in actual bnd to pred
            max_dist_fromy = np.max(min_distsy)
            dist_max = max(max(max_dist_fromx, max_dist_fromy), dist_max)#compare from, fromy, and among both labels
        max_dist.append(dist_max.astype(np.float16))
    if as_list:
        return max_dist
    else:
        return sum(max_dist)/yh.shape[0]

def get_edg_c(tens):
    #get edges for each channel with simple convolution
    """`

    Parameters
    ----------
    tens : `torch.tensor`
        image

    Returns
    -------
    `torch.tensor`
        edge image
    """
    sobely1 = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobely2 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    sobelx1 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobelx2 = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    #depth = x.size()[1]
    if len(tens.shape)==3:
        tens = tens.float().unsqueeze(dim=1)
    channels = tens.size()[1]

    sobel_kernely1 = torch.tensor(sobely1, dtype=torch.float32).unsqueeze(0).expand(channels,1 , 3, 3)
    sobel_kernely2 = torch.tensor(sobely2, dtype=torch.float32).unsqueeze(0).expand(channels, 1, 3, 3)
    sobel_kernelx1 = torch.tensor(sobelx1, dtype=torch.float32).unsqueeze(0).expand(channels, 1, 3, 3)
    sobel_kernelx2 = torch.tensor(sobelx2, dtype=torch.float32).unsqueeze(0).expand(channels, 1, 3, 3)
    edgex1 = F.conv2d(tens, sobel_kernelx1.to(tens.device), stride=1, padding=1, groups=channels)#, groups=inter_x.size(1))
    edgex2 = F.conv2d(tens, sobel_kernelx2.to(tens.device), stride=1, padding=1, groups=channels)#, groups=inter_x.size(1))
    edgey1 = F.conv2d(tens, sobel_kernely1.to(tens.device), stride=1, padding=1, groups=channels)
    edgey2 = F.conv2d(tens, sobel_kernely2.to(tens.device), stride=1, padding=1, groups=channels)
    # all non-zero value locations are part of boundary

    #test with other norm also
    inst = nn.InstanceNorm2d(edgex1.shape[1])
    rl = nn.ReLU(inplace=True)
    edge = rl(inst(edgex1))+rl(inst(edgex2))+rl(inst(edgey1))+rl(inst(edgey2))
    return edge

def edge_conv_loss(probs, act):
    #find edge through simple conv, apply dice loss
    """

    Parameters
    ----------
    probs : `torch.tensor`
        logits
    act : `torch.tensor`
        target

    Returns
    -------
    `torch.tensor`
        dice loss
    """
    edgeX = get_edg_c(probs)
    edgeY = get_edg_c(get_hot_enc(act))
    return dice_loss_sq(edgeX, edgeY)

def get_edg_conv(tens, if_inst=False):
    """get edges by conv by sobel

    Parameters
    ----------
    tens : tensor
        array whose edges are to be found
    if_inst : bool, optional
        if instance norm to be applied, by default False

    Returns
    -------
    tensor
        with edge locations emphasized
    """
    sobely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobelx = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    #depth = x.size()[1]
    if len(tens.shape)==3:
        tens = tens.float().unsqueeze(dim=1)
    channels = tens.size()[1]

    sobel_kernely = torch.tensor(sobely, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
    sobel_kernelx = torch.tensor(sobelx, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
    edgex = F.conv2d(tens, sobel_kernelx.to(tens.device), stride=1, padding=1)#, groups=inter_x.size(1))
    edgey = F.conv2d(tens, sobel_kernely.to(tens.device), stride=1, padding=1)
    # all non-zero value locations are part of boundary
    edge = edgex+edgey
    #test with other norm also
    inst = nn.InstanceNorm2d(edge.shape[1])
    rl = nn.ReLU(inplace=True)
    if if_inst:
        edge = inst(edge)
        edge = rl(edge)
    return edge

def get_edg_conv(tens, if_inst=False):
    """get edges by conv by sobel

    Parameters
    ----------
    tens : tensor
        array whose edges are to be found
    if_inst : bool, optional
        if instance norm to be applied, by default False

    Returns
    -------
    tensor
        with edge locations emphasized
    """
    sobely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    #depth = x.size()[1]
    if len(tens.shape)==3:
        tens = tens.float().unsqueeze(dim=1)
    channels = tens.size()[1]

    sobel_kernely = torch.tensor(sobely, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
    sobel_kernelx = torch.tensor(sobelx, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
    edgex = F.conv2d(tens, sobel_kernelx.to(tens.device), stride=1, padding=1)#, groups=inter_x.size(1))
    edgey = F.conv2d(tens, sobel_kernely.to(tens.device), stride=1, padding=1)
    # all non-zero value locations are part of boundary
    edge = edgex+edgey
    #test with other norm also
    inst = nn.InstanceNorm2d(edge.shape[1])
    rl = nn.ReLU(inplace=True)
    if if_inst:
        edge = inst(edge)
        edge = rl(edge)
    return edge

def get_edge_ski(act1):
    """for 2d array

    Parameters
    ----------
    act1 : `torch.tensor`
        2d image

    Returns
    -------
    `numpy.array`
        edge image
    """
    border_img = sobel(act1.cpu())
    non0_inds = np.nonzero(border_img)
    edge_img = torch.zeros_like(act1)
    edge_img[non0_inds[0], non0_inds[1]] = 1
    return edge_img