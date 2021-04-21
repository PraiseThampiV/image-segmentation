import random, os
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from functools import partial
from loss.dice import dice_coefficient
# implementing test time aug

def aug_img(batchimgs, trans):
    """augment image with the desired transformation

    Parameters
    ----------
    batch : `torch.tensor`
        batch of images
    trans : function
        transform function
    param : list
        param for transform

    Returns
    -------
    `torch.tensor`
        augmented image
    """
    batch = batchimgs.clone()
    topil = transforms.ToPILImage()
    totens = transforms.ToTensor()
    bat, channel, h, w = batch.shape
    bat_flat = batch.flatten(end_dim=1) 
    img_pil = [topil(ele.numpy()) for ele in bat_flat] #input image is in fact one channel but 3 times repeated, tonp to avoid scaling
    img_aug = [totens(trans(image)) for image in img_pil]
    img_aug = torch.stack(img_aug)
    img_aug = img_aug.view((bat, channel, h, w))
    return img_aug

def choose_func(func_ind, param):
    """return partial func with appropriate param
    0-rotation, 1-shear, 2-hflip, 3-scale

    Parameters
    ----------
    func_ind : int
        0-rotate, 1-shear, 2-hflip, 3-scale
    angle : int, optional
        angle, by default 0
    scale : int, optional
        scale, by default 1
    shear : list, optional
        shear values, by default [0,0]
    """
    func_list=[partial(TF.rotate, angle=param),
    partial(TF.affine, angle=0, scale=1, shear=param, translate=list([0,0])),
    TF.hflip,
    partial(TF.affine, angle=0, scale=param, shear=0, translate=list([0,0]))]
    return func_list[func_ind]

def varres(batch, model, func_ind, param, axes=None, index_disp=0):#variant results
    """receives img, transform, pass it through model, reverse transform, return result

    Parameters
    ----------
    img :  `torch.tensor`
        image
    func_ind:list
        0-rotation, 1-shear, 2-hflip, 3-scale
    """
    #flatten image volume to BtimesC X H X W, and then reshape after aug
    trans = choose_func(func_ind, param)
    img1 = aug_img(batch, trans)

    model.eval()
    with torch.no_grad():
        out = model(img1)

    if axes is not None:
        im = axes[0].imshow(torch.argmax(out, dim=1)[index_disp])
    #reverse aug
    if func_ind==3:#for scale
        param = 1/param
    else:
        if param is not None:#for shear, rotation
            param = [-ele for ele in param] if isinstance(param, list) else -param

    trans = choose_func(func_ind, param)
    out = aug_img(out, trans)

    if axes is not None:
        im = axes[1].imshow(torch.argmax(out, dim=1)[index_disp])
    return out

def aug_imgs(model, batch, chooseaugs=[0,1,2,3], rand_angle=10, angle=None, save_dir=None):
    """wrapper for augmentation methods

    Parameters
    ----------
    model : `torch.nn.module`
        model
    batch : `torch.tensor`
        batch
    chooseaugs : list, optional
        choose types of aug, by default [0,1,2,3]
         0-rotation, 1-shear, 2-hflip, 3-scale
    rand_angle:
        choose within range
    angle:
        fixed angle for transform

    Returns
    -------
    `torch.tensor`
        augmented image
    """
    #apply same aug as during input
    # no softmax applied
    #receive a batch of size BXCXHXW
    # generate a batch of size augXBXCXHXW
    #after passing through model, get output of augXBXCXHXW, softmax
    # group to BXCXHXW after averaging
    batch = batch.detach()
    # aug=4
    if angle is None:
        angle = np.random.randint(rand_angle)

    topil = transforms.ToPILImage()
    totens = transforms.ToTensor()
    # rota = np.random.randint(10)
    scale = round(random.uniform(0.8, 1.4),2)
    shear = list(np.round(np.random.uniform(0, 4, size=2), 2))
    # translate = np.random.randint(2,size=2)

    if save_dir is not None:
        no_cols = 1+len(chooseaugs)
        fig, ax = plt.subplots(2, no_cols)
        fig.set_size_inches(6*no_cols,20)
        [axeach.axis("off") for axeach in ax.flatten()]

    ax_plt = ax[:, 0] if save_dir is not None else None
    actual = varres(batch, model, 0, 0, axes=ax_plt)#actual output, same as rotation 0 
    augs = [actual]
    if 0 in chooseaugs:
        aug_ind = 0
        ax_plt = ax[:, 1+chooseaugs.index(aug_ind)] if save_dir is not None and aug_ind in chooseaugs else None
        rot = varres(batch, model, 0, param=angle, axes=ax_plt)
        augs.append(rot)
    if 1 in chooseaugs:
        aug_ind = 1
        ax_plt = ax[:, 1+chooseaugs.index(aug_ind)] if save_dir is not None and aug_ind in chooseaugs else None
        # ax_plt = [ax[0, 1+chooseaugs.index(aug_ind)],ax[1, 1+chooseaugs.index(1)]] if save_dir is not None and 1 in chooseaugs else None
        shearimg = varres(batch, model, aug_ind, param=shear, axes=ax_plt)
        augs.append(shearimg)
    if 2 in chooseaugs:
        aug_ind = 2
        ax_plt = ax[:, 1+chooseaugs.index(aug_ind)] if save_dir is not None and aug_ind in chooseaugs else None
        flip = varres(batch, model, aug_ind, param=None, axes=ax_plt)
        augs.append(flip)
    if 3 in chooseaugs:
        aug_ind = 3
        ax_plt = ax[:, 1+chooseaugs.index(aug_ind)] if save_dir is not None and aug_ind in chooseaugs else None
        scaleaug = varres(batch, model, 3, param=scale, axes=ax_plt)
        augs.append(scaleaug)

    if save_dir is not None:
        ax[1, 0].remove()
        plt.savefig(os.path.join(save_dir, "check.png"), bbox_inches = 'tight')#, padding=0)
        plt.close("all")
    avg_out = torch.mean(torch.stack(augs,dim=0),dim=0)
    # torch.mean(torch.stack([batchout, img_aug1.view_as(batchout), img_aug2.view_as(batchout), 
    # img_aug3.view_as(batchout), img_aug4.view_as(batchout)], dim=0), dim=0)
    return avg_out


def tensor3dtopilwithoutscale(img):
    """convert tensor to pil without change in value

    Parameters
    ----------
    img : `torc.tensor`
        image
    """
    topil = transforms.ToPILImage()
    img_pillist = [topil(image) for image in img]
    torch.stack(img_pillist)

def save_inter(segm):#save interpolation error displays
    #0-rotation, 1-shear, 2-hflip, 3-scale
    """save intermediate results

    Parameters
    ----------
    segm : `torch.tensor`
        target
    """
    from loss.edge_loss import get_hot_enc
    trans = choose_func(0, 10)
    img1 = aug_img(get_hot_enc(segm), trans)
    trans = choose_func(0, -10)
    img1 = aug_img(img1, trans)
    print(dice_coefficient(img1, segm))
    fig, ax = plt.subplots(1, 2)
    [axs.axis("off") for axs in ax]
    index_disp = 1
    label_sel = 1
    pos = torch.where(segm[index_disp]==label_sel)
    xmax = pos[-1].max()
    xmin = pos[-1].min()
    ymax = pos[-2].max()
    ymin = pos[-2].min()
    halfwidth = (xmax-xmin)//2
    ymid = (ymin+ymax)//2
    ymin = ymid - halfwidth
    ymax = ymid + halfwidth
    ends = [xmin, ymin, xmax, ymax]
    m = 10
    imgpiece = img1[..., ymin-m: ymax+m, xmin-m: xmax+m]
    imgpiece = torch.argmax(imgpiece, dim=1)
    segm1 = segm[..., ymin-m: ymax+m, xmin-m: xmax+m]
    ax[1].imshow(imgpiece[index_disp])
    ax[0].imshow(segm1[index_disp])
    fig.savefig(r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/check.png")
