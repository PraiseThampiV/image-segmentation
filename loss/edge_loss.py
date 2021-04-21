import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from torch import nn
from loss.dice import dice_loss_sq, dice_coefficient
from loss.focal import focal_loss_with_logits

def show_enc_orig(enc, orig,save_path=None):
    #show encoded vector and original image
    """

    Parameters
    ----------
    enc : `torch.tensor`
        encoded vector
    orig : `torch.tensor`
        image
    save_path : str, optional
        save location, by default None
    """
    chann = enc.shape[0]
    fig, ax = plt.subplots(1, chann+1)
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0].imshow(orig.clone().detach().cpu())
    for ind, axis in enumerate(ax[1:]):
        axis.imshow(enc.clone().detach().cpu()[ind])
    if save_path:
        plt.savefig(save_path)
    plt.close("all")

def get_hot_enc(input, channels=3):
    """get one hot encoded vector

    Parameters
    ----------
    input : `torch.tensor`
        image
    channels : int, optional
        channel count, by default 3

    Returns
    -------
    `torch.tensor`
        oe hot encoded vector
    """
    if len(input.shape)==2:
        input = input.unsqueeze(dim=0).unsqueeze(dim=0)
    if len(input.shape)==3:
        input = input.view(input.shape[0], 1, input.shape[2], input.shape[2])
    input_zer = (torch.zeros(input.shape[0], channels, *input.shape[2:]))
    if input.is_cuda:
        input_zer = input_zer.to(input.get_device())
    input_hot = input_zer.scatter(1, input.long(), 1)
    return input_hot
    
def get_edge_img(act1):
    """get border image

    Parameters
    ----------
    act1 : `torch.tensor`
        target

    Returns
    -------
    `torch.tensor`
        border image
    """
    border_img = sobel(act1.cpu().numpy().astype(np.int16))
    non0_inds = np.nonzero(border_img)
    edge_img = torch.zeros_like(act1)
    edge_img[non0_inds[0], non0_inds[1]] = 1
    # edge_img[non0_inds[0], non0_inds[1]] = torch.tensor(1).type((edge_img.type()))
    return edge_img

def ppce_edgeloss(prob1, act1):
    """computes edges pixels from segm and calculate dice loss on this output and probs
    expecting outputs as output from EdgeNet, a tuple of 3 tensors

    Parameters
    ----------
    probs : `torch.tensor `
        predictions
    segm : `torch.tensor`
        target
    """
    if isinstance(prob1,tuple):
        prob1 = prob1[-1]
    act1_enc = torch.cat(list(map(get_hot_enc, act1))) # BX3XhXW shaped img
    act_origs = act1_enc.shape#act original shape
    act_flat = torch.flatten(act1_enc, 0,1)
    edge_img = torch.stack(list(map(get_edge_img, act_flat)))
    edge_img = edge_img.view(act_origs)
    # ce=torch.nn.CrossEntropyLoss()
    # loss = ce(prob1, edge_img)
    # edge_img = torch.argmax(edge_img, dim=1)
    # loss_focal = focal_loss_with_logits(prob1, edge_img)
    dice_l = dice_loss_sq(prob1, edge_img[:,:,...], is_all_chann=False, no_sft=False)# 2 channeled output for edge and edge_net is one hotencoded vector
    return dice_l

def pp_edgeacc(prob1, act1, is_list_bat=False, nosft=False):
    """computes edges pixels from segm and calculate CE on this output and probs

    Parameters
    ----------
    probs : `torch.tensor `
        predictions
    segm : `torch.tensor`
        target
    """
    if isinstance(prob1,tuple):
        prob1 = prob1[-1]
    # edge_img = torch.stack(list(map(get_edge_img, act1)))
    act1_enc = torch.cat(list(map(get_hot_enc, act1))) # BX3XhXW shaped img
    act_origs = act1_enc.shape#act original shape
    act_flat = torch.flatten(act1_enc, 0,1)
    edge_img = torch.stack(list(map(get_edge_img, act_flat)))
    edge_img = edge_img.view(act_origs)
    # edge_img = torch.argmax(edge_img, dim=1)
    edge_img = edge_img[:, :, ...]
    dice_scr = dice_coefficient(prob1, edge_img, is_list_bat, nosft=False, channelcnt=3, is_all_chann=False)
    return dice_scr #+ loss_focal

def get_act_bnd_lbs(act):
    #get actual boundary labels.not binary image
    act_enc = torch.cat(list(map(get_hot_enc, act))) # BX3XhXW shaped img
    act_origs = act_enc.shape#act original shape
    act_flat = torch.flatten(act_enc, 0,1)
    edge_img = torch.stack(list(map(get_edge_img, act_flat)))
    edge_img = edge_img.view(act_origs)
    edge_img[:,0,...] = ((edge_img[:,0,...]+1)%2)#invert values in bg channel
    return torch.argmax(edge_img, dim=1)

def bp_edgeloss(prob, act):
    """computes edges pixels from segm and calculate CE on this output and probs
    expecting outputs as output from EdgeNet, a tuple of 3 tensors

    Parameters
    ----------
    probs : [type]
        [description]
    segm : [type]
        [description]
    """
    sft = nn.Softmax2d()
    #  predicted boundary
    predb = prob[-9]
    # predb = sft(predb)
    #  predicted mask
    predm = prob[-1]
    # predm = sft(predm)

    act_hot=(torch.zeros(act.shape[0],3,*act.shape[1:]))#for one hot encoding, 3 channels and then reduce to 2 channels for loss comp
    act_hot = act_hot.to(act.device)
    act_m = act_hot.scatter(1, act.unsqueeze(dim=1), 1)

    act_enc = torch.cat(list(map(get_hot_enc, act))) # BX3XhXW shaped img
    act_origs = act_enc.shape#act original shape
    act_flat = torch.flatten(act_enc, 0,1)
    edge_img = torch.stack(list(map(get_edge_img, act_flat)))
    edge_img = edge_img.view(act_origs)
    # edge_img = torch.argmax(edge_img, dim=1)

    # edge_img = torch.stack(list(map(get_edge_img, act)))
    # edge_hot=(torch.zeros(edge_img.shape[0],edge_img.max()+1,*edge_img.shape[1:]))
    # edge_hot = edge_hot.to(edge_img.device)
    # act_b = edge_hot.scatter(1, edge_img.unsqueeze(dim=1), 1)

    # dl = dice_loss_sq#torch.nn.MSELoss()
    #negating bg channel
    # edge_img[:,0,...] = ((edge_img[:,0,...]+1)%2)
    edge_img = edge_img[:,:,...]
    lossb = dice_loss_sq(predb, edge_img, no_sft=False, is_all_chann=False)# + focal_loss_with_logits(predb, torch.argmax(edge_img, dim=1))
    #trying 2 channel ouput for mask, no meaning as we need softmax at final layer
    # lossm = dice_loss_sq(predm, act_m[:,1:,...], no_sft=True) #+ focal_loss_with_logits(predm, act)
    bce = nn.BCELoss(reduction='sum')
    # mse = nn.MSELoss()
    lossm = dice_loss_sq(predm, act_m[:,:,...])
    return lossb + lossm

def bp_edgeacc(prob, act, is_list_bat=False):
    """computes edges pixels from segm and calculate CE on this output and probs

    Parameters
    ----------
    probs : [type]
        [description]
    segm : [type]
        [description]
    """
    if isinstance(prob,tuple):
        prob = prob[-1]
    act_hot=(torch.zeros(act.shape[0],3,*act.shape[1:]))#for one hot encoding, 3 channels and then reduce to 2 channels for loss comp
    act_hot = act_hot.to(act.device)
    act_m = act_hot.scatter(1, act.unsqueeze(dim=1), 1)
    dice_scr = dice_coefficient(prob, act_m[:,:,...], is_list_bat, channelcnt=3, nosft=False)

    #TODO try toinclude boundary acc and display in save_img
    # predb = prob[-9]
    # # predb = sft(predb)
    # act_enc = torch.cat(list(map(get_hot_enc, act))) # BX3XhXW shaped img
    # act_origs = act_enc.shape#act original shape
    # act_flat = torch.flatten(act_enc, 0,1)
    # edge_img = torch.stack(list(map(get_edge_img, act_flat)))
    # edge_img = edge_img.view(act_origs)

    # edge_img = torch.stack(list(map(get_edge_img, act)))

    return dice_scr #+ loss_focal