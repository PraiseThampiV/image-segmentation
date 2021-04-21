import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def pat_extract_batch(img, dist_pat=None, pat_radius=None, save_pat_loc=None, is_save=False, num_pat_along_1dim=5, 
coincide_len=5, patimg_join=False):
    """expects square input

    Parameters
    ----------
    img : `torch.tensor`
        image
    dist_pat : int
        distance between patch centres
    pat_radius : int
        half patch width
    save_pat_loc : str, optional
        save location, by default None
    is_save : bool, optional
        whether to save as figure, by default False
    num_pat_along_1dim : int, optional
        number of patches along one dim, by default 4
    patimg_join : bool, optional
        whether patches from an image to be packed adjacent.

    Returns
    -------
    list
        patches set
    """
    # expecting squared inputs, outputs only square patches
    if dist_pat is not None and pat_radius is not None:
        num_pat_along_1dim = int(img.shape[-1]/(dist_pat+1))
        front_pad = pat_radius - dist_pat
        back_pad = pat_radius-img.shape[-1]%(dist_pat+1)
        padding_len = front_pad +  back_pad
    else:
        wid = img.shape[-1]
        # assume initially
        pat_wid_ini = int(wid/num_pat_along_1dim)
        # we need odd number for pat_wid
        pat_wid_ini -= (pat_wid_ini-1)%2
        dist_pat = (pat_wid_ini-1)
        extra_pix = wid - pat_wid_ini*num_pat_along_1dim
        pat_radius = int(dist_pat/2 + math.ceil(extra_pix/2))
        padding_len = extra_pix%2
        if pat_radius - dist_pat/2 < coincide_len:
            padding_len = 2*coincide_len - extra_pix

    # if front+back pad add to odd number, then pad with half number of pixels to left and half+1 to right
    pad_len_l = int(padding_len/2)
    pad_len_r = pad_len_l + padding_len%2
    # pad last dim by first tuple of 2 numbers and 2nd to last by (next tuple of two numbers)
    pad_img = F.pad(img, (pad_len_l, pad_len_r, pad_len_l, pad_len_r))
    # img_wid/dist_pat -1 anchors will exist,

    # arange gives values from begin to end-1
    # first anchor at pat_radius-th ind which is 0 to (pat_radius-1) + 1
    anchors=pat_radius + np.arange(0, num_pat_along_1dim)*(dist_pat+1)
    #all patches expected to have equal radius to left and right, 

    pat_set = []
    if is_save:
        fig_p, ax_p = plt.subplots(len(anchors),len(anchors))
    # fig_s, ax_s = plt.subplots(len(anchors),len(anchors))
    for row_ind in range(len(anchors)):
        for col_ind in range(len(anchors)):
            up_end=anchors[row_ind]-pat_radius
            bot_end = anchors[row_ind]+pat_radius
            left_end=anchors[col_ind]-pat_radius
            right_end = anchors[col_ind]+pat_radius
            # size of patch [2*pat_radius+1 X 2*pat_radius+1 X ]
            pat=pad_img[...,up_end:bot_end+1,left_end:right_end+1].clone()
            pat_set.append(pat)
            if is_save:
                pat_pic = pat[...,:,:]
                if pat_pic.shape[0]==3 and len(pat_pic.shape)==3:
                    pat_pic = pat[0,:,:]
                pat_dis=ax_p[row_ind][col_ind].imshow(pat_pic.cpu().detach().numpy())
                ax_p[row_ind][col_ind].axis("off")
                pat_dis.set_clim(0, pad_img.max())
    if is_save:
        fig_p.savefig(save_pat_loc)
        plt.close("all")
    pats=torch.stack(pat_set)
    if patimg_join:
        pats = torch.transpose(pats, 0, 1)
    pats_set = torch.flatten(pats, 0, 1)
    return pats_set

def pat_extr(img, dist_pat, pat_radius, save_pat_loc=None, is_save=False):
    """extracts patches

    Parameters
    ----------
    img : `torch.tensor`
        image
    dist_pat : int
        distance between patch centres
    pat_radius : int
        half patch width
    save_pat_loc : str, optional
        location to save, by default None
    is_save : bool, optional
        whether to save, by default False

    Returns
    -------
    list
        patches set
    """
    # expecting squared inputs, outputs only square patches
    num_pat_along_1dim = int(img.shape[-1]/(dist_pat+1))
    # padding to get equal sized patches from regions near to image edges
    front_pad = pat_radius - dist_pat
    back_pad = pat_radius-img.shape[-1]%(dist_pat+1)
    # if front+back pad add to odd number, then pad with half number of pixels to left and half+1 to right
    pad_len_l = int((front_pad+back_pad)/2)
    pad_len_r = pad_len_l + (front_pad+back_pad)%2
    # pad last dim by first tuple of 2 numbers and 2nd to last by (next tuple of two numbers)
    pad_img = F.pad(img, (pad_len_l, pad_len_r, pad_len_l, pad_len_r))
    # img_wid/dist_pat -1 anchors will exist,

    # arange gives values from begin to end-1
    # first anchor at pat_radius-th ind which is 0 to (pat_radius-1) + 1
    anchors=pat_radius + np.arange(0, num_pat_along_1dim)*(dist_pat+1)
    #all patches expected to have equal radius to left and right, 

    pat_set = []
    if is_save:
        fig_p, ax_p = plt.subplots(len(anchors),len(anchors))
    # fig_s, ax_s = plt.subplots(len(anchors),len(anchors))
    for row_ind in range(len(anchors)):
        for col_ind in range(len(anchors)):
            up_end=anchors[row_ind]-pat_radius
            bot_end = anchors[row_ind]+pat_radius
            left_end=anchors[col_ind]-pat_radius
            right_end = anchors[col_ind]+pat_radius
            # size of patch [2*pat_radius+1 X 2*pat_radius+1 X ]
            pat=pad_img[...,up_end:bot_end+1,left_end:right_end+1].clone()
            pat_set.append(pat)
            if is_save:
                if len(pat.shape)==3:
                    disp_img = pat[0] 
                pat_dis=ax_p[row_ind][col_ind].imshow(disp_img)#pat[...,:,:])
                ax_p[row_ind][col_ind].axis("off")
                pat_dis.set_clim(0, pad_img.max())
    if is_save:
        fig_p.savefig(save_pat_loc)
        plt.close("all")
    return pat_set

# aggregation
# inputs, patch_set, dist_pat, pat_radius, img_dim
def aggr_pat(pat_set, dist_pat, pat_radius, img_dim, save_img_loc):
    """aggregate patches

    Parameters
    ----------
    pat_set : list
        list of patches
    dist_pat : int
        distance between patch centres
    pat_radius : int
        half patch width
    img_dim : list
        image dimension
    save_img_loc : str
        location to save

    Returns
    -------
    `torch.tensor`
        aggregated image

    Raises
    ------
    Exception
        shape mismatch
    """
    #pat_wid > dis_pat to ensure overlapping patches
    # pat_radius : pixels to left,right,up and bottom to get square patch
    img_wid = img_dim[-1]
    # expecting squared patches, outputs only squared image output
    # img_wid/dist_pat -1 anchors will exist,
    num_pat_along_1dim = int(img_wid/(dist_pat+1))
    anchors=pat_radius + np.arange(0, num_pat_along_1dim)*(dist_pat+1)
    # get outplut image dimension when padded
    front_pad = pat_radius - dist_pat
    back_pad = pat_radius-img_dim[-1]%(dist_pat+1)
    out_img_wid = img_wid+ front_pad+back_pad
    out_img_dim = list(img_dim)
    out_img_dim[-2:]=[out_img_wid]*2

    # if front+back pad add to odd number, then pad with half number of pixels to left and half+1 to right
    pad_len_l = int((front_pad+back_pad)/2)
    pad_len_r = pad_len_l + (front_pad+back_pad)%2
    #all patches expected to have equal radius to left and right, 
    #first and last anchor's coord  will be changed to adjust to imgends
    out_img = torch.zeros(())
    out_img = out_img.new_full(out_img_dim, -float("Inf"))
    if pat_set[0].is_cuda:
        out_img = out_img.to(pat[0].get_device())
    fig_i, ax_i = plt.subplots()
    pat_set_ind = 0

    if len(img_dim)==2:
        out_img = out_img.unsqueeze(dim=0)
    for row_ind in range(len(anchors)):
        for col_ind in range(len(anchors)):
            up_end=anchors[row_ind]-pat_radius
            bot_end = anchors[row_ind]+pat_radius
            left_end=anchors[col_ind]-pat_radius
            right_end = anchors[col_ind]+pat_radius
            # size of patch [2*pat_radius+1 X 2*pat_radius+1 X ]
            pat_img=out_img[:,up_end:bot_end+1,left_end:right_end+1]
            out_img[:,up_end:bot_end+1,left_end:right_end+1]=torch.where(pat_set[pat_set_ind]>pat_img, pat_set[pat_set_ind].type(out_img.type()), pat_img)
            pat_set_ind+=1
    # for img end =out_img_wid-1-pad_len_r and +1 for slicing 
    out_img = out_img[:, pad_len_l:out_img_wid-pad_len_r, pad_len_l:out_img_wid-pad_len_r]
    ax_i.axis("off")
    ax_i.imshow(out_img[0])
    fig_i.savefig(save_img_loc)
    plt.close("all")
    if out_img.shape!=img_dim:
        out_img = out_img.squeeze()
    if out_img.shape!=img_dim:
       raise Exception(f"out image dimensions ({out_img.shape}) do not match expected ({img_dim})")
    return out_img

def aggr_pat_n_to_1(pat_set, dist_pat, pat_radius, img_dim, save_img_loc=None):
    """each input patch has n channels but with only one in corresponding pixels in all channels has non-zero value.
    pat_wid > dis_pat to ensure overlapping patches
    pat_radius : pixels to left,right,up and bottom to get square patch

    Parameters
    ----------
    pat_set : list
        patches list
    dist_pat : int
        distance between patch centres
    pat_radius : int
        half patch width
    img_dim : list
        shape
    save_img_loc : str, optional
        save location, by default None

    Returns
    -------
    `torch.tensor`
        aggregated image

    Raises
    ------
    Exception
        shape mismatch
    """
    img_wid = img_dim[-1]
    num_in_chann = pat_set[0].shape[-3]
    # expecting squared patches, outputs only squared image output
    # img_wid/dist_pat -1 anchors will exist,
    num_pat_along_1dim = int(img_wid/(dist_pat+1))
    anchors=pat_radius + np.arange(0, num_pat_along_1dim)*(dist_pat+1)
    # get outplut image dimension when padded
    front_pad = pat_radius - dist_pat
    back_pad = pat_radius-img_dim[-1]%(dist_pat+1)
    out_img_wid = img_wid+ front_pad+back_pad
    # out_img_dim = list(img_dim)
    out_img_dim=[out_img_wid]*2

    # if front+back pad add to odd number, then pad with half number of pixels to left and half+1 to right
    pad_len_l = int((front_pad+back_pad)/2)
    pad_len_r = pad_len_l + (front_pad+back_pad)%2
    #all patches expected to have equal radius to left and right, 
    #first and last anchor's coord  will be changed to adjust to imgends
    out_img_val_per_chann = []
    out_img_count_chann = []
    for chann in range(num_in_chann):
        out_img = torch.zeros(())
        out_img = out_img.new_full(out_img_dim, -float("Inf"))
        if pat_set[0].is_cuda:
            out_img = out_img.to(pat[0].get_device())
        out_img_val_per_chann.append(out_img)
    for chann in range(num_in_chann):
        out_img = torch.zeros(())
        out_img = out_img.new_full(out_img_dim, 0)
        if pat_set[0].is_cuda:
            out_img = out_img.to(pat[0].get_device())
        out_img_count_chann.append(out_img)
    pat_set_ind = 0

    # if len(img_dim)==2:
    #     out_img = out_img.unsqueeze(dim=0)
    for row_ind in range(len(anchors)):
        for col_ind in range(len(anchors)):
            up_end=anchors[row_ind]-pat_radius
            bot_end = anchors[row_ind]+pat_radius
            left_end=anchors[col_ind]-pat_radius
            right_end = anchors[col_ind]+pat_radius
            # size of patch [2*pat_radius+1 X 2*pat_radius+1 X ]
            for channel in range(num_in_chann):
                # assigning the highest value
                pat_from_set = pat_set[pat_set_ind][channel,:,:]
                pat_from_set =pat_from_set.type(out_img_val_per_chann[channel].type())
                pat_from_out=out_img_val_per_chann[channel][up_end:bot_end+1,left_end:right_end+1]
                out_img_val_per_chann[channel][up_end:bot_end+1,left_end:right_end+1]=torch.max(pat_from_set, pat_from_out)
                # counting the channel occurrences
                pat_from_out_count=out_img_count_chann[channel][up_end:bot_end+1,left_end:right_end+1]
                out_img_count_chann[channel][up_end:bot_end+1,left_end:right_end+1]=torch.where(pat_from_set!=0, 
                pat_from_out_count+1, pat_from_out_count)
            pat_set_ind+=1
    out_img_full = torch.stack([o_img for o_img in out_img_count_chann])
    max_count, max_arg = torch.max(out_img_full, dim=0)

    # finding indices where max_count exists
    max_count_exp = max_count.clone().expand_as(out_img_full)
    #initialize with a less value, putting infinity led to strange behaviour
    ind_max_count=torch.full(max_count_exp.shape, -1e9)# -float("Inf"))
    ind_max_count[max_count_exp==out_img_full]=1

    # finding max_channel values in channels with max_counts
    max_chann_val_full = torch.stack([o_img for o_img in out_img_val_per_chann])
    max_chann_multi = max_chann_val_full*ind_max_count
    max_arg2 = torch.argmax(max_chann_multi, dim=0)

    max_arg[torch.where(max_count_exp==out_img_full)[1:]]=max_arg2[torch.where(max_count_exp==out_img_full)[1:]]
    # for img end =out_img_wid-1-pad_len_r and +1 for slicing 
    out_img = max_arg[..., pad_len_l:out_img_wid-pad_len_r, pad_len_l:out_img_wid-pad_len_r]
    if save_img_loc is not None:
        fig_i, ax_i = plt.subplots()
        ax_i.axis("off")
        aggr_dis = ax_i.imshow(out_img[...,:,:])
        fig_i.colorbar(aggr_dis)
        fig_i.savefig(save_img_loc)
        plt.close("all")
    if img_dim[0]==1:
        out_img = out_img.unsqueeze(dim=0)
    if out_img.shape!=img_dim:
       raise Exception(f"out image dimensions ({out_img.shape}) do not match expected ({img_dim})")
    return out_img

def aggr_pat_1_to_1_vote(pat_set, dist_pat, pat_radius, img_dim, save_img_loc):
    """  each input patch has 1 channel with labels, output of channel chosen by voting
    pat_wid > dis_pat to ensure overlapping patches
    pat_radius : pixels to left,right,up and bottom to get square patch

    Parameters
    ----------
    pat_set : list
        list of patches
    dist_pat : int
        distance between patch centres
    pat_radius : int
        half patch width
    img_dim : list
        image shape
    save_img_loc : str
        save location

    Returns
    -------
    `torch.tensor`
        aggregated image

    Raises
    ------
    Exception
        shape mismatch
    """
    img_wid = img_dim[-1]
    num_in_chann = torch.stack([each_p for each_p in pat_set]).max()+1
    # expecting squared patches, outputs only squared image output
    # img_wid/dist_pat -1 anchors will exist,
    num_pat_along_1dim = int(img_wid/(dist_pat+1))
    anchors=pat_radius + np.arange(0, num_pat_along_1dim)*(dist_pat+1)
    # get outplut image dimension when padded
    front_pad = pat_radius - dist_pat
    back_pad = pat_radius-img_dim[-1]%(dist_pat+1)
    out_img_wid = img_wid+ front_pad+back_pad
    # out_img_dim = list(img_dim)
    out_img_dim=[out_img_wid]*2

    # if front+back pad add to odd number, then pad with half number of pixels to left and half+1 to right
    pad_len_l = int((front_pad+back_pad)/2)
    pad_len_r = pad_len_l + (front_pad+back_pad)%2
    #all patches expected to have equal radius to left and right, 
    #first and last anchor's coord  will be changed to adjust to imgends
    out_img_count_chann = []
    for chann in range(num_in_chann):
        out_img = torch.zeros(())
        out_img = out_img.new_full(out_img_dim, 0)
        if pat_set[0].is_cuda:
            out_img = out_img.to(pat[0].get_device())
        out_img_count_chann.append(out_img)

    fig_i, ax_i = plt.subplots()
    pat_set_ind = 0

    # if len(img_dim)==2:
    #     out_img = out_img.unsqueeze(dim=0)
    for row_ind in range(len(anchors)):
        for col_ind in range(len(anchors)):
            up_end=anchors[row_ind]-pat_radius
            bot_end = anchors[row_ind]+pat_radius
            left_end=anchors[col_ind]-pat_radius
            right_end = anchors[col_ind]+pat_radius
            # size of patch [2*pat_radius+1 X 2*pat_radius+1 X ]
            for channel in range(num_in_chann):
                pat_from_set = pat_set[pat_set_ind][:,:]
                # counting the channel occurrences
                pat_from_out_count=out_img_count_chann[channel][up_end:bot_end+1,left_end:right_end+1]
                out_img_count_chann[channel][up_end:bot_end+1,left_end:right_end+1]=torch.where(pat_from_set==channel, 
                pat_from_out_count+1, pat_from_out_count)
            pat_set_ind+=1
    out_img_full = torch.stack([o_img for o_img in out_img_count_chann])
    max_count, max_arg = torch.max(out_img_full, dim=0)

    # for img end =out_img_wid-1-pad_len_r and +1 for slicing 
    out_img = max_arg[..., pad_len_l:out_img_wid-pad_len_r, pad_len_l:out_img_wid-pad_len_r]
    ax_i.axis("off")
    ax_i.imshow(out_img[...,:,:])
    fig_i.savefig(save_img_loc)
    plt.close("all")
    if img_dim[0]==1:
        out_img = out_img.unsqueeze(dim=0)
    if out_img.shape!=img_dim:
       raise Exception(f"out image dimensions ({out_img.shape}) do not match expected ({img_dim})")
    return out_img