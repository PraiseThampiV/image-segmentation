import torch
from torch import nn

def dice_loss_sq(pred, act, no_sft=False, is_all_chann=False):
    """computes dice loss (from dice coefficient squared)

    Parameters
    ----------
    pred : tensor
        predictions
    act : tensor
        target
    no_sft : bool, optional
        whether softmax to be applied on `pred`, by default False
    is_all_chann:bool, by default False
        whether to include all channels (or skip bg)
    Returns
    -------
    tensor
        dice loss
    """
    dice_coeff = dice_coefficient_sq(pred, act, no_sft=no_sft, is_all_chann=is_all_chann)  
    return 1-dice_coeff

def dice_coefficient_sq(probs, act, ignore_onlybg=False, no_sft=False, is_all_chann=False):
    """expects a batch of volumes as inputs, not single volume

    Parameters
    ----------
    probs : tensor
        predictions
    act : tensor
        actual segmentations
    ignore_onlybg: bool
        while computing average, ignore images having only bg
    no_sft:bool, by default False
        not to apply Softmax2d
    is_all_chann :  bool
        whether to include all channels while computing or to ignore first channel(bg)

    Returns
    -------
    tensor
        dice coefficient (squared version)
    """
    smooth=1
    if len(act.shape)!=4:
        act = act.view(act.shape[0],1,act.shape[2], act.shape[2])
        sgm_zer=(torch.zeros(act.shape[0],probs.shape[1],*act.shape[2:]))
        if act.is_cuda:
            sgm_zer = sgm_zer.to(act.get_device())
        act_hot = sgm_zer.scatter(1, act, 1)
    else:
        act_hot = act
    if not no_sft:
        sft_mx = nn.Softmax2d()
        probs = sft_mx(probs)
    num=probs*act_hot#b,c,h,w--p*g
    num = num.view(num.shape[0], num.shape[1], -1)
    num=torch.sum(num,dim=2)#b,    

    den1=probs*probs#--p^2
    den1=den1.view(den1.shape[0], den1.shape[1], -1)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=act_hot*act_hot#--g^2
    den2=den2.view(den2.shape[0], den2.shape[1], -1)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=(2*num+smooth)/(den1+den2+smooth)
    #we ignore bg dice val, and take the fg
    if is_all_chann or dice.shape[1]==1:
        dice_eso = dice
        num_fg_chann = torch.numel(dice[0,:])
    else:
        dice_eso=dice[:,1:]
        num_fg_chann = torch.numel(dice[0,1:])
    div_by = torch.numel(dice_eso)
    # consider only images having both foreground and bg labels
    imgfgbg_ind = torch.unique(torch.nonzero(act, as_tuple=True)[0])#num of items in batch with fg
    if len(imgfgbg_ind) and dice.shape[1]>1:
        cnt_fgbg = len(imgfgbg_ind)
        div_by = cnt_fgbg*num_fg_chann
    dice_coeff = torch.sum(dice_eso[imgfgbg_ind])/div_by
    return dice_coeff

def dice_coefficient(probs, act, is_list_bat=False, nosft=False, channelcnt=None, is_all_chann=False):
    """
    use only for testing or validation, argmax used, hence not differentiable
    expects inputs as batches eg: probs-torch.Size([6, 3, 128, 128]), act-torch.Size([6, 128, 128])
        taking max arg, not probabilities
    Parameters
    ----------
    probs : `torch.tensor`
        predictions
    act : `torch.tensor`
        targets
    is_list_bat:bool
        return loss as list
    channel_cnt:int
        channel count
    no_sft:bool, by default False
        not to apply Softmax2d
    is_all_chann :  bool
        whether to include all channels while computing or to ignore first channel(bg)

    Returns
    -------
    `torch.tensor`
        dice coefficient
    """
    smooth=1
    if len(act.shape)!=4:
        act = act.view(act.shape[0],1,act.shape[2], act.shape[2])
        # sgm_zer=(torch.zeros(act.shape[0],probs.shape[1],*act.shape[2:]))

        if not channelcnt:
            channelcnt = probs.shape[-3]#probs channel count
        sgm_zer=(torch.zeros(act.shape[0],channelcnt,*act.shape[2:]))

        if act.is_cuda:
            sgm_zer = sgm_zer.to(act.get_device())
        act_hot = sgm_zer.scatter(1, act, 1)
    else:
        act_hot = act
    #if probs is a computed one channel output
    if not nosft:
        sft_mx = nn.Softmax2d()
        probs = sft_mx(probs)
    if len(probs.shape)!=3:
        probs = torch.argmax(probs, dim=1)
        # prob_zer=(torch.zeros(probs.shape[0],pc_count,*probs.shape[-2:]))
        # if probs.is_cuda:
        #     prob_zer = prob_zer.to(probs.get_device())
    probs = probs.unsqueeze(dim=1)
    # probs_hot = prob_zer.scatter(1, probs, 1)
    # probs = probs_hot
    # if probs.shape[-3]==1 and len(probs.shape)==4 or len(probs.shape)==3:
    #     if not channelcnt:
    #         channelcnt = act.max()+1
    prob_zer=torch.zeros(probs.shape[0],channelcnt,*probs.shape[-2:])
    if probs.is_cuda:
        prob_zer = prob_zer.to(probs.device)
    probs_hot = prob_zer.scatter(1, probs.long(), 1)
    probs = probs_hot

    num=probs*act_hot#b,c,h,w--p*g
    num = num.view(num.shape[0], num.shape[1], -1)
    num=torch.sum(num,dim=2)#b,    

    den1=probs#--p^2
    den1=den1.view(den1.shape[0], den1.shape[1], -1)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=act_hot#--g^2
    den2=den2.view(den2.shape[0], den2.shape[1], -1)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=(2*num+smooth)/(den1+den2+smooth)

    if is_all_chann or dice.shape[1]==1:
        dice_eso = dice
        num_fg_chann = torch.numel(dice[0,:])
    else:
        dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg
        num_fg_chann = torch.numel(dice[0,1:])

    if is_list_bat:
        list_dice_scores = dice_eso
        return list_dice_scores

    div_by = torch.numel(dice_eso)
    # consider only images having both foreground and bg labels
    imgfgbg_ind = torch.unique(torch.nonzero(act, as_tuple=True)[0])
    if len(imgfgbg_ind):
        cnt_fgbg = len(imgfgbg_ind)
        div_by = cnt_fgbg*num_fg_chann
    dice_coeff = torch.sum(dice_eso[imgfgbg_ind])/div_by
    # dice_coeff = torch.sum(dice_eso)/torch.numel(dice_eso)
    return dice_coeff

def dice_coefficientold(probs, act, is_list_bat=False, nosft=False, channelcnt=None):
    """considering the probabilities for calc
    expects inputs as batches eg: probs-torch.Size([6, 3, 128, 128]), act-torch.Size([6, 128, 128])

    Parameters
    ----------
    probs : `torch.tensor`
        predictions
    act : `torch.tensor`
        targets
    is_list_bat:bool
        return loss as list
    channel_cnt:int
        channel count
    no_sft:bool, by default False
        not to apply Softmax2d
    Returns
    -------
    `torch.tensor`
        dice coefficient
    """
    smooth=1
    act = act.view(act.shape[0],1,act.shape[2], act.shape[2])
    if not channelcnt:
        sgm_zer=(torch.zeros(act.shape[0],probs.shape[1],*act.shape[2:]))
    else:
        sgm_zer=(torch.zeros(act.shape[0],channelcnt,*act.shape[2:]))
    if act.is_cuda:
        sgm_zer = sgm_zer.to(act.get_device())
    act_hot = sgm_zer.scatter(1, act, 1)
    #if probs is a computed one channel output
    if probs.shape[-3]==1 and len(probs.shape)==4 or len(probs.shape)==3:
        if not channelcnt:
            channelcnt = act.max()+1
        prob_zer=torch.zeros(probs.shape[0],channelcnt,*probs.shape[2:])
        if probs.is_cuda:
            prob_zer = prob_zer.to(probs.device)
        probs_hot = prob_zer.scatter(1, probs.long(), 1)
        probs = probs_hot
    elif not nosft:
        sft_mx = nn.Softmax2d()
        probs = sft_mx(probs)
    num=probs*act_hot#b,c,h,w--p*g
    num = num.view(num.shape[0], num.shape[1], -1)
    num=torch.sum(num,dim=2)#b,    

    den1=probs#--p^2
    den1=den1.view(den1.shape[0], den1.shape[1], -1)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=act_hot#--g^2
    den2=den2.view(den2.shape[0], den2.shape[1], -1)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=(2*num+smooth)/(den1+den2+smooth)
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg
    if is_list_bat:
        list_dice_scores = dice_eso
        return list_dice_scores

    div_by = torch.numel(dice_eso)
    # consider only images having both foreground and bg labels
    imgfgbg_ind = torch.unique(torch.nonzero(act, as_tuple=True)[0])
    if len(imgfgbg_ind):
        cnt_fgbg = len(imgfgbg_ind)
        div_by = cnt_fgbg*torch.numel(dice[0,1:])
    dice_coeff = torch.sum(dice_eso[imgfgbg_ind])/div_by
    # dice_coeff = torch.sum(dice_eso)/torch.numel(dice_eso)
    return dice_coeff

# def dice_coefficient_test(pred, act):
#     act = act.view(act.shape[0],1,act.shape[2], act.shape[2])
#     sgm_zer=(torch.zeros(act.shape[0],3,*act.shape[2:])).to(device)
#     act_hot = sgm_zer.scatter(1, act, 1)
#     act = act_hot

#     smooth = 1
#     act = act[:,1:,:,:]
#     iflat = act.contiguous().view(-1)
#     sft_mx = nn.Softmax2d()
#     pred = sft_mx(pred)
#     pred = pred[:,1:,:,:]
#     tflat = pred.contiguous().view(-1)
#     intersection = (iflat * tflat).sum()
    
#     return ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

def dice_coefficient_sa(probs, act, ignore_onlybg=False, no_sft=False):#selfadjusting
    """expects a batch of volumes as inputs, not single volume

    Parameters
    ----------
    probs : tensor
        predictions
    act : tensor
        actual segmentations
    ignore_onlybg: bool
        while computing average, ignore images having only bg
    no_sft:bool, by default False
        not to apply Softmax2d
    Returns
    -------
    tensor
        dice coefficient (squared version)
    """
    smooth=1
    if len(act.shape)!=4:
        act = act.view(act.shape[0],1,act.shape[2], act.shape[2])
        sgm_zer=(torch.zeros(act.shape[0],probs.shape[1],*act.shape[2:]))
        if act.is_cuda:
            sgm_zer = sgm_zer.to(act.get_device())
        act_hot = sgm_zer.scatter(1, act, 1)
    else:
        act_hot = act
    if not no_sft:
        sft_mx = nn.Softmax2d()
        probs = sft_mx(probs)

    weight = 1 - probs
    num=probs*act_hot*weight#b,c,h,w--p*g
    num = num.view(num.shape[0], num.shape[1], -1)
    num=torch.sum(num,dim=2)#b,    

    den1=weight*probs#--p^2
    den1=den1.view(den1.shape[0], den1.shape[1], -1)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=act_hot.clone()#--g^2
    den2=den2.view(den2.shape[0], den2.shape[1], -1)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=(2*num+smooth)/(den1+den2+smooth)
    #we ignore bg dice val, and take the fg
    dice_eso=dice[:,1:] if dice.shape[1]>1 else dice
    div_by = torch.numel(dice_eso)
    # consider only images having both foreground and bg labels
    imgfgbg_ind = torch.unique(torch.nonzero(act, as_tuple=True)[0])
    if len(imgfgbg_ind) and dice.shape[1]>1:
        cnt_fgbg = len(imgfgbg_ind)
        div_by = cnt_fgbg*torch.numel(dice[0,1:])
    dice_coeff = torch.sum(dice_eso[imgfgbg_ind])/div_by
    return dice_coeff

def dice_loss_sa(pred, act, no_sft=False):#self adjusting
    """computes dice loss (from dice coefficient squared)

    Parameters
    ----------
    pred : `torch.tensor`
        predictions
    act : `torch.tensor`
        target
    no_sft : bool, optional
        whether softmax to be applied on `pred`, by default False

    Returns
    -------
    `torch.tensor`
        dice loss
    """
    dice_coeff = dice_coefficient_sa(pred, act, no_sft=no_sft)  
    return 1-dice_coeff

def logcoshdiceloss(pred, act, no_sft=False):
    diceloss = dice_loss_sq(pred, act, no_sft=no_sft)
    return torch.log(torch.cosh(diceloss))