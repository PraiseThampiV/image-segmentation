import torch
import os, shutil
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
from torch import nn
from tqdm import tqdm, notebook
from torch.autograd import Variable
from torch import optim
from models.unetvgg import UNetVgg, UNetVggTwo
from models.resnt import DetectMid
from loss.dice import dice_coefficient
from loss.edge_loss import get_hot_enc
from loss.hausdorff import hdistance, get_edg_conv
from gen_utils.model_utils import get_saved_model
from detect_region.utils import reduce_dict

class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, is_best, save_model_dir, del_pre_model=True, use_saved=False):
    """    https://github.com/pytorch/examples/blob/master/imagenet/main.py to save model

    Parameters
    ----------
    model : `torch.module`
        network model
    is_best : bool
        if best model among epochs
    save_model_dir : str
        directory path
    del_pre_model : bool, optional
        delete previously saved file, by default True
    """
    filename=os.path.join(save_model_dir, 'checkpoint_epoch'+str(model["epoch"])+'.pth.tar')
    if is_best:
        prev_model_state = [name for name in os.listdir(save_model_dir) if "epoch" in name]
        if del_pre_model and len(prev_model_state):
            os.unlink(os.path.join(save_model_dir, prev_model_state[0]))
        torch.save(model, filename)

def use_prev(model, save_model_dir):
    """allow pretrained parameters to load

    Parameters
    ----------
    model : `torch.module`
        model
    save_model_dir : str
        path of saved model

    Returns
    -------
    `torch.model`
        saved model
    """
    prev_model_state = [name for name in os.listdir(save_model_dir) if "epoch" in name]
    saved_epoch = int(prev_model_state[0].split('epoch')[1][0])
    prev_model = get_saved_model(model, save_model_dir, with_edge=False)
    if next(model.parameters()).is_cuda:
        prev_model = prev_model.cuda()
    else:
        prev_model = prev_model.cpu()        
    return prev_model, saved_epoch

def to_np(x):
    """convert to numpy

    Parameters
    ----------
    x : array like
        array

    Returns
    -------
    numpy.array
        converted array
    """
    return x.data.cpu().numpy()

def get_imgslr(images, segm_act):
    """get image patches for left and right menisci from info on actual segmentation masks

    Parameters
    ----------
    images : torch.tensor
        image batch
    segm_act : torch.tensor
        mask batch

    Returns
    -------
    `torch.tensor`
        concatenated left and right halves of image
    """
    hmin, hmax = torch.nonzero(segm_act, as_tuple=True)[1].min(), torch.nonzero(segm_act, as_tuple=True)[1].max()
    wmin, wmax = torch.nonzero(segm_act, as_tuple=True)[2].min(), torch.nonzero(segm_act, as_tuple=True)[2].max()
    margin=15
    orig_hmax = segm_act.shape[-2]
    orig_wmax = segm_act.shape[-1]
    crop_hmin, crop_hmax = max(0, hmin-margin), min(hmax+margin, orig_hmax)
    crop_wmin, crop_wmax = max(0, wmin-margin), min(wmax+margin, orig_wmax)
    images = images[..., crop_hmin: crop_hmax, crop_wmin: crop_wmax].clone()
    # y_crop = y[:,:,max(0, hmin-margin): min(hmax+margin, orig_hmax), max(0, wmin-margin): min(wmax+margin, orig_wmax)].clone()

    width = images.shape[-1]
    height = images.shape[-2]
    xl = images[...,:int(width/2)].clone()
    xr = images[...,int(width/2):].clone()

    #upsampling in diff mode for segm and img
    if len(images.shape)==4:
        up_mode = 'bicubic'
        inter_xl = F.interpolate(xl, [256, 256], mode=up_mode)
        inter_xr = F.interpolate(xr, [256, 256], mode=up_mode)
    else:
        up_mode = 'nearest'
        upsample = torch.nn.Upsample([256, 256], mode=up_mode)
        # inter_xl = F.interpolate(xl, [256, 256])
        inter_xl = upsample(xl.unsqueeze(dim=1).float()).squeeze(dim=1).long()
        inter_xr = upsample(xr.unsqueeze(dim=1).float()).squeeze(dim=1).long()
    cat_img = torch.cat((inter_xl, inter_xr), dim=0)
    return cat_img

def collate_fn(batch):
    """used with dataloader

    Parameters
    ----------
    batch : list
        images, target

    Returns
    -------
    tuple
        tuple of zipped items
    """
    return tuple(zip(*batch))

def train(train_loader, model, criterion, epoch, num_epochs, device, batch_size, optimizer=None, lr=1e-4):
    """train model

    Parameters
    ----------
    train_loader : `torch.utils.data.DataLoader`
        data loader for train dataset
    model : `torch.module`
        model
    criterion : list
        list of loss functions
    epoch : int
        epoch number
    num_epochs : int
        number of epochs
    device : `torch.device`
        cpu or cuda
    batch_size : int
        batch size
    optimizer : `torch::optim::Optimizer`, optional
        optimizer, by default None
    lr : float, optional
        learning rate, by default 1e-4

    Returns
    -------
    float
        average loss
    """
    model.train()
    losses_avgr = AverageMeter()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    # set a progress bar
    pbar = tqdm(enumerate(train_loader), total=int(len(train_loader.dataset)/batch_size))
    # pbar = tqdm(iter(train_loader), total=len(train_loader.dataset))
    for i, (images, targets, idx) in pbar:
        # Convert torch tensor to Variable
        # images = Variable(images.to(device))#.cuda())
        # segm_act = Variable(segm_act.to(device))#cuda())
        optimizer.zero_grad()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.train()
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # measure loss
        # loss = criterion(outputs, segm_act)

        losses_avgr.update(loss_value, len(images))

        # compute gradient and do SGD step
        losses.backward()
        optimizer.step()

        # logging

        # # add the model graph
        # logger.add_graph(model, outputs)

        # # log loss values every iteration
        # logger.add_scalar('data/(train)loss_val', losses.val, i + 1)
        # logger.add_scalar('data/(train)loss_avg', losses.avg, i + 1)

        # # log the layers and layers gradient histogram and distributions
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.add_histogram('model/(train)' + tag, to_np(value), i + 1)
        #     logger.add_histogram('model/(train)' + tag + '/grad', to_np(value.grad), i + 1)

        # # log the outputs given by the model (The segmentation)
        # logger.add_image('model/(train)output', make_grid(outputs.data), i + 1)

        # update progress bar status
        pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(epoch avg) '
                             % (epoch + 1, num_epochs, losses_avgr.val, losses_avgr.avg))

    # return avg loss over the epoch
    return losses_avgr.avg

def evaluate(val_loader, model, criterion, val_acc_func, epoch, num_epochs, device):
    """validate model

    Parameters
    ----------
    val_loader : `torch.utils.data.DataLoader`
        data loader for validation dataset
    model : `torch.module`
        model
    criterion : list
        list of loss functions
    epoch : int
        epoch number
    num_epochs : int
        number of epochs
    val_acc_func : list
        list of accuracy functions
    device : `torch.device`
        cpu or cuda

    Returns
    -------
    list
        average loss and/or accuracy
    """
    # model.eval()
    losses_avgr = AverageMeter()
    val_accs = AverageMeter()
    hdorff = AverageMeter()

    # set a progress bar
    pbar = tqdm(enumerate(val_loader))#, total=len(val_loader))
    #pbar = tqdm(iter(val_loader), total=len(val_loader.dataset))
    for i, (images, targets, idx) in pbar:
        # Convert torch tensor to Variable
        # images = Variable(images.to(device))
        # segm_act = Variable(segm_act.to(device))

        with torch.no_grad():
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            losses_avgr.update(loss_value, len(images))#aggr_loss.item()

            # valid_val = val_acc_func(outputs, segm_act)
            # val_accs.update(valid_val) #function already averages based on batch count
            # if val_acc_func==dice_coefficient:
            #     loss_name = "DICE SCORE"
            # else:

            pbar.set_description('[VALID] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(epoch avg) '
                             % (epoch + 1, num_epochs, losses_avgr.val, losses_avgr.avg))
       
    # return avg loss over the epoch
    # if val_acc_func==dice_coefficient:
    #     print(f"hausdorff distance is {hdorff.avg.round(2)}")
    return losses_avgr.avg

def plt_loss(data, max_epo, title, save_file_path, valid_after):
    """plot graphs

    Parameters
    ----------
    data : list
        loss and accuracy values in epochs
    max_epo : int
        number of epochs
    title : str
        title
    save_file_path : str
        save file path
    valid_after : int
        validation check every
    """
    fig_plt, ax_plt = plt.subplots(1, 2)
    fig_plt.set_size_inches(19, 7)
    for ind, ax in enumerate(ax_plt):
        xmax=len(data[ind])
        ax.plot(range(xmax), data[ind], 'dodgerblue')#, label='training')
        # plt.plot(range(max_epo), validation_history['loss'], 'orange', label='validation')
        ax.axis(xmin=0,xmax=xmax)
        ax.set_xlabel(f'Epochs/{valid_after}' if ind!=0 else "Epochs")
        # plt.ylabel('Loss')
        ax.set_title(title[ind])
    plt.savefig(save_file_path, bbox_inches="tight")
    plt.close("all")
    # plt.legend();

def valid(model,x_valid,y_valid,criterion):
    """validation

    Parameters
    ----------
    model : `torch.module`
        model
    x_valid : `torch.tensor'
        image dataset
    y_valid : `torch.tensor`
        target dataset
    criterion : `torch.nn`
        loss function

    Returns
    -------
    float
        loss
    """
    with torch.no_grad():
        model.eval()
        y_pred = model(x_valid)
        loss = criterion(y_pred, y_valid)
        print('test-loss',t, loss.item(),end=' ')
        return loss.item()

def naive_train(train_set, valid_set, model, criterion, val_acc_func, num_epochs, batch_size, device, out_path, kwargs, experiment_name="", 
valid_after=4, lr=1e-4, use_saved=False):
    """wrapper for train, valid 

    Parameters
    ----------
    train_set : `torch.utils.data.dataset.Dataset`
        training set
    valid_set : `torch.utils.data.dataset.Dataset`
        validation set
    model : `torch.module`
        model
    criterion : list
        list of loss functions
    val_acc_func : `list`
        list of accuracy functions
    num_epochs : int
        number of epochs
    batch_size : int
        batch size
    device : `torch.device`
        cpu or cuda
    out_path : str
        path of output dir
    kwargs : list
            num of workers, wheter to pin_memory
    experiment_name : str, optional
        name of expt, by default ""
    valid_after : int, optional
        validation check every, by default 4
    lr : float, optional
        learning rate, by default 1e-4
    use_saved : bool, by default False
        whether to pretrain
    """
    best_loss = 10000
    metric_dict = {"train_loss":[], "val_loss":[], "val_acc":[]}
    train_loader = DataLoader(train_set, batch_size=batch_size, **kwargs, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, **kwargs, collate_fn=collate_fn)
    prev_epo = 0
    if use_saved:
        model, prev_epo = use_prev(model, out_path)
        num_epochs = num_epochs - prev_epo
    for epoch in range(0, num_epochs):
        # train for one epoch
        curr_loss = train(train_loader, model, criterion, epoch, num_epochs, device, batch_size, lr=lr)
        metric_dict["train_loss"].append(curr_loss)

        if not epoch%valid_after:
            val_loss = evaluate(val_loader, model, criterion, val_acc_func, epoch, num_epochs, device)
            metric_dict["val_loss"].append(val_loss)

            # store best loss and save a model checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1 + prev_epo,
                'arch': experiment_name,
                'state_dict': model.state_dict(),
                # 'best_prec1': best_loss,
                # 'optimizer': optimizer.state_dict(),
            }, is_best, out_path)

    # logger.close()
    val_times = len(metric_dict["val_loss"])
    graph_path = os.path.join(out_path, "graph")# for graph_type in metric_dict.keys()]
    # list(map(plt_loss, metric_dict.values(),[num_epochs, val_times, val_times], metric_dict.keys(), graph_paths))
    plt_loss(list(metric_dict.values()),num_epochs, list(metric_dict.keys()), graph_path, valid_after)
