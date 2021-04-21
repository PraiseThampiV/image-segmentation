import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from loss.dice import dice_loss_sq, dice_coefficient
from loss.fft_loss import fft_loss_logits
from loss.focal import focal_loss_with_logits
from models.unetvgg import UNetVgg256, UNetVgg, UNetVggWholeImg
from detect_region.naive_train import naive_train
from pl_utils.pl_mod import SegmModelPL
from pl_utils.pl_test import test_model
from pl_utils.pl_train import train_model
from result_utils.save_img import test_pred, save_pred_one, save_test_img_grid


def train_general(dataset, out_path, expt_name="", val_set_perc=0.15, test_set_perc=.05, device=None,
                  num_epochs=1, is_naive_train=False, mode=None, val_set=None, test_set=None, valid_after=4, 
                  net_model=None, batch_size=3, test_batch_size = 7, loss_criterion=None, val_acc_func=None, 
                  sgm_train=True, lr=1e-4, use_saved=False):
    """general wrapper function for train,valid, test

    Parameters
    ----------
    dataset : `torch.utils.data.dataset.Dataset`
        dataset
    out_path : str
        output path
    expt_name : str, optional
        experiment name, by default ""
    val_set_perc : float, optional
        validation set percentage, by default 0.15
    test_set_perc : float, optional
        test set percentage, by default .05
    device : `torch.device`, optional
        cpu or cuda, by default None
    num_epochs : int, optional
        number of epochs, by default 1
    is_naive_train : bool, optional
        if pytorch lightning train or general, by default False
    mode : str, optional
        if train set, validation set etc predefined, by default None
    val_set : `torch.utils.data.dataset.Dataset`, optional
        validation dataset, by default None
    test_set : `torch.utils.data.dataset.Dataset`, optional
        test dataset, by default None
    valid_after : int, optional
        validation check every, by default 4
    net_model : `torch.module`, optional
        netowrk model, by default None
    batch_size : int, optional
        batch size, by default 3
    test_batch_size : int, optional
        test batch size, by default 7
    loss_criterion : list, optional
        list of loss function, by default None
    val_acc_func : list, optional
        list of accuracy functions for validation, by default None
    sgm_train : bool, optional
        whether for segmentation mask training, by default True
    lr : float, optional
        learning rate, by default 1e-4

    Returns
    -------
    `torch.module`
        network model
    """
    if mode is None:
        full_data = dataset
        val_size = int(val_set_perc * len(full_data))
        test_size = int(test_set_perc * len(full_data))
        train_size = len(full_data) - test_size - val_size
        train_set, val_set, test_set = torch.utils.data.random_split(full_data, [train_size, val_size, test_size])
    else:
        train_set = dataset
    # #delete below
    # train_set, val_set, test_set = full_data, full_data, full_data
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    htc_gpu = torch.cuda.device_count() if device.type=='cuda' else 0
    kwargs = {'num_workers': 4*htc_gpu, 'pin_memory': True} if device == 'cuda' else {}

    if net_model is None:
        net_model = UNetVgg256()
    if loss_criterion is None:
        loss_criterion = [dice_loss_sq]#, fft_loss_logits]#focal_loss_with_logits]#, focal_loss_patch]#, extr_pat_loss]#, ]
    if val_acc_func is None:
        val_acc_func=dice_coefficient
    lightn_mod = SegmModelPL(net_model, train_set, val_set, test_set, batch_size, test_batch_size,
    loss_criterion=loss_criterion, val_acc_func=val_acc_func,experiment_name=expt_name)
    use_mod=lightn_mod.to(device)
    if is_naive_train:
        naive_train(train_set, val_set, net_model, loss_criterion, val_acc_func, num_epochs, batch_size, device, out_path, 
        kwargs, valid_after=valid_after, lr=lr, use_saved=use_saved)
        trainer=None
        use_mod = net_model
    else:
        trainer = train_model(lightn_mod, out_path, del_pre_files=True, device=device, num_epochs=num_epochs)

    if test_set_perc:
        test_model(lightn_mod, out_path, trainer)

        # saving results
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
        if sgm_train:
            img, segm, pred = test_pred(use_mod, out_path, test_loader, expt_name, trainer)
            bat_one = True if len(segm.shape) == 3 and segm.shape[0] == 1 else False
            save_pred_one(img, segm, pred, out_path, bat_one=bat_one,fig_name="out1"+expt_name)
            if img.shape[-1]!=pred.shape[-1]:
                img = F.interpolate(img, pred.shape[-1])
            save_test_img_grid(img, segm, pred, out_path, fig_name="outs"+expt_name)
        else:
            test_pred(use_mod, out_path, test_loader, expt_name, trainer, sgm_train=False,loss_func=val_acc_func)
    return use_mod


def train_whole_img(saved_model, output_path, dataset, num_epochs, valid_after=4, device=None):
    """helper function for patch aggregation

    Parameters
    ----------
    saved_model : `torch.module`
        network model
    output_path : str
        path of output directory
    dataset : `torch.utils.data.dataset.Dataset`
        dataset
    num_epochs : int
        number of epochs
    valid_after : int, optional
        validation check every, by default 4
    device : `torch.device`, optional
        cpu or cuda, by default None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for filename in os.listdir(saved_model_dir):
    #     file_path = os.path.join(saved_model_dir, filename)
    #     if os.path.isfile(file_path) and "epoch" in file_path:
    #         save_model_path = file_path
    #         break
    model = UNetVgg()#WholeImg(save_model_path)
    train_general(dataset, output_path, net_model= saved_model, expt_name="_full_img", test_set_perc=.15,
                  num_epochs=num_epochs, is_naive_train=True, valid_after=valid_after, device=device)