import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from loss.dice import dice_loss_sq, dice_coefficient
from loss.focal import focal_loss_with_logits
from models.unetvgg import UNetVgg256, UNetVgg, UNetVggWholeImg
from perlosstrain.naive_train import naive_train
from pl_utils.pl_mod import SegmModelPL
from pl_utils.pl_test import test_model
from pl_utils.pl_train import train_model
from perlosstrain.res_save import test_pred, save_pred_one, save_test_img_grid


def train_general(dataset, out_path, expt_name="", val_set_perc=0.15, test_set_perc=.05, device=None,
                  num_epochs=1, is_naive_train=False, mode=None, val_set=None, test_set=None, valid_after=4, net_model=None,
                  loss_criterion=None, val_acc_func=None, edge_mode=True, trained_edgemodel=None, batch_size=3,
                  test_batch_size=7):
    """split into test,train,test dataset and send for training, subsequently saves test outputs

    Parameters
    ----------
    dataset : torch dataset
        whole dataset
    out_path : path like
        output path
    expt_name : str, optional
        name of experiment, by default ""
    val_set_perc : float, optional
        valid set percentage, by default 0.15
    test_set_perc : float, optional
        test percentage, by default .05
    device : torch device, optional
        cuda or cpu, by default None
    num_epochs : int, optional
        number of epochs, by default 1
    is_naive_train : bool, optional
        if naive_train function to be used, by default False
    mode : str, optional
        if iterable dataset is input, by default None
    val_set : torch dataset, optional
        validation set if iterabledataset, by default None
    test_set : torch dataset, optional
        iterable dataset, by default None
    valid_after : int, optional
        number of epochs between validations, by default 4
    net_model : torch model, optional
        neural network model, by default None
    loss_criterion : list, optional
        list of loss functions, by default None
    val_acc_func : list, optional
        validation accuracy function, by default None
    edge_mode : bool, optional
        if edge mode, by default True
    trained_edgemodel : torch model, optional
        trained edge model, by default None
    batch_size : int, optional
        batch size, by default 3
    test_batch_size : int, optional
        test batch size, by default 7

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
        loss_criterion = [dice_loss_sq, focal_loss_with_logits]#, focal_loss_patch]#, extr_pat_loss]#, ]
    if val_acc_func is None:
        val_acc_func=dice_coefficient
    lightn_mod = SegmModelPL(net_model, train_set, val_set, test_set, batch_size, test_batch_size,
    loss_criterion=loss_criterion, val_acc_func=val_acc_func,experiment_name=expt_name)
    use_mod=lightn_mod.to(device)
    if is_naive_train:
        naive_train(train_set, val_set, net_model, loss_criterion, val_acc_func, num_epochs, batch_size, device, out_path, kwargs, 
        valid_after=valid_after, edge_mode=edge_mode, edge_model=trained_edgemodel, experiment_name=expt_name)
        trainer=None
        use_mod = net_model
    else:
        trainer = train_model(lightn_mod, out_path, del_pre_files=True, device=device, num_epochs=num_epochs)

    if test_set_perc:
        test_model(lightn_mod, out_path, trainer)

        # saving results
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
        img, segm, pred = test_pred(use_mod, out_path, test_loader, expt_name, trainer)
        bat_one = True if len(segm.shape) == 3 and segm.shape[0] == 1 else False
        save_pred_one(img, segm, pred, out_path, bat_one=bat_one,fig_name="out1"+expt_name, valid_func=val_acc_func)
        if img.shape[-1]!=pred.shape[-1]:
            img = F.interpolate(img, pred.shape[-1])
        save_test_img_grid(img, segm, pred, out_path, fig_name="outs"+expt_name)
    return use_mod


def train_whole_img(saved_model_dir, data_loc, dataset, num_epochs, valid_after=4, device=None):
    """helper function for patch aggregation

    Parameters
    ----------
    saved_model_dir : `torch.module`
        network model
    data_loc : str
        path of data
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
    for filename in os.listdir(saved_model_dir):
        file_path = os.path.join(saved_model_dir, filename)
        if os.path.isfile(file_path) and "epoch" in file_path:
            save_model_path = file_path
            break
    model = UNetVgg()#WholeImg(save_model_path)
    train_general(dataset, data_loc, saved_model_dir, net_model= model, expt_name="_full_img", test_set_perc=.15,
                  num_epochs=num_epochs, is_naive_train=True, valid_after=valid_after, device=device)