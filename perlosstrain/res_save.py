import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision.utils import make_grid 
# from loss.edge_loss import pp_edgeacc, get_edge_img
from loss.dice import dice_coefficient

#TODO replace whole file code into a class
def naive_test(testloader, model, valid_func=dice_coefficient, edge_mode=True):
    """calculate test set accuracy

    Parameters
    ----------
    testloader : `torch.utils.data.DataLoader`
        testset loader
    model : `torch.nn.module`
        model
    valid_func : list, optional
        list of loss/accuracy functions, by default `dice_coefficient`
    edge_mode : bool, optional
        whether to train for edges, by default True
    """
    total = 0
    dice_total = 0
    if valid_func is None:
        valid_func = dice_coefficient
    with torch.no_grad():
        for data in testloader:
            images, segm = data
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs=outputs[0] 
            dice_total+=valid_func(outputs, segm) #already averaged by batc size in dice coeff func
            total += 1

    print(f'Dice coefficient of the network on test images:{dice_total / total:.4f}')

def get_saved_model(model, saved_model_dir, with_edge=True):
    saved_model = model#UNETvgg()
    for filename in os.listdir(saved_model_dir):
        file_path = os.path.join(saved_model_dir, filename)

        if os.path.isfile(file_path) and "epoch" in file_path:
            save_model_path = file_path
            break
    checkpoint = torch.load(save_model_path)    
    saved_model.load_state_dict(checkpoint['state_dict'])
    return saved_model

def test_pred(model, saved_model_dir, test_loader, experiment_name="", trainer=None, print_test_acc=True, valid_func=None,
edge_mode=True):
    """wrapper function, return x, pred, y from test batch 

    Parameters
    ----------
    model : `torch.module`
        model
    saved_model_dir : str
        dir saved model
    test_loader : `torch.utils.data.DataLoader`
        test loader
    experiment_name : str, optional
        expt name, by default ""
    trainer : trainer, optional
        pytorch lightning trainer, by default None
    print_test_acc : bool, optional
        print test accuracy, by default True
    valid_func : list, optional
        list of valid functions, by default None
    edge_mode : bool, optional
        whether for edge training, by default True
    Returns
    -------
    list
        image, target, prediction
    """
    saved_mod
    saved_model = get_saved_model(model, saved_model_dir, with_edge=edge_mode)
    if next(saved_model.parameters()).is_cuda:
        saved_model.cpu() 

    saved_model.eval()
    
    if trainer is not None:
        trainer.test(saved_model)
    elif print_test_acc:
        naive_test(test_loader, saved_model, valid_func)
        
    test_bat_num = random.randint(0,int(len(test_loader.dataset)/test_loader.batch_size)) if len(test_loader.dataset) else 0
    for i, batch in zip(range(test_bat_num+1),test_loader):
        x, y = batch

    pred_sgm=saved_model(x)
    if isinstance(pred_sgm, tuple):
        pred_sgm = pred_sgm[0]
    pred_sgm = pred_sgm.detach()
    # sft = nn.Softmax2d()
    # sft_pred = sft(pred_sgm)
    return x, y, pred_sgm

def save_pred_one(x, y, pred_sgm, saved_res_dir, bat_one=False, fig_name="res", valid_func=None):
    """save output as image

    Parameters
    ----------
    x : `torch.tensor`
        input
    y : `torch.tensor`
        target
    pred_sgm : `torch.tensor`
        predicted mask
    saved_res_dir : str
        path of dir
    bat_one : bool, optional
        whether batch size 1, by default False
    fig_name : str, optional
        name, by default "res"
    nosft : bool, optional
        whether to apply softmax, by default False
    box : list, optional
        list of coordinates, by default None
    label : list, optional
        labels, by default None
    """
    # img with 1/3 layers, whatever, takes first channel for displaying
    # pred after softmax.3 laeyrs, then do argmax, or 2 layers, display as two figures
    #y, 3 layers--argmax,   or 1 or no  layer or 2 layers display as 2 figures
    img_num = random. randint(0,x.shape[0]-1) 
    dice_coef = dice_coefficient(pred_sgm, y)
    if len(pred_sgm.shape)==4:
        pred_sgm = np.argmax(pred_sgm,axis=1)
    if y.shape[1]==3:
        y = np.argmax(y,axis=1)
    y=y.squeeze()
    if bat_one:
        #only one img in batch
        y=y.unsqueeze(dim=0)
    print(f"image index is {img_num}")
    if y.shape[1]!=2:
        plt.tight_layout()
        fig_img, ax = plt.subplots(1,3)
        fig_img.set_size_inches(18,10)
        ax[0].set_title('Image')
        ax[0].axis("off")
        ax[0].imshow(x[img_num,0,:,:])
        ax[1].set_title('Actual Segmentation Mask')
        ax[1].axis("off")
        ax[1].imshow(y[img_num,:,:])
        ax[2].set_title(f'Predicted Mask, dice score:{dice_coef:.4f}')
        ax[2].axis("off")
        ax[2].imshow(pred_sgm[img_num,:,:])
        plt.savefig(os.path.join(saved_res_dir,f"{fig_name}.png"), bbox_inches = 'tight')
    else:
        for label in range(pred_sgm.shape[1]):
            plt.tight_layout()
            fig_img, ax = plt.subplots(1,3)
            fig_img.set_size_inches(18,10)
            ax[0].set_title('Image')
            ax[0].axis("off")
            ax[0].imshow(x[img_num,0,:,:])
            ax[1].set_title('Actual Segmentation Mask')
            ax[1].axis("off")
            ax[1].imshow(y[img_num,label,:,:])
            ax[2].set_title('Predicted Segmentation Mask')
            ax[2].axis("off")
            ax[2].imshow(pred_sgm[img_num,label,:,:])
            # plt.show()
            plt.savefig(os.path.join(saved_res_dir,f"{fig_name}{str(label)}.png"), bbox_inches = 'tight')

def save_test_img_grid(x, y, pred_sgm, saved_res_dir, fig_name=""):
    """save grid of outputs as figure

    Parameters
    ----------
    x : `torch.tensor`
        input
    y : `torch.tensor`
        target
    pred_sgm : `torch.tensor`
        pred masks
    saved_res_dir : str
        path of dir
    fig_name : str, optional
        name, by default ""
    """
    if y.shape[1]==3:
        y = np.argmax(y,axis=1)
    dice_coefs = dice_coefficient(pred_sgm, y, True)
    if pred_sgm.shape[1]!=1 and len(pred_sgm.shape)==4:
        pred_sgm = np.argmax(pred_sgm,axis=1)
    y=y.squeeze()
    img_dim = y.shape[-1]
    x=x[:,0,:,:].view(-1,1,img_dim,img_dim)
    y=y.view(-1,1,img_dim,img_dim).float()
    pred_sgm = pred_sgm.view(-1,1,img_dim, img_dim).float()
    num_rows = 3
    #make_grid expects tensor of form B*C*H*W
    # multiplying by 10 to x which was in [0,1]range, so image is clear in the midst of segm which is in range [0,2]
    xp = x-x.min()
    xn = 2*xp/xp.max()
    comparison=(torch.cat((xn, y, pred_sgm)))
    # dividing into 2 rows, upper one for actual and other for pred
    comparison_image = make_grid(comparison, nrow=int(comparison.shape[0]/num_rows)) 
    # after make_grid, it converts to 3 channel images, so the shape is 3*H*W
    fig, ax=plt.subplots()
    round_dice_coeffs =[list(map(lambda num: round(num,4), coef)) for coef in dice_coefs.tolist()]
    fig.suptitle(f"input images, actual masks, predicted masks\n dice scores:[medial, lateral]\n{*round_dice_coeffs,}")
    ax.axis("off")
    fig.set_size_inches(6*3,6*int(comparison_image.shape[0]/num_rows))
    #choosing only one channel in make_grid so imshow scales o/p labels [0,1,2]accordingly, otherchannels are just repetitions
    output = plt.imshow(comparison_image[0])
    #below line would not scale labels, but shows in terms of [0-255] range, so unclear/dim segm image, hence not used
    #output = plt.imshow(comparison_image.permute(1, 2, 0)# permute used because imshow expects 3rd dimension as number of channels
    plt.savefig(os.path.join(saved_res_dir,f"{fig_name}.png"), bbox_inches = 'tight')