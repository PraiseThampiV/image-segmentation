import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision.utils import make_grid 
from loss.dice import dice_coefficient
from train_helper.naive_train import get_imgslr, AverageMeter
from post_proc.tta import aug_imgs
from post_proc.crfunsup import dense_crf
from loss.hausdorff import hdistance
from result_utils.incorr_display import save_pred_act

#TODO replace whole file code into a class
def naive_test(testloader, model, loss_func=None, postpr='tta', save_incorr=False):
    """calculate test accuracy

    Parameters
    ----------
    testloader : [type]
        testset loader
    model : `torch.module`
        model 
    loss_func : list, optional
        list of loss/accuracy functions, by default None
    postpr : str, optional
        whihc postprocessing `tta`|`crf`, by default 'no'
    save_incorr : bool, optional
        whether to save figures of incorrectly predicted masks, by default False
    """
    hd = AverageMeter()#hausdorff 
    incorr_disp_cnt = 0
    total = 0
    no_sft=False
    dice_total = 0
    if loss_func is None:
        loss_func = dice_coefficient
    with torch.no_grad():
        model.eval()
        for indt, data in enumerate(testloader):
            if indt>len(testloader.dataset):
                break
            images, segm = data
            if postpr=='tt':
                outputs = aug_imgs(model, images, chooseaugs=[2, 3])#only shear, flip, scale
                save_tt_error_fig = False
                if save_tt_error_fig:
                    from post_proc.tta import choose_func,aug_img, save_inter
                    save_inter(segm)
            elif postpr=='crf':
                outputs = model(images)
                sft = nn.Softmax2d()
                sft_outputs = sft(outputs)
                outputs = torch.stack([torch.from_numpy(dense_crf(images[i], sft_outputs[i])) for i in range(outputs.shape[0])])
                no_sft=True
            elif postpr=='ttcrf':
                outputs = aug_imgs(model, images)
                sft = nn.Softmax2d()
                sft_outputs = sft(outputs)
                outputs = torch.stack([torch.from_numpy(dense_crf(images[i], sft_outputs[i])) for i in range(outputs.shape[0])])
                no_sft=True
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                outputs = model.to(device)(images.to(device)).detach().cpu()
            if loss_func==dice_coefficient:
                hd.update(hdistance(outputs, segm))
            save_incorr=False
            if save_incorr:
                dice_list=loss_func(outputs, segm, nosft=no_sft, is_list_bat=True)
                selected = (dice_list>0.9).all(dim=1)
                save_dir = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/incorr_display"
                for index in torch.where(~selected)[0]:
                    save_pred_act(images[index], segm[index], outputs[index], save_dir, "res"+str(incorr_disp_cnt), dice_list[index])
                    incorr_disp_cnt+=1
            acc_metric = loss_func(outputs, segm, nosft=no_sft) if loss_func==dice_coefficient else loss_func(outputs, segm)
            dice_total+=acc_metric#already averaged by batch size in dice coeff func
            total += 1
    if loss_func==dice_coefficient:
        loss_name = "Dice coefficient"
        print(f"Hausdorff distance: {hd.avg.round(2)}")
    else:
        loss_name = "Loss"
    print(f'{loss_name} of the network on test images:{dice_total / total:.4f}')


def test_pred(model, saved_model_dir, test_loader, experiment_name="", trainer=None, print_test_acc=True, 
loss_func=None, sgm_train=True):
    """return x, pred, y as set containing test batch  items

    Parameters
    ----------
    model1 : `torch.nn.Module`
         model
    saved_model_dir : str
        saved model dir
    test_loader : `torch.utils.data.DataLoader`
        test loader
    experiment_name : str, optional
        expt name, by default ""
    trainer : trainer, optional
        pytorch lightning trainer, by default None
    print_test_acc : bool, optional
        print test accuracy, by default True
    loss_func : list, optional
        list of loss functions, by default None
    sgm_train : bool, optional
        whether for segmentation, by default True
    Returns
    -------
    list
        image, target, prediction
    """
    saved_model = model#UNETvgg()
    for filename in os.listdir(saved_model_dir):
        file_path = os.path.join(saved_model_dir, filename)
        if os.path.isfile(file_path) and "epoch" in file_path:
            save_model_path = file_path
            break
    checkpoint = torch.load(save_model_path)
    saved_model.load_state_dict(checkpoint['state_dict'])
    # if model in cuda change to cpu
    #TODO check if below needed for lytorch lightning test as well
    if next(saved_model.parameters()).is_cuda:
        saved_model.cpu() 

    saved_model.eval()
    
    if trainer is not None:
        trainer.test(saved_model)
    elif print_test_acc:
        naive_test(test_loader, saved_model, loss_func)
        
    if sgm_train:
        test_bat_num = random.randint(0,int(len(test_loader.dataset)/test_loader.batch_size)) if len(test_loader.dataset) else 0
        for i, batch in zip(range(test_bat_num+1),test_loader):
            x, y = batch

        if model.patch_train:
            x = get_imgslr(x, y)
            y = get_imgslr(y, y)
        pred_sgm=saved_model.to(x.device)(x).detach()#.cpu()
        #aug_imgs(saved_model, x)
        # sft = nn.Softmax2d()
        # sft_pred = sft(pred_sgm)
        return x, y, pred_sgm
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save_testimgno = 3
        #TODO delete old results
        save_cnt = 0
        save_img_folder = os.path.join(saved_model_dir, "saved_img")
        if not os.path.exists(save_img_folder):
            os.makedirs(save_img_folder)
        for batch in test_loader:
            x, y = batch
            pred_sl=saved_model.to(device)(x.to(device)).detach()
            if pred_sl.shape[-1]==80 or pred_sl.shape[-1]!=1:
                pred_sl = torch.argmax(pred_sl,dim=1)
            pred_sl = pred_sl.type(torch.int)
            act_sls = y.clone().detach().cpu().squeeze().numpy()
            pred_sls = pred_sl.clone().detach().cpu().squeeze().numpy()
            print(f"actual and predicted mid slices (m, l) are {act_sls}, {pred_sls}")
            predict_side = True
            if predict_side:
                for img, act, pred in zip(x, act_sls, pred_sls):
                    print(x.shape)
                    f, a = plt.subplots()
                    a.imshow(img[0])
                    f.savefig(os.path.join(saved_model_dir, f"img_{str(save_cnt)}_act_{act}_pred_{pred}.png"))
                    plt.close("all")
                    save_cnt += 1
                    if save_cnt > save_testimgno:
                        break
            else:
                if pred_sls[0]!=pred_sls[1]:
                    name_sls = {pred_sls[0]:"medial", pred_sls[1]:"lateral"}#name slices in figure
                else:
                    name_sls = None
                if save_cnt<save_testimgno:
                    oneimg_folder = os.path.join(save_img_folder, "img"+str(save_cnt)+"_pred_ml_"+str(pred_sls[0])+"_"+str(pred_sls[1])+"_act_ml_"+str(act_sls[0])+"_"+str(act_sls[1]))
                    if not os.path.exists(oneimg_folder):
                        os.makedirs(oneimg_folder)
                    for slno, slice in enumerate(x[0,0]):
                        f,a = plt.subplots()
                        a.axis("off")
                        a.imshow(slice)
                        if slno in pred_sls:
                            f.savefig(os.path.join(oneimg_folder, f"slice_{str(slno)}_pred_middle_slice_{name_sls[slno]}")) if name_sls is not None else f.savefig(os.path.join(oneimg_folder, f"slice_{str(slno)}_pred_middle_slice_medial_lateral"))
                        else:
                            f.savefig(os.path.join(oneimg_folder, "slice_"+str(slno)))
                        plt.close("all")
                save_cnt += 1


def save_pred_one(x, y, pred_sgm, saved_res_dir, bat_one=False, fig_name="res", nosft=False, channelcnt=None):
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
    no_sft : bool
        whether apply softmax2d
    channel_cnt : int
        num channels
    """
    # img with 1/3 layers, whatever, takes first channel for displaying
    # pred after softmax.3 laeyrs, then do argmax, or 2 layers, display as two figures
    #y, 3 layers--argmax,   or 1 or no  layer or 2 layers display as 2 figures
    img_num = random. randint(0,x.shape[0]-1) 
    dice_coef = dice_coefficient(pred_sgm, y, nosft=nosft, channelcnt=channelcnt)
    if y.shape[1]==3:
        y = np.argmax(y,axis=1)
    if channelcnt is None:
        channelcnt = 3
    if pred_sgm.shape[1]==channelcnt:
        pred_sgm = np.argmax(pred_sgm,axis=1)
    y=y.squeeze()
    if bat_one:
        #only one img in batch
        y=y.unsqueeze(dim=0)
    print(f"image index is {img_num}")
    if y.shape[1]!=2:
        # dice_coef = dice_coefficient(pred_sgm[img_num].view(1,1,pred_sgm.shape[-1],pred_sgm.shape[-1]), y[img_num].unsqueeze(dim=0))
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
            ax[0].imshow(x[img_num,0,:,:], cmap="gray")
            ax[1].set_title('Actual Segmentation Mask')
            ax[1].axis("off")
            ax[1].imshow(y[img_num,label,:,:])
            ax[2].set_title('Predicted Segmentation Mask')
            ax[2].axis("off")
            ax[2].imshow(pred_sgm[img_num,label,:,:])
            # plt.show()
            plt.savefig(os.path.join(saved_res_dir,f"{fig_name}{str(label)}.png"), bbox_inches = 'tight')

def save_test_img_grid(x, y, pred_sgm, saved_res_dir, fig_name="", nosft=False, channel_cnt=None):
    """save outputs as image grid

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
    no_sft : bool
        whether apply softmax2d
    channel_cnt : int
        num channels
    """
    if y.shape[1]==2:
        return
    dice_coefs = dice_coefficient(pred_sgm, y, is_list_bat=True, nosft=nosft, channelcnt=channel_cnt)
    if y.shape[1]==3:
        y = np.argmax(y,axis=1)
    if channel_cnt is None:
        channel_cnt = 3
    if pred_sgm.shape[1]==channel_cnt:
        pred_sgm = np.argmax(pred_sgm,axis=1)
    y=y.squeeze()
    img_dim = y.shape[-1]
    x=x[:,0,:,:].view(-1,1,img_dim,img_dim)
    y=y.view(-1,1,img_dim,img_dim).float()
    pred_sgm = pred_sgm.view(-1,1,img_dim, img_dim).float()
    # dice_coefs = dice_coefficient(pred_sgm.long(), y.long(), is_list_bat=True)
    num_rows = 3
    #make_grid expects tensor of form B*C*H*W
    # confine values to 0 to 1 and then multiplying by 3 to x which was in [0,1]range, so image is clear in the midst of segm which is in range [0,2]
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