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

def get_wholesegm(box, pred, segm, label):
    """do softmax, add output patch to whole predimg, take argmax, send it(make sure no softmax again)

    Parameters
    ----------
    box : list
        list of coordinates
    pred : `torch.tensor`
        logits
    segm : `torch.tensor`
        target
    label : list
        list of labels

    Returns
    -------
    wholesegm
        pred segm
    """
    xmin, ymin, xmax, ymax = box
    rem_label = {1:2,2:1}
    size = [ymax-ymin, xmax-xmin]
    modi_segm = segm.clone()
    sft = nn.Softmax2d()
    pred = sft(pred)
    wholesegm = torch.zeros_like(pred)
    # for img, lab, ywid, xwid, ind in zip(modi_segm, label, size[0], size[1], range(len(segm))):
    for ind in range(len(segm)):
        resize = nn.Upsample(size=[size[0][ind], size[1][ind]], mode="bilinear")
        outputs = resize(pred[ind].unsqueeze(dim=0)).squeeze(dim=0)
        # outputs = resize(pred.unsqueeze(dim=1).float()).squeeze(dim=1).long()#TODO try resizing after argmax also
        modi_segm[ind][modi_segm[ind]==rem_label[label[ind].item()]]=0
        modi_segm[ind][modi_segm[ind]!=0]=1
        wholesegm[ind,:,ymin[ind]:ymax[ind], xmin[ind]:xmax[ind]] = outputs
    # wholesegm = sft(wholesegm)
    wholesegm = torch.argmax(wholesegm, dim=1)
    return wholesegm, modi_segm
#TODO replace whole file code into a class
def naive_test(testloader, model, loss_func=None, postpr='no', save_incorr=False):
    """calculate test set accuracy

    Parameters
    ----------
    testloader : `torch.utils.data.DataLoader`
        testset loader
    model1 : `torch.nn.module`
        model
    loss_func : list, optional
        list of loss/accuracy functions, by default None
    postpr : str, optional
        which postprocessing, `crf`|`tta`, by default 'no'
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
        for data in testloader:
            images, segm, box, label, segm_orig = data
            # if postpr=='tt':
            #     outputs = aug_imgs(model, images)
            # elif postpr=='crf':
            #     outputs = model(images)
            #     sft = nn.Softmax2d()
            #     sft_outputs = sft(outputs)
            #     outputs = torch.stack([torch.from_numpy(dense_crf(images[i], sft_outputs[i])) for i in range(outputs.shape[0])])
            #     no_sft=True
            # elif postpr=='ttcrf':
            #     outputs = aug_imgs(model, images)
            #     sft = nn.Softmax2d()
            #     sft_outputs = sft(outputs)
            #     outputs = torch.stack([torch.from_numpy(dense_crf(images[i], sft_outputs[i])) for i in range(outputs.shape[0])])
            #     no_sft=True
            # else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            outputs = model.to(device)(images.to(device)).detach().cpu()
            wholesegm, segm = get_wholesegm(box, outputs, segm_orig, label)
            
            # hd.update(hdistance(outputs, segm))
            save_incorr=False
            if save_incorr:
                dice_list=loss_func(wholesegm, segm, nosft=no_sft, is_list_bat=True)
                selected = (dice_list>0.9).all(dim=1)
                save_dir = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/check_img"
                for index in torch.where(~selected)[0]:
                    save_pred_act(images[index], segm[index], wholesegm[index], save_dir, "res"+str(incorr_disp_cnt), dice_list[index])
                    incorr_disp_cnt+=1
            no_sft=True
            acc_metric = dice_coefficient(wholesegm, segm, nosft=no_sft, channelcnt=2)
            dice_total+=acc_metric#already averaged by batch size in dice coeff func
            total += 1
    if loss_func==dice_coefficient:
        loss_name = "Dice coefficient"
    else:
        loss_name = "Loss"
    print(f'{loss_name} of the network on test images:{dice_total / total:.4f}')
    # print(f"Hausdorff distance: {hd.avg.round(2)}")

def test_pred(model, saved_model_dir, test_loader, experiment_name="", trainer=None, print_test_acc=True, 
loss_func=None, sgm_train=True):
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
            x, y, box, label, segm_orig = batch

        if model.patch_train:
            x = get_imgslr(x, y)
            y = get_imgslr(y, y)
        pred_sgm=saved_model.to(x.device)(x).detach()#.cpu()
        #aug_imgs(saved_model, x)
        # sft = nn.Softmax2d()
        # sft_pred = sft(pred_sgm)
        return x, segm_orig, pred_sgm, box, label
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
            if pred_sl.shape[-1]==80:
                pred_sl = torch.argmax(pred_sl,dim=1)
            pred_sl = pred_sl.type(torch.int)
            act_sls = y.clone().detach().cpu().squeeze().numpy()
            pred_sls = pred_sl.clone().detach().cpu().squeeze().numpy()
            print(f"actual and predicted mid slices (m, l) are {act_sls}, {pred_sls}")
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


def save_pred_one(x, y, pred_sgm, saved_res_dir, bat_one=False, fig_name="res", nosft=False, box=None, label=None):
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
    pred_sgm, y = get_wholesegm(box, pred_sgm, y, label)
    img_num = random. randint(0,x.shape[0]-1) 
    dice_coef = dice_coefficient(pred_sgm, y, nosft=nosft,channelcnt=2)
    if y.shape[1]==3:
        y = np.argmax(y,axis=1)
    if len(pred_sgm.shape)==4 and pred_sgm.shape[1]!=1:
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
            ax[0].imshow(x[img_num,0,:,:])
            ax[1].set_title('Actual Segmentation Mask')
            ax[1].axis("off")
            ax[1].imshow(y[img_num,label,:,:])
            ax[2].set_title('Predicted Segmentation Mask')
            ax[2].axis("off")
            ax[2].imshow(pred_sgm[img_num,label,:,:])
            # plt.show()
            plt.savefig(os.path.join(saved_res_dir,f"{fig_name}{str(label)}.png"), bbox_inches = 'tight')

def save_test_img_grid(x, y, pred_sgm, saved_res_dir, fig_name="", nosft=False, box=None, label=None):
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
    nosft : bool, optional
        whether to apply softmax, by default False
    box : list, optional
        list of coordinates, by default None
    label : list, optional
        labels, by default None
    """
    pred_sgm, y = get_wholesegm(box, pred_sgm, y, label)
    if y.shape[1]==2:
        return
    dice_coefs = dice_coefficient(pred_sgm, y, is_list_bat=True, nosft=nosft, channelcnt=2)
    if y.shape[1]==3:
        y = np.argmax(y,axis=1)
    if len(pred_sgm.shape)==4 and pred_sgm.shape[1]!=1:
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