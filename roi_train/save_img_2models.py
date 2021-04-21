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
from gen_utils.img_utils import get_roi_tens, get_patch

def choose_majority(img):
    """assign majority occuring label to each image

    Parameters
    ----------
    img : `torch.tensor`
        image

    Returns
    -------
    `torch.tensor`
        modified image
    """
    labels, cnts = torch.unique(img, return_counts=True)
    if len(labels)==3:
        if cnts[1]>cnts[2]:
            img[img==2]=1
        else:
            img[img==1]=2
        return img

def clean_outliers(pred_imgs):
    """pred_img containing labels,
    the labels of one menisci predicted at the region of other type menisci and make it to expected.
    basically assigning majority label to clusters in half mage

    Parameters
    ----------
    pred_img : `torch.tensor`
        image
    returns: bool
        returns if pred_img is faulty(predicting both clusters with same label)

    """
    #if mid width pixel lying in the range of ends, expecting two fg labels
    #if mutliple labels in half image, clean it
    if_faulty = []
    for pred_img in pred_imgs: 
        if len(torch.unique(pred_img))==1:
            if_faulty.append(True)
            continue
        ends = get_roi_tens(pred_img)
        mid_wid = pred_img.shape[-1]//2
        mid_lie = ends[0]<mid_wid and ends[2]>mid_wid#whether image mid-width lie in range of xmin,xmax
        cnt_lbls = len(torch.unique(pred_img))
        if cnt_lbls==3 and mid_lie:
            half1 = pred_img[...,:mid_wid]
            half1 = choose_majority(half1)
            half2 = pred_img[...,mid_wid:]
            half2 = choose_majority(half2) #assigning to source image
            if_faulty.append(False) if len(torch.unique(pred_img))==3 else if_faulty.append(True) #when majority label in both halves is same,faulty image
        if cnt_lbls==2 and mid_lie or cnt_lbls==1:
            if_faulty.append(True)
        if cnt_lbls==2 and not mid_lie:
            if_faulty.append(False)
    return if_faulty, pred_imgs
def detectandsegm(img, model1, model2, crop_size, allow_faulty=True):
    """get img, use first model, detect upto two regions, pass them through model2, combine images, send whole img pred
    img : `torch.tensor`
        input image
    model1 : `torch.module`
        whole image trained model
    model2 : `torch.module`
        trained `UNetSimple` module for patches
    crop_size : int, optional
        crop image to size
    allow_faulty: bool
        when detection results two clusters with same label, still allow such preds or not

    returns : None/torch.tensor
        nothing or image depending on `allow_faulty`
    """
    detectout = model1(img)
    sft = nn.Softmax2d()
    detectout = sft(detectout)
    outlabel = torch.argmax(detectout, dim=1)#out with labels
    try:
        if_faulty, outlabel = clean_outliers(outlabel)
    except Exception as e:
        k=1
    if_faulty = np.array(if_faulty, np.bool)
    if not allow_faulty and if_faulty.any():
        img = img[~ if_faulty]
        outlabel = outlabel[~ if_faulty]
        if not len(img):
            return None #TODO return nothing and verify
    poss_lbls = torch.unique(outlabel)#possible labels
    #TODO remove 0 from poss_lbls, or sort it
    wholepredlbl = torch.zeros_like(outlabel)
    for img_ind in range(len(img)):#work on image by image for easy aggregation of 2 patches
        if len(torch.unique(outlabel[img_ind]))==1:
            continue
        label_list = []
        ends_list = []
        patches = []
        # if if_faulty[img_ind]:
            #then two clusters with same label
        mid_wid = outlabel[img_ind].shape[-1]//2
        half1 = outlabel[img_ind][...,:mid_wid]
        half2 = outlabel[img_ind][...,mid_wid:]
        for hi, eachhalf in enumerate([half1, half2]):#hi half index
            lbls = torch.unique(eachhalf)
            if len(lbls)==1:
                continue
            elif len(lbls)==2:
                lbl=lbls.max()
            else:
                print("clear outliers failed, multiple labels per half")
                continue
            label_list.append(lbl)
            ends = get_roi_tens(eachhalf, margin=0)#margin=10 while training
            if hi==1:
                ends[0] += mid_wid
                ends[2] += mid_wid #adding additional width for second half
            #make patches to squares (not rectangles) impt, happens inside get patch func
            margin = 10#TODO, try 5,10,0
            ends[0] = max(ends[0]-margin, 0)#xmin
            ends[2] = min(ends[2]+margin, outlabel.shape[-1]-1)
            patch, ends = get_patch(ends, img[img_ind], crop_size, margin=0, is_long=False, send_ends=True)
            patches.append(patch)
            ends_list.append(ends)
        # else:
        #     for lbl in poss_lbls[1:]:
        #         label_list.append(lbl)
        #         ends = get_roi_tens(outlabel[img_ind], label=lbl)
        #         ends_list.append(ends)
        #         patches.append(get_patch(ends, img[img_ind], crop_size, margin=0, is_long=False))

        in_patches = torch.stack(patches)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out_pats = model2.to(device)(in_patches.to(device)).detach().cpu()
        out_pats = sft(out_pats)
        for i in range(len(patches)):
            ends = ends_list[i]
            # imgupsam = nn.Upsample(size=[ends[3]-ends[1], ends[2]-ends[0]], mode='bilinear', align_corners=False)
            y, x = ends[3]-ends[1], ends[2]-ends[0]
            # imgupsam = nn.Upsample(size=[y, x], mode='bilinear', align_corners=True)
            imgupsam = nn.Upsample(size=[y, x], mode='bicubic', align_corners=True)
            patch = imgupsam(out_pats[i].unsqueeze(dim=0)).squeeze(dim=0)
            pat_lbl=torch.argmax(patch, dim=0)#patch with labels
            pat_lbl[pat_lbl!=0]=label_list[i]
            wholepredlbl[img_ind][...,ends[1]:ends[3], ends[0]:ends[2]] = pat_lbl
    return img, wholepredlbl, if_faulty

def get_wholesegm(box, pred, segm, label):
    """do softmax, add output patch to whole predimg, take argmax, send it(make sure no softmax again)

    Parameters
    ----------
    box : list
        list of box coordinates
    pred : `torch.tensor` 
        output prediction
    segm : `torch.tensor`
        target
    label : list
        label

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
def naive_test(testloader, model1, loss_func=None, postpr='no', save_incorr=False, crop_size=None, model2=None, allow_faulty=None):
    """calculate test accuracy

    Parameters
    ----------
    testloader : [type]
        testset loader
    model1 : `torch.module`
        model for whole image
    loss_func : list, optional
        list of loss/accuracy functions, by default None
    postpr : str, optional
        whihc postprocessing, by default 'no'
    save_incorr : bool, optional
        whether to save figures of incorrectly predicted masks, by default False
    crop_size : int, optional
        crop to size, by default None
    model2 : `torch.module`, optional
        segmentation model for patches, by default None
    allow_faulty : bool, optional
        include null predictions, by default None
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
            images, segm = data
            res = detectandsegm(images, model1, model2, crop_size, allow_faulty=allow_faulty)
            if res is None:
                continue
            images, output, if_faulty = res
            if not allow_faulty:
                segm = segm[~if_faulty]
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # outputs = model.to(device)(images.to(device)).detach().cpu()
            # wholesegm, segm = get_wholesegm(box, outputs, segm_orig, label)
            
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
            acc_metric = dice_coefficient(output, segm, nosft=True, channelcnt=3)
            dice_total+=acc_metric#already averaged by batch size in dice coeff func
            total += 1
    if loss_func==dice_coefficient:
        loss_name = "Dice coefficient"
    else:
        loss_name = "Loss"
    print(f'{loss_name} of the network on test images:{dice_total / total:.4f}')
    # print(f"Hausdorff distance: {hd.avg.round(2)}")

def test_pred(model1, model2, test_loader, experiment_name="",
loss_func=None, sgm_train=True, crop_size=None, allow_faulty=True):
    """return x, pred, y as set containing test batch  items

    Parameters
    ----------
    model1 : `torch.nn.Module`
         model for whole images
    model2 : `torch.module`
        trained segmentation model for patches
    test_loader : `torch.utils.data.DataLoader`
        test loader
    experiment_name : str, optional
        expt name, by default ""
    loss_func : list, optional
        list of loss functions, by default None
    sgm_train : bool, optional
        whether for segmentation, by default True
    crop_size : int, optional
        crop to, by default None
    allow_faulty : bool, optional
        include null prediction, by default True

    Returns
    -------
    list
        image, target, prediction
    """
    # saved_model = model#UNETvgg()
    # for filename in os.listdir(saved_model_dir):
    #     file_path = os.path.join(saved_model_dir, filename)
    #     if os.path.isfile(file_path) and "epoch" in file_path:
    #         save_model_path = file_path
    #         break
    # checkpoint = torch.load(save_model_path)
    # saved_model.load_state_dict(checkpoint['state_dict'])
    # if model in cuda change to cpu
    #TODO check if below needed for lytorch lightning test as well
    if next(model1.parameters()).is_cuda:
        model1.cpu() 
    model1.eval()
    if next(model2.parameters()).is_cuda:
        model2.cpu() 
    model2.eval()
    
    naive_test(test_loader, model1, loss_func, crop_size=crop_size, model2=model2, allow_faulty=allow_faulty)
        
    if sgm_train:
        test_bat_num = random.randint(0,int(len(test_loader.dataset)/test_loader.batch_size)) if len(test_loader.dataset) else 0
        for i, batch in zip(range(test_bat_num+1),test_loader):
            x, y = batch

        x, pred_sgm, if_faulty=detectandsegm(x, model1, model2, crop_size, allow_faulty)
        if not allow_faulty:
            y=y[~if_faulty]
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