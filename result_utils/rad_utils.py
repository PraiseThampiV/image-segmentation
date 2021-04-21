import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import matplotlib
from PIL import Image 
from numpy import asarray
import PIL 
from skimage.measure import label
def save_3d_psl(img3d, pred_sl, save_img_folder, save_testimgno=1, save_cnt=0, foldername=None):
    """save 3d image and predicted slice num

    Parameters
    ----------
    img3d : `array like`
        3d image
    pred_sl : int
        predicted slice num
    save_img_folder : str
        save location
    save_testimgno : int, optional
        nth image getting saved(for naming), by default 1
    save_cnt : int, optional
        total count of images to be saved, by default 0
    foldername : str, optional
        folder name, by default None
    """
    if save_cnt<save_testimgno:
        if torch.is_tensor(pred_sl):
            pred_sl = pred_sl.item()
        if foldername is None:
            foldername = "img"+str(save_cnt)
        oneimg_folder = os.path.join(save_img_folder, foldername+"_pred_midslice_"+str(pred_sl))
        if not os.path.exists(oneimg_folder):
            os.makedirs(oneimg_folder)
        for slno, slic in enumerate(img3d[0,0]):
            f,a = plt.subplots()
            a.axis("off")
            a.imshow(slic)
            if slno==pred_sl:
                f.savefig(os.path.join(oneimg_folder, "slice_"+str(slno)+"_pred_middle_slice"), bbox_inches = 'tight', pad_inches = 0)
            else:
                f.savefig(os.path.join(oneimg_folder, "slice_"+str(slno)), bbox_inches = 'tight', pad_inches = 0)
            plt.close("all")

#saving image volume with 2 middle slices, predited slices 2
def save_3d_psl2(img3d, pred_sl, save_img_folder, save_testimgno=1, save_cnt=0, foldername=None):
    """save 3d image and predicted slice num

    Parameters
    ----------
    img3d : array like
        image 3d
    pred_sl : int
        predicted slice
    save_img_folder : str
        save location
    save_testimgno : int, optional
        nth image getting saved(for naming), by default 1
    save_cnt : int, optional
        total count of images to be saved, by default 0
    foldername : str, optional
        folder name, by default None
    """
    if save_cnt<save_testimgno:
        if torch.is_tensor(pred_sl):
            pred_sl = pred_sl.clone().detach().cpu().squeeze().numpy()
        if foldername is None:
            foldername = "img"+str(save_cnt)
        oneimg_folder = os.path.join(save_img_folder, foldername+"_pred_midmedial_"+str(pred_sl[0])+"_midlateral_"+str(pred_sl[1]))
        if not os.path.exists(oneimg_folder):
            os.makedirs(oneimg_folder)
        name_fig = {pred_sl[0]:"medial", pred_sl[1]:"lateral"} if pred_sl[0]!=pred_sl[1] else None
        for slno, slic in enumerate(img3d[0,0]):
            f,a = plt.subplots()
            a.axis("off")
            a.imshow(slic)
            if slno in pred_sl:
                if name_fig is not None:
                    f.savefig(os.path.join(oneimg_folder, f"slice_{str(slno)}_pred_middle_{name_fig[slno]}"), bbox_inches = 'tight', pad_inches = 0)
                else:
                    f.savefig(os.path.join(oneimg_folder, f"slice_{str(slno)}_pred_mid_medial_lateral"), bbox_inches = 'tight', pad_inches = 0)
            else:
                f.savefig(os.path.join(oneimg_folder, "slice_"+str(slno)), bbox_inches = 'tight', pad_inches = 0)
            plt.close("all")

def save_3dimg(img3d,save_img_folder, img_ind=0, foldername=None):
    """save 3d image

    Parameters
    ----------
    img3d : array like
        image 3d
    save_img_folder : str
        save location
    img_ind : int, optional
        nth image getting saved(for naming), by default 1
    foldername : str, optional
        folder name, by default None
    """
    if foldername is None:
        foldername = "img"+str(img_ind)
    oneimg_folder = os.path.join(save_img_folder, foldername)
    if not os.path.exists(oneimg_folder):
        os.makedirs(oneimg_folder)
    for slno, slic in enumerate(img3d):
        f,a = plt.subplots()
        a.axis("off")
        a.imshow(slic)
        f.savefig(os.path.join(oneimg_folder, "slice_"+str(slno)), bbox_inches = 'tight', pad_inches = 0)
        plt.close("all")



def get_png_save_np():
    """adding manually segmented images to training set
    """
    check_img = True
    if check_img:
        root_folder = "/home/students/thampi/PycharmProjects/meniscus_data/segm_npy"
        for ind in range(1, 22):
            ind = str(ind)
            img = np.load(os.path.join(root_folder, "img_npy", "batch"+ind+".npy"))
            seg = np.load(os.path.join(root_folder, "seg_npy", "batch"+ind+".npy"))
            f, a = plt.subplots()
            a.imshow(img, cmap="gray")
            a.imshow(seg, alpha=0.5)
            f.savefig(r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/manual_save/img"+ind+".png")    
            plt.close("all")

    segorimg = None
    png_loc = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/manual_add_train_set"
    seg_png = os.listdir(png_loc+"/segm")
    img_png = os.listdir(png_loc+"/img")
    subfolder = os.listdir(png_loc)[segorimg]

    save_dir = r"/home/students/thampi/PycharmProjects/meniscus_data/segm_npy"
    out_sub = [ "seg_npy", "img_npy"][segorimg]
    for img_name in os.listdir(os.path.join(png_loc, subfolder)):
        img = Image.open(os.path.join(png_loc, subfolder, img_name)).convert('LA')   
        if subfolder=="img":
            img = img.resize([512, 512])
            numpydata = asarray(img).copy()[:,:,0]
            numpydata = numpydata/numpydata.max()
        else:
            img = img.resize([512, 512], resample=PIL.Image.NEAREST)
            numpydata = asarray(img).copy()[:,:,0]
            numpydata[numpydata==127]=2
            numpydata[numpydata==255]=1
            if len(np.unique(numpydata))!=3:
                print("wrong mask")
        with open(os.path.join(save_dir, out_sub, "batch"+img_name.split(".")[0]+".npy"), 'wb') as f:
            np.save(f, numpydata)

def transform_sitk(original_CT, reference_size):
    #https://stackoverflow.com/questions/48065117/simpleitk-resize-images
    # original_CT = sitk.ReadImage(patient_CT,sitk.sitkInt32)
    dimension = original_CT.GetDimension()
    reference_physical_size = np.zeros(original_CT.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]
    
    reference_origin = original_CT.GetOrigin()
    reference_direction = original_CT.GetDirection()

    # reference_size = [round(sz/resize_factor) for sz in original_CT.GetSize()] 
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(original_CT.GetDirection())

    transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    
    return sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)


def save_masks(items, im_th, save_fig_dir, knee_img_cnt=None, heading="", if_correct=False):
    """save mask

    Parameters
    ----------
    items : list
        image, mask, locatio, label
    im_th : int
        turn of this image
    save_dir : str
        save dir
    knee_img_cnt : int, optional
        `items`th turn in the whole knee image set, by default None
    if_correct : bool
        if mask prediction correct or wrong
    """
    # for filename in os.listdir(save_fig_dir):
    #     file_path = os.path.join(save_fig_dir, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print('Failed to delete %s. Reason: %s' % (file_path, e))
    mask_per_page = 1
    no_r = mask_per_page
    no_col  = 2
    img, seg, slice_loc, men_label = items
    which_menisc = ["medial(blue)", "lateral(green)"]
    if not if_correct:
        heading = heading + "(bad prediction)"
    if not im_th%mask_per_page:
        fig, ax = plt.subplots(no_r, no_col)
        [axi.set_axis_off() for axi in ax.ravel()]
        fig.set_size_inches(18, 10)  # width, height
        fig.suptitle(f"Segmentation Mask for central {which_menisc[men_label-1]} meniscus {heading}")
    row_ind = im_th%mask_per_page#, col = np.unravel_index(im_th, [no_r, no_r])
    col_ind = 0#col*2
    if mask_per_page>1:
        curr_ax = ax[row_ind]
    else:
        curr_ax = ax
    curr_ax[col_ind].imshow(img, cmap="gray") 
    curr_ax[col_ind].title.set_text("image")
    # ax[row_ind][col_ind].axis("off")
    sg_mask = np.ma.masked_array(seg, seg == 0)
    curr_ax[col_ind+1].imshow(img, cmap="gray")
    colorsList = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
    CustomCmap = matplotlib.colors.ListedColormap(colorsList)
    im = curr_ax[col_ind+1].imshow(sg_mask, cmap=CustomCmap)#"gist_rainbow")
    curr_ax[col_ind+1].title.set_text("with mask")
    cbar = fig.colorbar(im, ax=curr_ax[col_ind+1], fraction=0.046, pad=0.04, ticks=[0,1,2])
    im.set_clim(0, 2)
    # ax[row_ind][col_ind+1].axis("off")
    if im_th%mask_per_page or mask_per_page==1:
        fig.savefig(os.path.join(save_fig_dir,f"img_{str(knee_img_cnt)}_pred_mask{men_label}.png"))
        plt.close("all")


def get_largest_island(pred):
    # retaining only the largest island mask
    island_lab = label(pred)#gives label to every similarly labeled connected region, island with labels
    _, cnt_label = np.unique(island_lab, return_counts=True)
    new_pred = np.zeros_like(pred)
    if len(cnt_label) > 1:
        meni_label = np.unique(pred)[1]#sorted array, get medial or lateral menisci label
        req_label = np.argsort(cnt_label)[-2]#getting the label with second largest, required label
        new_pred[island_lab==req_label] = meni_label
    return new_pred

def save_both(both_state, slice_ind, sfolders, medial_items, lateral_items, wrong_mask_cnt, good_mask_cnt, knee_img_cnt):
    #save both medial and lateral masks in folders accordingly if both, one or neither among them correct
    if slice_ind:#during the turn of lateral mask, prediction, save both
        spath = sfolders[both_state.how_good()]#save folder path
        for ind, items in enumerate([medial_items, lateral_items]):
            img, mask, if_corr = items
            if if_corr:
                save_masks([img, mask, "", 1+ind], good_mask_cnt, spath, knee_img_cnt, if_correct=if_corr)
                good_mask_cnt += 1
            else:
                wrong_mask_cnt += 1
                save_masks([img, mask, "", 1+ind], wrong_mask_cnt, spath, knee_img_cnt, if_correct=if_corr)
    return good_mask_cnt, wrong_mask_cnt

def choose_quadrant(side_pred, pred_mask, slice_type_ind):
    """retain only bottom-left or bottom right quadrant of pred mask image, depending on prediction `if_left` from `modelside` 

    Parameters
    ----------
    side_pred : tuple
        whether lateral meniscus on left(1, 0) or right(0, 1)
    slice_type_ind : int
        if 0: medial, 1: lateral
    """
    side = np.argmax(side_pred, axis=1)
    out_img = np.zeros_like(pred_mask)
    width, height = pred_mask.shape[-1], pred_mask.shape[-2]
    if (slice_type_ind==0 and side==1) or (slice_type_ind==1 and side==0):#medial, then lateral
        out_img[..., height//2 :, :width//2] = pred_mask[..., height//2: , : width//2]
    else:
        out_img[..., height//2: , width//2:] = pred_mask[..., height//2: , width//2:]
    return out_img