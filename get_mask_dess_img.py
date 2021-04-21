import six
import os, shutil
import json
import torch
import math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import radiomics
import torch.nn.functional as F
import torchvision.transforms as tvn
from radiomics import featureextractor
from torch.utils.data import DataLoader

from semeda_train.res_save import get_saved_model, test_pred
from data_read.meniscus_data import MeniscusDataFromArray, MeniscusDataAllOptions
from models.unetvgg import UNetSimple, SpiderNet
from data_read.read_3dinput import OAIData3DFromArray
from train_helper.naive_train import get_imgslr
from post_proc.tta import aug_imgs
from post_proc.crfunsup import dense_crf
from skimage.measure import label
from models.resnt import DetectMid, DetectSide
from result_utils.rad_utils import save_3d_psl2, save_3dimg, transform_sitk, save_masks, get_largest_island, save_both, choose_quadrant, get_png_save_np

def combine_imgslr(imglr, segm):
    """receives a batch containingleft images and right images, and combines them to form whole images,
    assuming that first half set of images in batch belong to left iamges and the second half of batch has 
    corresponding right images

    Parameters
    ----------
    imglr : `torch.tensor`
        left and right images
    segm : `torch.tensor`
        segmentation image

    Returns
    -------
    `torch.tensor`
    """
    hmin, hmax = torch.nonzero(segm, as_tuple=True)[1].min(), torch.nonzero(segm, as_tuple=True)[1].max()
    wmin, wmax = torch.nonzero(segm, as_tuple=True)[2].min(), torch.nonzero(segm, as_tuple=True)[2].max()
    margin=15
    orig_hmax = segm.shape[-2]
    orig_wmax = segm.shape[-1]
    crop_hmin, crop_hmax = max(0, hmin-margin), min(hmax+margin, orig_hmax)
    crop_wmin, crop_wmax = max(0, wmin-margin), min(wmax+margin, orig_wmax)
    # first half images to left
    outl = imglr[:imglr.shape[0]//2,...]
    outr = imglr[imglr.shape[0]//2:,...]

    height, width = crop_hmax - crop_hmin, crop_wmax - crop_wmin
    lastl = torch.nn.AdaptiveMaxPool2d([height,width//2] )
    lastr = torch.nn.AdaptiveMaxPool2d([height,width-width//2] )

    outl=lastl(outl)
    outr=lastr(outr)

    out = torch.cat((outl, outr), dim=-1)
    out = F.pad(out, (crop_wmin-1, orig_wmax-crop_wmax+1, crop_hmin-1, orig_hmax-crop_hmax+1))
    return out

def get_radfeatures(index, data_path, excel_svdir=None, model_loc=None, test_set=None, kwargs=None, model=None):
    """sample function to check radiometric extraction

    Parameters
    ----------
    index : int
        slice number of image slice
    data_path : str
        data path
    excel_svdir : str, optional
        save dir, by default None
    model_loc : str, optional
        model  location, by default None
    test_set : `torch.Dataset`, optional
        test set, by default None
    kwargs : list, optional
        hyper parameters, by default None
    model : `torch.module`, optional
        model, by default None

    Returns
    -------
    dict
        radiometrics
    """
    img_dir = os.path.join(data_path, "img_npy")
    
    all_npy_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) 
        and "Slice" in os.path.join(img_dir, f) and ".npy" in os.path.join(img_dir, f)]
    img_slice = np.load(os.path.join(img_dir, all_npy_files[index]))

    if model_loc:
        test_loader = DataLoader(test_set, batch_size=1, **kwargs)
        cnt_img = 600
        for batch, cnt in zip(test_loader, range(cnt_img)):
            x, y = batch
            saved_model = get_saved_model(model, model_loc, with_edge=False)
            saved_model.eval()
            # x, y, pred = test_pred(model, model_loc, test_loader, edge_mode=False)
            pred = saved_model(x)[0].detach().cpu().numpy()
            img_slice = x[0][0]
            mask_slice =np.argmax(pred[0], axis=0)#almost like squeezing op
            slice_nm = '1' #  naming file

            sitk_img = sitk.GetImageFromArray(img_slice)

            sitk_mask = sitk.GetImageFromArray(mask_slice)

            extractor = featureextractor.RadiomicsFeatureExtractor()
            # extractor.disableAllFeatures()
            # # extractor.enableImageTypeByName('LoG')
            # for name in ['firstorder', 'shape', 'ngtdm']:
            #     extractor.enableFeatureClassByName(name)
            #     # glrm, glszm, glcm,  'glszm', 'gldm', 
            # extractor.enableFeaturesByName(glcm=['Autocorrelation', 'Homogeneity1', 'SumSquares'])
            # features = extractor.execute(sitk_img, sitk_mask)
            features = extractor.execute(sitk_img, sitk_mask, label=2)
            if cnt==0:
                df = pd.DataFrame.from_dict(data=features, orient='index')
            else:
                df["scan_"+str(cnt)]=features.values()
            # df = (df.T)
            print (df)
        if excel_svdir:
            excel_svpath = os.path.join(excel_svdir, f'feat_{slice_nm}.xlsx')
            df.to_excel(excel_svpath)
        return features

    else:
        seg_dir = os.path.join(data_path, "seg_npy")
        mask_slice = np.load(os.path.join(seg_dir, all_npy_files[index]))
        slice_nm = all_npy_files[index].replace('.npy',"")


def np_to_dicom(arr, sample_dicom_slice):
    """convert numpy array to numpy

    Parameters
    ----------
    arr : `numpy.array`
        image
    sample_dicom_slice : `SimpleITK.Image`
        simple itk image

    Returns
    -------
    `SimpleITK.Image`
        converted image
    """
    img = sample_dicom_slice
    sitk_arr = sitk.GetImageFromArray(arr)
    sitk_arr.SetSpacing(img.GetSpacing())
    sitk_arr.SetDirection(img.GetDirection())#important
    sitk_arr.SetOrigin(img.GetOrigin())
    return sitk_arr

def round_np(vals):
    """round the decimal values

    Parameters
    ----------
    vals : `float`
        values from radiometrics

    Returns
    -------
    `float`
        rounded values/array
    """
    if isinstance(vals, np.float) or isinstance(vals, np.ndarray):
        vals = vals.round(3)
    elif isinstance(vals[0], np.float):
        vals = [eac.round(2) for eac in vals]
    return vals

class excel_radfeat():
    #alowing feature extration for only one label per image
    """helper for saving radiometrics
    """
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.count = 0
        self.flist1 = {}
        self.flist2 = {}
        self.sdicts = [None, None]#1 for medial, 2 for lateral, series dicts
        self.special_keys = [ #'diagnostics_Mask-original_Size', 512X512X1
        'diagnostics_Mask-original_BoundingBox','diagnostics_Mask-original_CenterOfMassIndex',# 'diagnostics_Mask-original_CenterOfMass', #TODO correct direction
         'original_shape_Elongation', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea']
    def gt_upd_radfeat_sitk(self, img, seg, dir_dicom, label):
        """img, seg in sitk format

        Parameters
        ----------
        img : SimpleITK Image
            image
        seg : Simple ITK Image
            segmentation mask
        dir_dicom : str
            dicom location
        label : int
            label value
        """
        extractor = featureextractor.RadiomicsFeatureExtractor()
        sitk_mask = sitk.GetImageFromArray(seg)
        sitk_mask.SetSpacing(img.GetSpacing())
        sitk_mask.SetDirection(img.GetDirection())#important
        sitk_mask.SetOrigin(img.GetOrigin())
        #TODO:if needed add below, ask Justus/Daniel
        # for key in img.GetMetaDataKeys():
        #     sitk_mask.SetMetaData(key, img.GetMetaData(key))
        features = extractor.execute(img, sitk_mask, label=label)
        # features1 = extractor.execute(img, sitk_mask, label=1)
        self.update(features, dir_dicom, label)
    
    def update(self, features, dir_dicom, label):
        item_dict = {"Location":str(dir_dicom)}
        item_dict.update(features)
        if self.sdicts[label-1] is None:
            self.sdicts[label-1] = pd.Series(item_dict)
        else:
            if len(self.sdicts[label-1].shape) == 2:
                if not self.sdicts[label-1].shape[1]%100 and self.sdicts[label%2] is not None:#check if other sheet is not None
                    self.save()
                print(self.sdicts[label-1].shape[1])
            self.sdicts[label-1] = pd.concat([self.sdicts[label-1],pd.Series(item_dict)], axis=1)
    def save(self):
        if self.sdicts[0] is None or self.sdicts[1] is None:
            raise NotImplementedError("update with feature values first")
        excel_svpath = os.path.join(self.save_dir, f'features.xlsx')
        dict0 = self.sdicts[0].transpose()
        dict1 = self.sdicts[1].transpose()
        with pd.ExcelWriter(excel_svpath) as writer:
            dict1.to_excel(writer, sheet_name="label 2(Lateral Menisci)")
            dict0.to_excel(writer, sheet_name="label 1(Medial Menisci)")

def get_dess_dir():
    """get dir path of all DESS images

    Returns
    -------
    list
        list of dirs
    """
    json_path = r"/images/Shape/Medical/Knees/OAI/Full/metadata_sorted_by_value.json"
    with open(json_path) as json_file:
        data = json.load(json_file)
    dir_list = []
    for path in data["ProtocolName"]["SAG 3D DESS WE"]:
        if "Baseline" in path:
            dir_list.append(path)
    return dir_list

def initi():
    #dictionary for storing image properties
    list_propr = {"location":[],"shape":[],"mean mid volslice":[], "plane":[], "patient position":[],
    "pixel spacing": [], "slice thickness": [], "rep time": [], "flip angle": [], "pixel bandwidth": [], "echo time":[],
    "image type": []}
    return list_propr
def finis(list_propr, save_dir):
    """save excel with file information

    Parameters
    ----------
    list_propr : list
        list of properties
    save_dir : str
        save dir
    """
    di = pd.DataFrame.from_dict(list_propr)
    excel_svpath = os.path.join(save_dir, f'baselinefoldertypes.xlsx')
    with pd.ExcelWriter(excel_svpath) as writer:
        di.to_excel(writer)

def get_img_orient(dir, tags=[0, 1, 2, 3, 4, 5, 6, 7, 8]):
    """get values of selected dicom tags
    [0, 1, 2, 3, 4, 5...]
    0 image orientation patient 0020|0037
    1 plane
    2 PixelSpacing attribute 0028, 0030, 
    3 SliceThickness 0018, 0050, 
    4 Repetition Time 0018,0080, 
    5 Flip Angle 0018,1314 
    6 Pixel Bandwidth 0018,0095 
    7 echo time 0018,0081
    8 image type 0008,0008 
    Parameters
    ----------
    dir : str
        dicom folder

    Returns
    -------
    list
        list of requested parameters
    """
    ret_list = []#return list
    reader = sitk.ImageFileReader()
    try:
        reader.SetFileName(os.path.join(dir, os.listdir(dir)[2]))
        reader.LoadPrivateTagsOn()

        reader.ReadImageInformation()
        if 0 in tags:
            patient_pos = reader.GetMetaData('0020|0037')#image orientation patient
            patient_pos = [np.float(x) for x in patient_pos.split("\\")]
            ret_list.append(patient_pos)
            if 1 in tags:
                plane = file_plane(patient_pos)
                ret_list.append(plane)
        
        metatag = ["0028|0030","0018|0050","0018|0080","0018|1314","0018|0095", "0018|0081", "0008|0008"]
        for ind, poss_tags in enumerate(range(2, len(metatag)+2)):#possible tags
            if poss_tags in tags:
                attri = reader.GetMetaData(metatag[ind])#PixelSpacing
                ret_list.append(attri)
    except Exception as e:
        ret_list = ["Not definded" for _ in range(len(tags))]
    return ret_list

def file_plane(IOP):
    """get image plane

    Parameters
    ----------
    IOP : list
        dicom value of 'image orientation patient'

    Returns
    -------
    str
        type of plane
    """
    IOP_round = [round(x) for x in IOP]
    plane = np.cross(IOP_round[0:3], IOP_round[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return "Sagittal"
    elif plane[1] == 1:
        return "Coronal"
    elif plane[2] == 1:
        return "Axial"

def create_sfolders(dir):
    # create 3 folders for storing images, both menisci mask wrong, one right, both right
    names = ["wrong_both", "one_pred", "both_pred"]
    dir_list = []
    for ele in names:
        save_fig_dir = os.path.join(dir, ele)
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)
        dir_list.append(save_fig_dir)
    return dir_list

def check_baseline(img, list_propr, dir, save_dir, plane, patient_pos, sitk_img):
    """check baseline iamge attributes and add to list, return transformed sitk image

    Parameters
    ----------
    img : `numpy.array`
        image array in numpy form
    list_propr : list
        list of attributes
    dir : str
        location of dicom
    save_dir : str
        save dir
    plane : str
        `Coronal|Sagittal|Axial`
    patient_pos : list
        patient position
    sitk_img : Simple ITK Image
        simple itk image

    Returns
    -------
    Simple ITK Image
        simple itk image transformed
    """
    #filepath, shape of numpy, average of valuesof middle volume slice, save fig
    list_propr["location"].append(dir)
    list_propr["shape"].append(img.shape)
    list_propr["mean mid volslice"].append(np.mean(img[img.shape[0]//2,...]))
    list_propr["plane"].append(plane)
    list_propr["patient position"].append(patient_pos)
    attr_keys = ["pixel spacing" , "slice thickness" , "rep time" , "flip angle" , "pixel bandwidth" , "echo time", "image type"]
    img_attr = get_img_orient(dir, tags=[2, 3, 4, 5, 6, 7, 8])#image attributes
    for att, val in zip(attr_keys, img_attr):
        list_propr[att].append(val)
    #display numpy image
    # plt.imshow(img[img.shape[0]//2,...])
    # plt.savefig(save_dir+str(len(list_propr["location"]))+f"_{plane}.png")

    if plane=="Axial" or plane=="Sagittal":
        #display numpy img got from sitk image
        # image_np = sitk.GetArrayFromImage(sitk_img)
        # save_fig_nps(image_np, transform_np(image_np, plane), save_dir, str(len(list_propr["location"])))
        change_shape = {"Coronal":[512, 80, 512], "Sagittal":[512, 512, 80], "Axial":[512, 80, 512]}#sagittal, axial working
        # change_shape = [512, 80, 512] if plane=="Axial" else [512, 512, 80]
        sitk_img = transform_sitk(sitk_img, change_shape[plane])#permute the dimensions also..sagittal to frontal(but horizontal)
        change_dim = [0, 2, 1] if plane=="Axial" else [1, 0, 2]#making horizontal frontal to normal frontal
        sitk_img = sitk.PermuteAxes(sitk_img, change_dim)
        image_np = sitk.GetArrayFromImage(sitk_img)
    #     plt.imshow(image_np[image_np.shape[0]//2,...])
    #     plt.savefig(save_dir+str(len(list_propr["location"]))+"_trans.png")
    # plt.close("all")
    return sitk_img

def transform_np(img, plane):
    #transform axial and sagittal images to frontal views for given numpy
    plane_trans = {"Sagittal":[2, 1, 0], "Axial": [1, 0, 2]}
    np_img = np.transpose(img, plane_trans[plane])
    return np_img

def save_fig_nps(image_np, trans_np, save_dir, img_no):
    plt.imshow(image_np[image_np.shape[0]//2,...])
    plt.savefig(save_dir+img_no+"_.png")
    plt.imshow(trans_np[trans_np.shape[0]//2,...])
    plt.savefig(save_dir+img_no+"_trans.png")
    plt.close("all")

def count_dicoms(dir):
    #count only the dicom folders in a given directory
    count=0
    for walk_items in os.walk(dir):
        dir = walk_items[0]
        print("checking folder", count, dir)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dir)
        if len(dicom_names):
            reader.SetFileNames(dicom_names)
            try:
                image = reader.Execute()
                count+=1
            except RuntimeError as e:
                pass
    return count

class BothGood():
# counts of folders, where both central menisci masks are right, if one is right or nothing
    def __init__(self):
        self.is_good = [False, False]
    def update(self, label, is_good, counts_total=None):
        #label- 0:medial, 1:lateral
        self.is_good[label]= is_good
        if label==1 and counts_total is not None:#both passes done
            num_cor_masks = self.how_good()
            counts_total[num_cor_masks] += 1
            if num_cor_masks==1:
                 which_menisci = -2 if self.is_good[0] else -1#only medial correct
                 counts_total[which_menisci]+=1
        return counts_total
    def how_good(self):
        if all(self.is_good):
            return 2
        if any(self.is_good):
            return 1
        return 0
def read_dicom_folders(root_dir, kwargs, model1, model_loc, model_loc2, model2=None, excel_rad=None, 
device=None, cmfind_mloc=None, savecm_dir=None, savecm_count=0, save_dess=False, find_mask=True, save_mask_count=0, check_folders=30,
wrong_mask_cnt=0, find_mid_slice=True, model_side_path=None, modelside=None):
    """for middle slice pred, make sure image resolution(downscale) is given correctly, and apply aug to traiining data to resemble like OAI 
       predict middle slice, get their menisci masks, find side of lateral meniscus, generat radiometrics

    Parameters
    ----------
    root_dir : str
        root directory containing all dicom folders
    kwargs : list
        parameters used for training
    model1 : `torch.module`
        model
    model_loc : str
        model location
    model_loc2 : str
        model location
    model2 : `torch.module`, optional
        model, by default None
    excel_rad : `excel_radfeat`, optional
        for storing radiometrics, by default None
    device : `torch.device`, optional
        device, by default None
    cmfind_mloc : str, optional
        find middle slice model location, by default None
    savecm_dir : str, optional
        location for saving results, by default None
    savecm_count : int, optional
        how many results from central slice prediciton, by default 0
    save_dess : bool, optional
        whether to save original DESS image, by default False
    find_mask : bool, optional
        whether to find mask, by default True
    save_mask_count : int, optional
        how many results to save from mask prediction, by default 0
    check_folders : int, optional
        number of folders to inspect, by default 30
    wrong_mask_cnt : int, optional
        number of wrong prediction to be saved, by default 0
    find_mid_slice : bool, optional
        whether to find middle slice, by default True
    model_side_path : str, optional
        location of model, finding lateral menisci side , by default None
    modelside : `torch.module`, optional
        model finding side of lateral menisci, by default None

    Returns
    -------
    list
        count of wrong, partially correct, fully correct masks predicted
    """
    saveimg_cnt = 0#counting each img vol saved with pred middle slice num
    savemask_curcnt = 0#counting each mask saved
    folders_checked = 0#dicom folders
    knee_img_cnt = -1
    counts_good = [0,0,0,0,0]#if both  are bad, if one good, if both good,(2 masks per knee),among only 1 mask correct-only medial corr, only lateral corr
    if save_mask_count or wrong_mask_cnt:
        sfolders = create_sfolders(er.save_dir)#save image folders
    if model2 is None:
        model2=model1
    sft = torch.nn.Softmax2d()

    dess_path = r"/images/Shape/Medical/Knees/OAI/SAG_3D_DESS/OAIBaselineImages"#r"/images/Shape/Medical/Knees/OAI/Full/OAIBaselineImages"
    folders_dess_baseline = os.listdir(dess_path)
    list_propr = initi()

    for dirname in folders_dess_baseline:
        dir = os.path.join(dess_path, dirname)
        print("checking folder", folders_checked, dir)
        if not folders_checked%10:
            finis(list_propr, savecm_dir) 
        if knee_img_cnt==check_folders:#folders_checked>check_folders:   #and save_mask_count==savemask_cnt, saveimg_cnt==savecm_count
            return folders_checked, knee_img_cnt, counts_good
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(dir)
        if len(dicom_names):
            reader.SetFileNames(dicom_names)
            try:
                image = reader.Execute()
            except RuntimeError as e:
                print(e)
                print(dir)
                continue
            np_img_orig = sitk.GetArrayFromImage(image)
            folders_checked += 1
            np_img = np.asarray(np_img_orig, np.float32)
            #reason for transpose /home/students/thampi/PycharmProjects/MA_Praise/data_read/transpose_why.jpeg
            # np_img = np.transpose(np_img, (2, 1, 0))
            b, t, h = np_img.shape
            patient_pos, plane, slice_thick, img_type = get_img_orient(dir, [0,1,3,8])
            #if width not equal to height, not probably a knee image, if b<25: probably not MRI scan(but x-ray etc)
            if find_mid_slice and b>25 and t==h and float(slice_thick)<=0.7:# and b<=200:#==160:# and h==512 and b==512:#b>160for filtering sagitall DESS images
                knee_img_cnt += 1
                heading = plane + " " + slice_thick + " " + img_type 
                trans_sitk_img = check_baseline(np_img_orig, list_propr, dir, savecm_dir, plane, patient_pos, image)
                if plane!="Coronal":
                    np_img = transform_np(np_img, plane)
                mdata= MeniscusDataFromArray(np_img, in_channels=3, with_ind=True)#,if_crop=True)
                batch_size=5
                mdata_loader = DataLoader(mdata, batch_size=batch_size, **kwargs)
                saved_model = get_saved_model(model1, model_loc, with_edge=False)
                saved_model.eval()
                # for finding middle slice
                cmfind_model = DetectMid(if_regr=True, out_classes=2)
                ds3d = OAIData3DFromArray(np_img,scale_size=[80,512,512],downscale=0.25)#dataset
                # cmmodelloc = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/train_midlsl_good"
                findcmtr = get_saved_model(cmfind_model, cmfind_mloc, with_edge=False)
                findcmtr.eval()
                img3dt = ds3d[0].unsqueeze(dim=0)#img 3d tensor dataset
                sl = findcmtr(img3dt)
                if sl.shape[-1]==80:
                    sl = torch.argmax(sl)
                sl = sl.type(torch.int)
                #if to exclude training set images
                # if len(list(set(dir.split("/")).intersection(["9000798", "9001400","9001695", "9000296","9000798","9001695","9001897","9002316","9002411","9002817","9003316","9003380","9003406","9003815","9004184","9004315","9004462","9004905","9005321","9005413","9005905","9006407","9006723","9007422","9007827","9007904","9008322"]))):
                #     continue
                if saveimg_cnt<savecm_count:
                    dir_info = dir.split("/Full/")[1].replace("/","_")
                    save_img_folder = os.path.join(savecm_dir, dir_info)
                        # img3d = img3dt.clone().detach().cpu().squeeze().numpy()
                        # input3d = r"/home/students/thampi/PycharmProjects/meniscus_data/3d_inputs_all_slices_whole"
                        # npy_file_name = dir.split("Knees/")[1].replace("/", "_")+"_midslm_42_midsll_43.npy"
                        # npy_file_path = os.path.join(input3d, npy_file_name)
                        # with open(npy_file_path, 'wb') as f:
                        #     np.save(f, img3d)
                    if not os.path.exists(save_img_folder):
                        os.makedirs(save_img_folder)
                    #generating higher resoluton img just for saving
                    ds3d512 = OAIData3DFromArray(np_img,scale_size=[80,512,512],downscale=0.5)[0].unsqueeze(dim=0)
                    save_3d_psl2(ds3d512, sl, save_img_folder, savecm_count, saveimg_cnt, foldername="newimg")#to_80_512_512_")#savecm_count total num imgs to save
                    if save_dess:
                        save_3dimg(np_img_orig,save_img_folder, foldername="DESS_Img_160_384_384")#img_ind="dessimg_160_384_384"+str(saveimg_cnt))
                    saveimg_cnt += 1

                modelsidep = get_saved_model(modelside, model_side_path, with_edge=False)
                modelsidep.eval()
                saved_model2 = get_saved_model(model2, model_loc2, with_edge=False)
                saved_model2.eval()

            #         # for batch in fdata_loader:

            #         # batch_lr = get_imgslr(batch, torch.from_numpy(pred_segm))
                postpr='crf'
                findg = BothGood()#findgood
                if find_mask:
                    for eles in enumerate(sl.view(-1).tolist()):
                        sl_ind, slice_no = eles
                        #sl_ind - 0 for medial, 1 for lateral
                        img_dim = 512
                        ds3dhd = OAIData3DFromArray(np_img,scale_size=[80,img_dim,img_dim],downscale=1)[0].unsqueeze(dim=0)
                        batch = ds3dhd[0][...,slice_no,:,:].unsqueeze(dim=-3).expand([-1,1,img_dim,img_dim])#if in chan=1
                    #         batch = batch[valid_ind]#including only "valid" images
                    #         img_ind_valid = ind_img[valid_ind]
                        #normalizing as done for training images of model
                        mean = torch.mean(batch)
                        std = torch.std(batch)
                        tnormalize = tvn.Normalize(mean, std)
                        batch = tnormalize(batch)#torch normalize
                        plat_side = modelsidep(batch)#predict lateral side
                        if postpr=='tt':
                            pred = aug_imgs(saved_model2, batch, chooseaugs=[2, 3])
                            pred = sft(pred).detach()
                            pred = pred.cpu().numpy()
                        elif postpr=='crf':
                            pred = saved_model2.to(device)(batch.to(device))
                            pred = sft(pred).detach().cpu()
                            pred = list(map(dense_crf,batch, pred))
                            pred = np.asarray(pred)
                        else:
                            pred = saved_model2(batch)
                            pred = sft(pred).detach().cpu().numpy()

                        pred_amx = np.argmax(pred, axis=1)
                        pred_segm = choose_quadrant(plat_side.cpu().detach().numpy(), pred_amx, sl_ind)

                        input_img = batch[0][0].detach().cpu().numpy()
                        slice_loc = dir+"/midslice_"+str(slice_no)
                        pred_segm = get_largest_island(pred_segm)
                        pred_segm[pred_segm==1] = sl_ind + 1
                        psegm = pred_segm.squeeze()#predicted segm
                        mask_corr =  (np.max(label(pred_segm > 0)) == 1 and np.max(label(pred_segm)) == 1 and np.unique(pred_segm, return_counts=True)[1][1]>30)#3 clusters separated apart, skipped wrongly predicted masks
                        counts_good = findg.update(sl_ind, mask_corr, counts_good)
                        if not sl_ind:
                            medial_items = [input_img, psegm, mask_corr]# store medial central img, mask, if mask correct
                        savemask_curcnt, wrong_mask_cnt = save_both(findg, sl_ind, sfolders, medial_items, [input_img, psegm, mask_corr], wrong_mask_cnt, savemask_curcnt, knee_img_cnt)
                        if not mask_corr:
                            continue
                        #TODO save image, location of masks with np.unique(pred_segm, return_counts=True)[1][1]<350
                        #transforming sitk image to use as reference for dicoms created later #TODO correct frontal ones
                        if plane!="Coronal":
                            image = trans_sitk_img
                        else:
                            image = sitk.PermuteAxes(image,[2,1,0])
                            image = transform_sitk(image, [512, 80, 512])
                            image = sitk.PermuteAxes(image,[0,2,1])
                            slice_no=2
                        slice = sitk.SliceImageFilter()
                        slice.SetStart((slice_no, 0, 0))#random slice from original DESS volume
                        slice.SetStop((slice_no+1, image.GetSize()[0], image.GetSize()[1]))
                        slice.SetStep(1)
                        sliced_filter = slice.Execute(image)

                        sitk_input = np_to_dicom(np.expand_dims(input_img, axis=0).copy(), sliced_filter)
                        excel_rad.gt_upd_radfeat_sitk(sitk_input,pred_segm, slice_loc, label=sl_ind+1)

            print(dir, sitk.GetArrayFromImage(image).shape)
def display_imgs(items, im_th, save_dir, knee_img_cnt=None, heading=None):
    """save plots

    Parameters
    ----------
    items : list
        image, mask, location, label
    im_th : int
        count of this image in whole list
    save_dir : str
        save location
    knee_img_cnt : int, optional
        `items`th turn in the whole knee image set, by default None, by default None
    heading : str, optional
        figure heading, by default None

    Returns
    -------
    matplotlib figure
        figure
    """
    mask_per_page = 1
    no_r = mask_per_page
    no_col  = 2
    title_men = ["MEDIAL", "LATERAL"]
    img, seg, slice_loc, men_label = items
    fig, ax = plt.subplots(no_r, no_col)
    [axi.set_axis_off() for axi in ax.ravel()]
    fig.set_size_inches(18, 10)  # width, height
    fig.suptitle(f"Knee image with central {title_men[men_label-1]} meniscus {heading}", fontsize=20)
    row_ind = 0#im_th%mask_per_page#, col = np.unravel_index(im_th, [no_r, no_r])
    col_ind = 0#col*2
    curr_ax = ax
    curr_ax[col_ind].imshow(img, cmap='gray') 
    curr_ax[col_ind].title.set_text("image")
    # ax[row_ind][col_ind].axis("off")
    sg_mask = np.ma.masked_array(seg, seg != men_label)
    curr_ax[col_ind+1].imshow(img, cmap='gray')
    curr_ax[col_ind+1].imshow(sg_mask, cmap="spring")
    curr_ax[col_ind+1].title.set_text("with mask")
    # ax[row_ind][col_ind+1].axis("off")
    fig.savefig(os.path.join(save_dir,"img"+str(im_th)+f"_pred_mask_{knee_img_cnt}.png"))
    plt.close("all")
    return fig

if __name__=="__main__":
    excel_svdir=r"/home/students/thampi/PycharmProjects/MA_Praise/extract_feat"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    htc_gpu = torch.cuda.device_count() if device.type=='cuda' else 0
    kwargs = {'num_workers': 4*htc_gpu, 'pin_memory': True} if device == 'cuda' else {}
    root_dir = r"/images/Shape/Medical/Knees/OAI/Full"

    # model1 = UNetSimple(in_classes=3)#UNetNaiveMultiOut(in_classes=3)#
    # model2 = UNetSimple(in_classes=3)#, channelscale=64)
    # model= SpiderNet()

    model_loc = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/high_data_aug_segm_model"
    #bl_manual_add_sglab_jc_segm"#tetra_augh_sglab_jc_segmcopy"#bl_manual_add_sglab_jc_segm"#"#blsegmbrightaughigh"#singlelabelsegmcopy"

    model_side_path = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/find_side_segm_512"#for finding side of lateral-0 for left, 1 on right
    model_side = DetectSide(in_classes=1, if_regr=True, out_classes=2)
    cmfind_mloc =r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/train_mid_dess_slices"

    sv_dr = r"/home/students/thampi/PycharmProjects/MA_Praise/outputs/pyrad"
    er = excel_radfeat(sv_dr)
    savecm_count = 1
    save_mask_count = 8
    check_folders = 3000
    #to generate radiomic features or to save pred middle slice
    if savecm_count:
        for filename in os.listdir(sv_dr):
            file_path = os.path.join(sv_dr, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        save_img_folder = os.path.join(sv_dr, "saved_img")
        if not os.path.exists(save_img_folder):
            os.makedirs(save_img_folder)

        folders_checked, knee_img_cnt, counts_good = read_dicom_folders(root_dir, kwargs, UNetSimple(in_classes=1, channelscale=128, out_classes=2), model_loc, model_loc, UNetSimple(in_classes=1, channelscale=128, out_classes=2), device=device, excel_rad=er, cmfind_mloc=cmfind_mloc, savecm_dir=save_img_folder, savecm_count=0, find_mask=True, save_mask_count=20, check_folders=60, find_mid_slice=True, model_side_path = model_side_path, modelside=model_side)
        er.save()
        print(folders_checked, knee_img_cnt, counts_good)



