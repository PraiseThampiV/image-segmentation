from torch import nn
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils 

MAX_ITER = 5
POS_W = 1
POS_XY_STD = 50#with 5 same performance
Bi_W = 1
Bi_XY_STD = 1
Bi_RGB_STD = 3
# adapted from https://github.com/kunalmessi10/FCN-with-CRF-post-processing/blob/master/crf.py
#https://airccj.org/CSCP/vol8/csit89703.pdf
def dense_crf(img, output_probs):
    """apply crf

    Parameters
    ----------
    img : `torch.tensor`
        input
    output_probs : `torch.tensor`
        logits

    Returns
    -------
    `numpy.array`
        refined output
    """
    img=img*255/img.max()
    img=img.detach().numpy().astype(np.uint8)
    # img = img.transpose(1, 2, 0)
    output_probs = output_probs.detach().numpy()
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    #in memory contiguous array
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img.transpose(2, 1,0))
    # Image data format should be an unsigned integer 
    d = dcrf.DenseCRF2D(w, h, c)#c=nlabels  w-width h-height
    d.setUnaryEnergy(U)
    # compat is tstrength of this potential
    d.addPairwiseGaussian(
        # sxy=50, compat=2)#no idfference even after increasing sxy, but with increasing compat, masks is erased fully
        #if compat is a number, if its potts compatibility, 1 for i not equal to j, otherwise 0 
        sxy=POS_XY_STD, compat=POS_W)
    #Eg: outputprobs/feat of shape (5, 480, 640) and im.shape == (640,480,3)
    d.addPairwiseBilateral(
        # sxy=3, compat=5,#with increasing sxy(>5), mask getting erased
        sxy=Bi_XY_STD, compat=Bi_W,
        srgb=Bi_RGB_STD,
        # srgb=50, #no effect with increasing srgb
        # rgbim=img)
        rgbim=np.repeat(img, 3, axis=2) if img.shape[-1]==1 else img)
        

    #`s{dims,chan}` parameters are model hyper-parameters define strength of the location and image content bilaterals, respectively
    # pairwise_energy = utils.create_pairwise_bilateral(sdims=(10,10), schan=Bi_RGB_STD, img=img.transpose(0,1,2), chdim=0)
    # d.addPairwiseEnergy(pairwise_energy, compat=10)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    # Qn=np.argmax(Q, axis=0)
    # plt.imshow(Qn)
    # plt.savefig('out')
    return Q
