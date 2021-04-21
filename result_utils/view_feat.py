import os
import matplotlib.pyplot as plt

def plt_feat(x, save_loc=None, nrows=5, ncols=None):
    """to plot tensors/array_like object

    Parameters
    ----------
    x : array_like
        outputs or intermediate outputs in model
    save_loc : str, optional
        save location, by default None
    nrows : int, optional
        number of rows, by default 5
    ncols : int, optional
        number of columns, by default None
    """
    arr=x
    if x.is_cuda:
        x = x.cpu()
    if ncols is None:
        ncols=nrows
    if len(arr.shape)==4:
        arr=arr[0]
    fig, ax = plt.subplots(nrows, ncols)
    img_num=0

    for i in range(nrows):
        for j in range(ncols):
            ax[i][j].axis("off")
            ax[i][j].imshow(arr[img_num])
            img_num+=1
            if img_num==arr.shape[0]:
                break
        else:
            continue
        break
    if save_loc:
        fig.savefig(os.path.join(save_loc, 'view_tens.png'))
        plt.close('all')
