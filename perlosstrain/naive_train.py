import torch
import os, shutil
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tqdm.notebook import tqdm
from tqdm import tqdm, notebook
from functools import reduce
from torch.autograd import Variable
from torch import optim
from torch import nn
from loss.edge_loss import get_edge_img
from loss.dice import dice_loss_sq
from loss.msssim import msssim

class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, is_best, save_model_dir, del_pre_model=True):
    """    https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Parameters
    ----------
    model : torch model
        [description]
    is_best : bool
        whether current model better
    save_model_dir : path like
        path directory
    del_pre_model : bool, optional
        whether to delete previous model, by default True
    """
    filename=os.path.join(save_model_dir, f'checkpoint_epoch{str(model["epoch"])}_{model["arch"]}.pth.tar')
    if is_best:
        prev_model_state = [name for name in os.listdir(save_model_dir) if "epoch" in name]
        if del_pre_model:
            # previous model names
            for prev_modname in prev_model_state:
                os.unlink(os.path.join(save_model_dir, prev_modname))
        torch.save(model, filename)

def to_np(x):
    """convert to numpy

    Parameters
    ----------
    x : tensor
        tensor to be converted to numpy

    Returns
    -------
    numpy
        converted array
    """
    return x.data.cpu().numpy()

def prep_trainedge(segm_act, model, train_edge=True, epoch=100):
    # making act as hot encoded vector
    act_unsq = segm_act.view(segm_act.shape[0],1,segm_act.shape[2], segm_act.shape[2])
    sgm_zer=(torch.zeros(act_unsq.shape[0],3,*act_unsq.shape[2:]))
    sgm_zer = sgm_zer.to(act_unsq.device)
    in_net = sgm_zer.scatter(1, act_unsq, 1)
    if train_edge:# and epoch>4:
        noise = torch.randn_like(in_net)
        in_net = in_net + noise
        sft = nn.Softmax2d()
        in_net = sft(in_net)
    outputs = model(in_net)
    return outputs

def mask_train(model, trained_edgemodel, images, segm_act, sem_weights=None):
    sem_weights = [0, 1, 1, 1, 1, 1, 0]
    # predicted psis
    pred_segm, ppsi = model(images)
    # actual psis
    apsi = prep_trainedge(segm_act, trained_edgemodel.cuda(), False)
    mse = torch.nn.MSELoss()
    sft = nn.Softmax2d()
    edge_act = torch.stack(list(map(get_edge_img, segm_act)))

    if apsi[-1].shape[1]==2:
        sem_weights = [0, 1, 1, 1, 1, 1, 0]
    else:
        nm = len(segm_act.nonzero())
        nb = len(edge_act.nonzero())
        sem_weights1 = [nb/(nb+nm) for i in range(int(len(apsi)/2))]
        sem_weights2 = [nm/(nb+nm) for i in range(int(len(apsi)/2))]
        sem_weights = sem_weights1 + sem_weights2

    # d_loss = dice_loss_sq(ppsi4, edge_act)
    # layer output comparison
    layer_comp = list(map(lambda weight, x, y: weight*mse(x, y),sem_weights, ppsi, apsi))
    semeda_loss = sum(layer_comp)
    # [0]*mse(ppsi[i], apsi[i]) for i in range(len(ppsi)-1)
    # + sem_weights[2]*mse(ppsi3, apsi[2]) 
    # + sem_weights[3]*mse(sft(ppsi3), sft(apsi[2]))
    return pred_segm, semeda_loss#+d_loss   

def finalssim(final, segm_act):
    """"final" output and actual segm compare

    Parameters
    ----------
    final : tensor
      actual masks

    Returns
    -------
    scalar tensor
        loss
    """
    if len(segm_act.unique())!=3:
        return 0
    # compare conv4 to be img patch around mensci and conv5 as individual meniscus(TODO:do it accurately for labels, hor flipped case)
    wid_max = (segm_act!=0).nonzero(as_tuple=True)[-1].max()
    wid_min = (segm_act!=0).nonzero(as_tuple=True)[-1].min()
    h_max = (segm_act!=0).nonzero(as_tuple=True)[-2].max()
    h_min = (segm_act!=0).nonzero(as_tuple=True)[-2].min()
    mid_h = (h_max+h_min)//2
    wid_half = (wid_max-wid_min)//2
    margin=5

    sgm = segm_act.unsqueeze(dim=1).float().clone()
    actbg = sgm[..., mid_h-wid_half-margin: mid_h+wid_half+margin, wid_min-margin:wid_max+margin].clone()
    actbg[actbg!=0]=1

    #lateral to left and medial to right
    actm = sgm[..., mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min+wid_half-margin:wid_max+margin].clone() 
    actl = sgm[..., mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min-margin:wid_min+wid_half+margin].clone()
    predm =  final[...,1, mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min+wid_half-margin:wid_max+margin].unsqueeze(dim=1)
    predl =  final[...,2, mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min-margin:wid_min+wid_half+margin].unsqueeze(dim=1)
    if actl.max()!=2:
        actl, actm = actm.clone(), actl.clone()
        predl, predm = predm.clone(), predl.clone()
    actl[actl==2]=1

    predbg = final[:,0, mid_h-wid_half-margin: mid_h+wid_half+margin, wid_min-margin:wid_max+margin].unsqueeze(dim=1)

    mse = nn.MSELoss()
    return  0

def get_edg(tens, if_inst=False):
    """get edge pixels

    Parameters
    ----------
    tens : tensor
        array whose edges are to be found
    if_inst : bool, optional
        if instance norm to be applied, by default False

    Returns
    -------
    tensor
        with edge locations emphasized
    """
    sobely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    #depth = x.size()[1]
    if len(tens.shape)==3:
        tens = tens.float().unsqueeze(dim=1)
    channels = tens.size()[1]

    sobel_kernely = torch.tensor(sobely, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
    sobel_kernelx = torch.tensor(sobelx, dtype=torch.float32).unsqueeze(0).expand(1, channels, 3, 3)
    edgex = F.conv2d(tens, sobel_kernelx.to(tens.device), stride=1, padding=1)#, groups=inter_x.size(1))
    edgey = F.conv2d(tens, sobel_kernely.to(tens.device), stride=1, padding=1)
    # all non-zero value locations are part of boundary
    edge = edgex+edgey
    #test with other norm also
    inst = nn.InstanceNorm2d(edge.shape[1])
    rl = nn.ReLU(inplace=True)
    if if_inst:
        edge = inst(edge)
        edge = rl(edge)
    # edge = edge/edge.max()
    return edge
def loss_interxedge(images, segm_act, inter_x):
    """to make sure that conv3 gives an output as the image patch around region to be segmented

    Parameters
    ----------
    images : tensor
        input images to network
    segm_act : tensor
        actual segmentation masks
    inter_x : tensor
        intermediate output

    Returns
    -------
    tensor
        loss
    """
    if len(segm_act.unique())!=3:
        return 0

    #getting focussed region from actual image
    #TODO try making segmact in interx shape
    adaptm=nn.AdaptiveAvgPool2d([segm_act.shape[-2], segm_act.shape[-1]])
    inter_x = adaptm(inter_x)
    # adaptm=nn.AdaptiveAvgPool2d([inter_x.shape[-2], inter_x.shape[-1]])
    # segm_act = adaptm(segm_act.unsqueeze(dim=1).float())
    #TODO try with dice loss also
    mse = nn.MSELoss()


    pred = get_edg(inter_x, if_inst=True)
    target = get_edg(segm_act)

    t=torch.zeros_like(target)
    t[target!=0]=1
    # pred = F.softmax(pred.reshape(pred.size(0), pred.size(1), -1), 2).view_as(pred)
    # pred=pred-pred.min()
    # if pred.max():
    #  pred = pred/pred.max()
    loss =mse(pred, t)#dice_loss_sq(pred, t, no_sft=True)
    # loss =(pred*t) 
    # t = t*(pred*t).max()
    # loss = loss/(2*t.sum())
    # loss = 1-torch.sum(loss)
    return loss

def loss_conv3(images, segm_act, conv3):
    """to make sure that conv3 gives an output which the image patch around region to be segmented

    Parameters
    ----------
    images : [type]
        [description]
    segm_act : [type]
        [description]
    inter_x : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if len(segm_act.unique())!=3:
        return 0
    # compare conv4 to be img patch around mensci and conv5 as individual meniscus(TODO:do it accurately for labels, hor flipped case)
    wid_max = (segm_act!=0).nonzero(as_tuple=True)[-1].max()
    wid_min = (segm_act!=0).nonzero(as_tuple=True)[-1].min()
    h_max = (segm_act!=0).nonzero(as_tuple=True)[-2].max()
    h_min = (segm_act!=0).nonzero(as_tuple=True)[-2].min()
    mid_h = (h_max+h_min)//2
    wid_half = (wid_max-wid_min)//2
    margin=5
    img_patch = images[:,:, mid_h-wid_half-margin: mid_h+wid_half+margin, wid_min-margin:wid_max+margin].clone()

    #getting left and right halves: TODO try with image patch left and right halves
    #lateral to left and medial to right



    #getting focussed region from actual image
    adaptm=nn.AdaptiveAvgPool2d([conv3.shape[-2], conv3.shape[-1]])
    resized_patch = adaptm(img_patch)
    mse = nn.MSELoss()
    img_input = resized_patch[:,0,:,:].unsqueeze(dim=1)
    conv3_partial = conv3[:,:conv3.shape[1]//2,:,:].clone()
    act_img = img_input.expand_as(conv3_partial)
    act_img = (act_img-act_img.min())/act_img.max()
    pred_img = (conv3_partial-conv3_partial.min())/conv3_partial.max()
    loss = mse(pred_img, act_img)
    return loss

def loss_conv5_2in1(images, segm_act, conv5):
    """computing loss from intermediate result, geting just one channel output of intermediate result

    Parameters
    ----------
    images : [type]
        [description]
    segm_act : [type]
        [description]
    inter_x : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if len(segm_act.unique())!=3:
        return 0
    # compare conv4 to be img patch around mensci and conv5 as individual meniscus(TODO:do it accurately for labels, hor flipped case)
    wid_max = (segm_act!=0).nonzero(as_tuple=True)[-1].max()
    wid_min = (segm_act!=0).nonzero(as_tuple=True)[-1].min()
    h_max = (segm_act!=0).nonzero(as_tuple=True)[-2].max()
    h_min = (segm_act!=0).nonzero(as_tuple=True)[-2].min()
    mid_h = (h_max+h_min)//2
    wid_half = (wid_max-wid_min)//2
    margin=5
    #getting left and right halves: TODO try with image patch left and right halves
    #lateral to left and medial to right

    img_patchl = segm_act[..., mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min-margin:wid_min+wid_half+margin] 
    img_patchr = segm_act[..., mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min+wid_half-margin:wid_max+margin] 
    if img_patchl.max()!=2:
        img_patchl, img_patchr = img_patchr.clone(), img_patchl.clone()

    adaptm=nn.AdaptiveAvgPool2d([conv5.shape[-2], conv5.shape[-1]])
    conv5_input0 = conv5[:,:conv5.shape[1]//2,:,:]
    conv5_input1 = conv5[:,conv5.shape[1]//2:,:,:]
    patchl = adaptm(img_patchl.float().unsqueeze(dim=1)).expand_as(conv5_input0)
    patchr = adaptm(img_patchr.float().unsqueeze(dim=1)).expand_as(conv5_input1)
    mse = nn.MSELoss()
    loss0_conv5 = mse(conv5_input0/conv5_input0.max(), patchl/patchl.max())
    loss1_conv5 = mse(conv5_input1/conv5_input1.max(), patchr/patchr.max())
    return loss0_conv5+loss1_conv5

def loss_interx(images, segm_act, inter_x):
    """computing loss from intermediate result, geting just one channel output of intermediate result

    Parameters
    ----------
    images : [type]
        [description]
    segm_act : [type]
        [description]
    inter_x : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if len(segm_act.unique())!=3:
        return 0
    conv4, conv5 = inter_x
    # compare conv4 to be img patch around mensci and conv5 as individual meniscus(TODO:do it accurately for labels, hor flipped case)
    wid_max = (segm_act!=0).nonzero(as_tuple=True)[-1].max()
    wid_min = (segm_act!=0).nonzero(as_tuple=True)[-1].min()
    h_max = (segm_act!=0).nonzero(as_tuple=True)[-2].max()
    h_min = (segm_act!=0).nonzero(as_tuple=True)[-2].min()
    mid_h = (h_max+h_min)//2
    wid_half = (wid_max-wid_min)//2
    margin=5
    img_patch = images[:,:, mid_h-wid_half-margin: mid_h+wid_half+margin, wid_min-margin:wid_max+margin] 

    #getting left and right halves: TODO try with image patch left and right halves
    #lateral to left and medial to right

    img_patchl = segm_act[..., mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min-margin:wid_min+wid_half+margin] 
    img_patchr = segm_act[..., mid_h-wid_half//2-margin: mid_h+wid_half//2+margin, wid_min+wid_half-margin:wid_max+margin] 
    if img_patchl.max()!=2:
        img_patchl, img_patchr = img_patchr.clone(), img_patchl.clone()

    #getting focussed region from actual image
    adaptm=nn.AdaptiveAvgPool2d([conv4.shape[-2], conv4.shape[-1]])
    resized_patch = adaptm(img_patch)
    mse = nn.MSELoss()
    img_input = resized_patch[:,0,:,:].unsqueeze(dim=1)
    interx_input = conv4[:,0,:,:].unsqueeze(dim=1)
    loss_conv4 = mse(interx_input/interx_input.max(), img_input/img_input.max()).item()

    adaptm=nn.AdaptiveAvgPool2d([conv5.shape[-2], conv5.shape[-1]])
    patchl = adaptm(img_patchl.float().unsqueeze(dim=1))
    patchr = adaptm(img_patchr.float().unsqueeze(dim=1))
    conv5_input0 = conv5[:,0,:,:].unsqueeze(dim=1)
    conv5_input1 = conv5[:,1,:,:].unsqueeze(dim=1)
    loss0_conv5 = mse(conv5_input0/conv5_input0.max(), patchl/patchl.max())
    loss1_conv5 = mse(conv5_input1/conv5_input1.max(), patchr/patchr.max())
    return loss_conv4+loss0_conv5+loss1_conv5
    

def train(train_loader, model, criterion, epoch, num_epochs, device, batch_size, optimizer=None, lr=1e-4, 
trained_edgemodel=None, edge_mode=True):
    model.train()
    losses = AverageMeter()
    if edge_mode:
        lr = 1e-2 if epoch<10 else 1e-3
    if epoch > 20:
        lr = 1e-5
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(),lr=lr)
    # set a progress bar
    pbar = tqdm(enumerate(train_loader), total=int(len(train_loader.dataset)/batch_size))
    # pbar = tqdm(iter(train_loader), total=len(train_loader.dataset))
    for i, (images, segm_act) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.to(device))#.cuda())
        segm_act = Variable(segm_act.to(device))#cuda())

        # compute output
        optimizer.zero_grad()
        outputs, [dec0, dec1, dec2, dec3, dec4] = model(images)
        # outputs, _ = model(images)

        aggr_loss = 0
        inter_loss = 0
        # inter_loss+=loss_interxedge(outputs,segm_act, dec0)
        # inter_loss+=loss_interxedge(outputs,segm_act, dec1)
        # inter_loss+=loss_interxedge(outputs,segm_act, dec2)
        # inter_loss+=loss_interxedge(outputs,segm_act, dec3)
        # inter_loss+=loss_interxedge(outputs,segm_act, dec4)
        # aggr_loss+=inter_loss/5
        # loss_interx(images, segm_act, inter_x)
        for loss_ind, loss in enumerate(criterion):
            loss_each = loss(outputs, segm_act)
            aggr_loss+=loss_each

        losses.update(aggr_loss.item(), images.size(0))

        # compute gradient and do SGD step
        aggr_loss.backward()
        optimizer.step()

        # logging

        # # add the model graph
        # logger.add_graph(model, outputs)

        # # log loss values every iteration
        # logger.add_scalar('data/(train)loss_val', losses.val, i + 1)
        # logger.add_scalar('data/(train)loss_avg', losses.avg, i + 1)

        # # log the layers and layers gradient histogram and distributions
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.add_histogram('model/(train)' + tag, to_np(value), i + 1)
        #     logger.add_histogram('model/(train)' + tag + '/grad', to_np(value.grad), i + 1)

        # # log the outputs given by the model (The segmentation)
        # logger.add_image('model/(train)output', make_grid(outputs.data), i + 1)

        # update progress bar status
        pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(epoch avg) '
                             % (epoch + 1, num_epochs, losses.val, losses.avg))

    # return avg loss over the epoch
    return losses.avg

def evaluate(val_loader, model, criterion, val_acc_func, epoch, num_epochs, device, trained_edgemodel=None, edge_mode=True):
    """validation step

    Parameters
    ----------
    val_loader : torch dataloader
        validation dataloader
    model : torch model
        torch model
    criterion : list
        list of loss functions
    val_acc_func : function
        validation accuracy function
    epoch : [type]
        [description]
    num_epochs : [type]
        [description]
    device : [type]
        [description]
    trained_edgemodel : [type], optional
        [description], by default None
    edge_mode : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    model.eval()
    losses = AverageMeter()
    val_accs = AverageMeter()

    # set a progress bar
    pbar = tqdm(enumerate(val_loader))#, total=len(val_loader))
    #pbar = tqdm(iter(val_loader), total=len(val_loader.dataset))
    for i, (images, segm_act) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.to(device))
        segm_act = Variable(segm_act.to(device))

        with torch.no_grad():
            # compute output

            # measure loss
            aggr_loss = 0

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            for loss_ind, loss in enumerate(criterion):
                loss_each = loss(outputs, segm_act)
                aggr_loss+=loss_each
            losses.update(aggr_loss.item(), images.size(0))#aggr_loss.item()

            valid_val = val_acc_func(outputs, segm_act)
            val_accs.update(valid_val) #function already averages based on batch count
            
            pbar.set_description('[VALID] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(epoch avg) - BATCH DICE SCORE: %.4f/%.4f(epoch avg) '
                             % (epoch + 1, num_epochs, losses.val, losses.avg, val_accs.val, val_accs.avg))
       
    # return avg loss over the epoch
    return losses.avg, val_accs.avg

def plt_loss(data, max_epo, title, save_file_path, valid_after):
    """plot loss graph

    Parameters
    ----------
    data : list
        loss values
    max_epo : int
        count epochs
    title : str
        title
    save_file_path : path like
        figure save path
    valid_after : int
        number of epochs between validations
    """
    fig_plt, ax_plt = plt.subplots(1, 3)
    fig_plt.set_size_inches(19, 7)
    for ind, ax in enumerate(ax_plt):
        xmax=len(data[ind])
        ax.plot(range(xmax), data[ind], 'dodgerblue')#, label='training')
        # plt.plot(range(max_epo), validation_history['loss'], 'orange', label='validation')
        ax.axis(xmin=0,xmax=xmax)
        ax.set_xlabel(f'Epochs/{valid_after}' if ind!=0 else "Epochs")
        # plt.ylabel('Loss')
        ax.set_title(title[ind])
    plt.savefig(save_file_path, bbox_inches="tight")
    plt.close("all")
    # plt.legend();

def get_sav_mdl():
    pass
    # path=r'/models/myprefix_mymodel_128.pth'
    # model = SmallUNET256().to(device)
    # model.load_state_dict(torch.load(path))
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()

def valid(model,x_valid,y_valid,criterion):
    with torch.no_grad():
        model.eval()
        y_pred = model(x_valid)
        loss = criterion(y_pred, y_valid)
        print('test-loss',t, loss.item(),end=' ')
        return loss.item()

def naive_train(train_set, valid_set, model, criterion, val_acc_func, num_epochs, batch_size, device, out_path, kwargs, experiment_name="", 
valid_after=4, edge_mode=True, edge_model=None):
    """train model

    Parameters
    ----------
    train_set : torch dataset
        train set
    valid_set : torch dataset
        validation set
    model : torch model
        torch neural network model
    criterion : list
        loss functions list
    val_acc_func : function
        validation accuracy function
    num_epochs : int
        number of epochs
    batch_size : int
        batch size
    device : torch device
        cuda or cpu
    out_path : path like
        directory where results are saved
    kwargs : list
        additional parameters for dataloader
    experiment_name : str, optional
        experiment name, by default ""
    valid_after : int, optional
        number of epochs between validations, by default 4
    edge_mode : bool, optional
        if trained for edges, by default True
    edge_model : torch model, optional
        torch saved model for edges by default None
    """
    best_loss = 1000
    metric_dict = {"train_loss":[], "val_loss":[], "val_acc":[]}
    train_loader = DataLoader(train_set, batch_size=batch_size, **kwargs)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, **kwargs)
    for epoch in range(0, num_epochs):
        # train for one epoch
        curr_loss = train(train_loader, model, criterion, epoch, num_epochs, device, batch_size, 
        trained_edgemodel=edge_model, edge_mode=edge_mode)
        metric_dict["train_loss"].append(curr_loss)

        if not epoch%valid_after:
            val_loss, val_acc = evaluate(val_loader, model, criterion, val_acc_func, epoch, num_epochs, device, edge_model, edge_mode)
            metric_dict["val_loss"].append(val_loss)
            metric_dict["val_acc"].append(val_acc)

            # store best loss and save a model checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': experiment_name,
                'state_dict': model.state_dict(),
                # 'best_prec1': best_loss,
                # 'optimizer': optimizer.state_dict(),
            }, is_best, out_path)

    # logger.close()
    val_times = len(metric_dict["val_acc"])
    graph_path = os.path.join(out_path, "graph"+experiment_name)# for graph_type in metric_dict.keys()]
    # list(map(plt_loss, metric_dict.values(),[num_epochs, val_times, val_times], metric_dict.keys(), graph_paths))
    plt_loss(list(metric_dict.values()),num_epochs, list(metric_dict.keys()), graph_path, valid_after)
