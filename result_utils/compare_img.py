import matplotlib.pyplot as plt
def pred_act_diff(pred1, segm1):
    """compare mask and predictions true postive, tn,fp,fn

    Parameters
    ----------
    pred1 : `torch.tensor`
        logits
    segm1 : `torch.tensor`
        target
    """
    #take one
    index = 4
    pred1 = outputs[index].clone()
    pred1 = torch.argmax(pred1, dim=0)
    segm1 = segm[index].clone()
    comp_img = torch.zeros_like(segm1)
    comp_img[segm1==pred1]=1 #true positives
    true_bg = [segm1==0] and [pred1==0]
    comp_img[true_bg] = 0
    comp_img[segm1!=pred1] = 2