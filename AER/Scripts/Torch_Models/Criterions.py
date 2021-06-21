import torch
import torch.nn as nn
from torch.autograd import Variable
# import torch.nn.functional as F
import numpy as np

def calc_CCC(prediction, ground_truth):
    mean_gt = torch.mean (ground_truth, 0)
    mean_pred = torch.mean (prediction, 0)
    var_gt = torch.var (ground_truth, 0)
    var_pred = torch.var (prediction, 0)
    v_pred = prediction - mean_pred
    v_gt = ground_truth - mean_gt
    denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
    cov = torch.mean(v_pred * v_gt)
    numerator=2*cov
    ccc = numerator/denominator
    return ccc

class CCC_Loss(nn.Module):
    def __init__(self):
        super(CCC_Loss, self).__init__()

    def forward(self, prediction, ground_truth):
        # ground_truth = (ground_truth == torch.arange(self.num_classes).cuda().reshape(1, self.num_classes)).float()
        # ground_truth = ground_truth.squeeze(0)
        # prediction = prediction.view(prediction.size()[0]*prediction.size()[1])#.squeeze(1)
        # print("")
        # print("ground_truth", ground_truth.shape)
        # print("prediction", prediction.shape)
        prediction = prediction.view(-1)
        ground_truth = ground_truth.view(-1)
        ccc = calc_CCC(prediction, ground_truth)
        # print("ccc", ccc, mean_gt, mean_pred,var_gt,var_pred)
        return 1-ccc

# class multi_CCC


