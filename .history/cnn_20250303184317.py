import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
# from CenterLoss import CenterLoss
from torch.autograd.function import Function

import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1_1=nn.Conv2d(1,32,kernel_size=5,padding=2)
        self.prelu1_1=nn.PReLU()
        self.conv1_2=nn.Conv2d(32,32,kernel_size=5,padding=2)
        self.prelu1_2=nn.PReLU()
        self.conv2_1=nn.Conv2d(32,64,kernel_size=5,padding=2)
        self.prelu2_1=nn.PRelu()
        self.conv2_2=nn.Conv2d(64,64,kernel_size=5,padding=2)
        self.prelu2_2=nn.PReLU()
        self.conv3_1=nn.Conv2d(64,128,kernel_size=5,padding=2)
        self.prelu3_1=nn.PReLL()
        self.conv3_2=nn.Conv2d(128,128,kernel_size=5,padding=2)
        self.prelu3_2=nn.PReLU()
        self.preluip1=nn.PReLU()
        self.ip1=nn.Linear(128*3*3,2)
        self.ip2=nn.Linear(2,10,bias=False)


    def forward(self,x):
        x = self.prelu1_1(self.conv1_1(x))
        x=self.prelu1_2(self.conv1_2(x))
        x=F.max_pool2d(x,2)
        x=self.prelu2_1(self.conv2_1(x))
        x=self.prelu2_2(self.conv2_2(x))
        x=F.max_pool2d(x,2)
        x=self.prelu3_1(self.conv3_1(x))
        x=self.prelu3_2(self.conv3_2(x))
        x=F.max_pool2d(x,2)
        x=x.view(-1,128*3*3)
        ip1=self.preluip1(self.ip1(x))
        ip2=self.ip2(ip1)


        return ip1,F.log_softmax(ip2,dim=1)





class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss

class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None




def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.xlim(xmin=-8,xmax=8)
    plt.ylim(ymin=-8,ymax=8)
    plt.text(-7.8,7.3,"epoch=%d" % epoch)
    # plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)