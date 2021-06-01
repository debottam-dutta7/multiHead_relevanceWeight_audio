import torch
import torch.nn as nn
from torch.autograd import Variable
#import scipy
#import scipy.signal
#import scipy.io as sio
import numpy as np
import torch.nn.functional as F 


class NIN_weights_acoustic(torch.nn.Module):
    def __init__(self):
        super(NIN_weights_acoustic, self).__init__()
        self.ngf = 80

        self.fc1 = nn.Linear(51, 10)
        self.fc2 = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0,3,2,1)  # B,40,51,1
        x = x.reshape(batch_size * x.shape[1], -1)  # (*, 51)
        x = self.sigmoid(self.fc1(x))
        x = (self.fc2(x))
        x = x.reshape(batch_size, -1)
        # print out.shape
        out = self.softmax(x)  # B, 40
        return out


class NIN_weights_mod(torch.nn.Module):
    def __init__(self):
        super(NIN_weights_mod, self).__init__()
        self.ngf = 80
        # self.patch_length = 101
        self.num_mod_filt = 40

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3))
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3))
        self.fc1 = nn.Linear(224, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is (N, mod_filt=40, patch_len=47, ngf=76)
        batch_size = x.shape[0]
        x = x.reshape(batch_size * x.shape[1], 1, x.shape[2], x.shape[3])
        x = self.sigmoid(self.conv1(x))  # Bx40,8,45,74
        x = self.pool1(x)  # Bx40, 8, 15, 24
        x = self.sigmoid(self.conv2(x)) # Bx40, 8, 13, 23
        x = self.pool2(x)  # Bx40, 8, 4, 7
        x = x.reshape(x.shape[0], -1) # Bx40, 8x4x7=224
        x = self.fc1(x)
        x = x.reshape(-1, self.num_mod_filt)
        # print out.shape
        out = self.softmax(x)
        return out


class Net (nn.Module):
    def __init__(self):
        super(Net,self).__init__()
    
        self.nin_filter_wts_acoustic1 = NIN_weights_acoustic()
        self.nin_filter_wts_acoustic2 = NIN_weights_acoustic()
        self.nin_filter_wts_mod = NIN_weights_mod()

        self.num_mod_filt = 40
        self.ngf = 80
        self.num_classes = 10
        self.hid_layer_size = 256

        self.bn = torch.nn.BatchNorm2d(1)
        self.instance_norm = torch.nn.InstanceNorm2d(self.ngf, eps=1e-4)
        self.conv1   = nn.Conv2d(1, 40, kernel_size=(5,5))
        self.pool1   = nn.MaxPool2d(kernel_size=(1,3))
        self.bn_1 = torch.nn.BatchNorm2d(self.num_mod_filt, eps=1e-4, affine=True, track_running_stats=True)
        self.lstm1   = nn.LSTM(25*self.num_mod_filt, self.hid_layer_size, num_layers=2, batch_first=True, dropout = 0.25)
        self.fc      = nn.Linear(self.hid_layer_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # B, 1, 80, 51
        batch_size = x.shape[0]
        x = x.permute(0,1,3,2)

        # x is (N, C, patch_len, ngf)
        x = self.bn(x)
        x1 = x[:,:,:,:self.ngf//2]
        x2 = x[:,:,:,self.ngf//2:]
    
        # x is (N, C, patch_len, ngf)
        filter_wts_acoustic1 = self.nin_filter_wts_acoustic1(x1)
        filter_wts_acoustic2 = self.nin_filter_wts_acoustic2(x2)
        filter_wts_acoustic1 = filter_wts_acoustic1.reshape(filter_wts_acoustic1.shape[0], filter_wts_acoustic1.shape[1], 1, 1)
        filter_wts_acoustic2 = filter_wts_acoustic2.reshape(filter_wts_acoustic2.shape[0], filter_wts_acoustic2.shape[1], 1, 1)

        x1 = filter_wts_acoustic1 * x1.permute(0,3,2,1) # B, 80, 51, 1
        x2 = filter_wts_acoustic2 * x2.permute(0,3,2,1) # B, 80, 51, 1
        
        # SKIP ADD
        x = x.permute(0,3,2,1)
        x1 = x1 + x[:,:self.ngf//2,:,:]
        x2 = x2 + x[:,self.ngf//2:,:,:]
        
        # CONCATE
        x = torch.cat((x1,x2), 1)
        assert x.shape[1] == self.ngf

        x = self.instance_norm(x)
        x = x.permute(0,3,2,1) # B, 1, 51, 80

        x = (self.conv1(x)) # B, 40, 17, 76
       
        x = self.bn_1(x)

        x = self.pool1(x)          # B, 40, 17, 25
        seq_len = x.shape[2]
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        x,_ = self.lstm1(x)
        x = x[:, seq_len-1, :]
        x = self.fc(x)
        return x
