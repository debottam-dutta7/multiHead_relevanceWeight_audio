import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy
import scipy.signal
import scipy.io as sio
import numpy as np
import torch.nn.functional as F


class Relevance_weights_acousticFB1(torch.nn.Module):
    def __init__(self, ngf=80, splice = 10):
        super(Relevance_weights_acousticFB1, self).__init__()
        self.ngf = ngf
        self.splice = splice
        self.patch_len = 2*self.splice + 1
        self.fc1 = nn.Linear(self.patch_len, 50)
        self.fc2 = nn.Linear(50,1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #x is (B, 1, ngf=80, patch_length=431)

        B,t,f = x.shape[0],x.shape[3],x.shape[2]
        
        x = F.unfold(x, (self.ngf, self.patch_len), padding=(0,self.splice)) # B, patch_len*ngf,431
        x = x.permute(0,2,1)
        x = x.reshape(B,t,f,self.patch_len) # B,431,80,21

        
        x = x.reshape(B*t*f, -1) #  (*,21)
        x = self.sigmoid(self.fc1(x))
        x = (self.fc2(x))
        x = x.reshape(B,t, -1)  # B, 431, 80
        # print out.shape
        #out = self.softmax(x)
        out = self.sigmoid(x)

        return out

class Relevance_weights_acousticFB2(torch.nn.Module):
    def __init__(self, ngf=80, splice = 10):
        super(Relevance_weights_acousticFB2, self).__init__()
        self.ngf = ngf
        self.splice = splice
        self.patch_len = 2*self.splice + 1
        self.fc1 = nn.Linear(self.patch_len, 50)
        self.fc2 = nn.Linear(50,1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #x is (B, 1, ngf=80, patch_length=431)

        B,t,f = x.shape[0],x.shape[3],x.shape[2]
        
        x = F.unfold(x, (self.ngf, self.patch_len), padding=(0,self.splice)) # B, patch_len*ngf,431
        x = x.permute(0,2,1)
        x = x.reshape(B,t,f,self.patch_len) # B,431,80,21

        
        x = x.reshape(B*t*f, -1) #  (*,21)
        x = self.sigmoid(self.fc1(x))
        x = (self.fc2(x))
        x = x.reshape(B,t, -1)  # B, 431, 80
        # print out.shape
        #out = self.softmax(x)
        out = self.sigmoid(x)

        return out



class channel_attention(nn.Module):
    def __init__(self,input_channel = 10, ratio=8):
        super(channel_attention,self).__init__()
        
        # input's 2nd axis is channel axis
        #self.input_channel = input_channel
        self.shared_layer_one = nn.Linear(input_channel,input_channel//ratio,
                                         bias=True)
        self.relu = nn.ReLU()
        
        self.shared_layer_two = nn.Linear(input_channel//ratio,input_channel,
                                         bias=True)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,x):
        
        #x -> N,C,H,W
        batch_size = x.size()[0]
        input_channel = x.size()[1]
        y = F.avg_pool2d(x,kernel_size=x.size()[2:]) # y-> N,C,1,1
        y = y.reshape(batch_size,-1) 
        y = self.shared_layer_one(y) # y->N,C//ratio
        y = self.relu(y)
        y = self.shared_layer_two(y)
        y = self.relu(y)
        assert y.size()[-1] == input_channel
        y = y.reshape(batch_size,1,1,input_channel)
        
        #print(x.shape)
        z = F.max_pool2d(x,kernel_size=x.size()[2:])
        z = z.reshape(batch_size,-1)
        assert z.size()[-1] == input_channel
        z = self.shared_layer_one(z)
        z = self.relu(z)
        z = self.shared_layer_two(z)
        #print(z.shape)
        assert z.shape[-1] == input_channel
        z = z.reshape(batch_size,1,1,input_channel)
        
        cbam_feature = torch.add(y,z)
        cbam_feature = self.sigmoid(cbam_feature) # batch_size,1,,1,C
        cbam_feature = cbam_feature.permute(0,3,1,2) #batch_size,C,1,1
        
        
        return cbam_feature * x


class model_fcnn(nn.Module):
    def __init__(self, num_classes=10, input_shape=[3,128,400],
                num_filters=[24,48,96], wd_in=1e-3):
        super(model_fcnn,self).__init__()
        
        #input --> N,C,H,W = B,3,128,400/431
        self.num_channels = input_shape[0]
        self.num_classes = num_classes
        self.relu = nn.ReLU()

        """ conv_layer1"""
        self.kernel_size1_1 = [5,5]
        self.kernel_size1_2 = [3,3]
        self.stride1_1 = [2,2]
        self.stride1_2 = [1,1]

        #1
        self.bn1_1 = nn.BatchNorm2d(self.num_channels, affine=True)
        self.conv1_1 = nn.Conv2d(self.num_channels, self.num_channels*num_filters[0], kernel_size=self.kernel_size1_1,
            stride=self.stride1_1,padding=(2,2), bias=False)
        #2
        self.bn1_2 = nn.BatchNorm2d(self.num_channels*num_filters[0], affine=True)
        self.conv1_2 = nn.Conv2d(num_filters[0]*self.num_channels,num_filters[0]*self.num_channels,kernel_size=self.kernel_size1_2,
            stride=self.stride1_2,padding=(1,1),bias=False)

        self.bn1_3 = nn.BatchNorm2d(self.num_channels*num_filters[0], affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        
        """ conv_layer2"""
        self.input_channels2 = self.num_channels * num_filters[0]
        self.kernel_size2 = [3,3]
        self.stride2 = [1,1]
        #3
        self.conv2_1 = nn.Conv2d(self.input_channels2, self.num_channels*num_filters[1],
                              kernel_size=self.kernel_size2,stride=self.stride2,
                              padding=(1,1), bias=False)
        #4
        self.bn2_1 = nn.BatchNorm2d(num_filters[1]*self.num_channels, affine=True)
        self.conv2_2 = nn.Conv2d(self.num_channels*num_filters[1],self.num_channels*num_filters[1],
                              kernel_size=self.kernel_size2,stride=self.stride2,
                              padding=(1,1), bias=False)
        #5
        self.bn2_2 = nn.BatchNorm2d(num_filters[1]*self.num_channels, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2))


        """ conv_layer3"""
        self.input_channels3 = self.num_channels * num_filters[1]
        self.kernel_size3 = [3,3]
        self.stride3 = [1,1]

        #6
        self.conv3_1 = nn.Conv2d(self.input_channels3,num_filters[2]*self.num_channels,
                              kernel_size=self.kernel_size3, stride=self.stride3,
                              padding=(1,1), bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_filters[2]*self.num_channels,
                affine=True)
        self.d1 = nn.Dropout(0.3)
        
        #7
        self.conv3_2 = nn.Conv2d(num_filters[2]*self.num_channels,num_filters[2]*self.num_channels,
                              kernel_size=self.kernel_size3, stride=self.stride3,
                              padding=(1,1), bias=False)
        self.bn3_2 = nn.BatchNorm2d(num_filters[2]*self.num_channels,
                                  affine=True)
        self.d2 = nn.Dropout(0.3)
        #8
        self.conv3_3 = nn.Conv2d(num_filters[2]*self.num_channels,num_filters[2]*self.num_channels,
                              kernel_size=self.kernel_size3,stride=self.stride3,
                              padding=(1,1),bias=False)
        self.bn3_3 = nn.BatchNorm2d(self.num_channels*num_filters[2],
                                  affine=True)
        self.d3 = nn.Dropout(0.3)
        #9
        self.conv3_4 = nn.Conv2d(num_filters[2]*self.num_channels,num_filters[2]*self.num_channels,
                              kernel_size=self.kernel_size3,stride=self.stride3,
                              padding=(1,1), bias=False)
        self.bn3_4 = nn.BatchNorm2d(self.num_channels*num_filters[2],
                                  affine=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

        
        """ resnet_layer"""
        self.input_channelsR = self.num_channels * num_filters[2]
        self.kernel_sizeR = 1
        self.strideR = 1

        self.convR_1 = nn.Conv2d(self.input_channelsR,self.num_classes,
                               kernel_size=self.kernel_sizeR,stride=self.strideR,bias=False)
        self.bnR = nn.BatchNorm2d(self.num_classes, affine=False)

        
        """ channel_attention """

        self.channel_attn = channel_attention(input_channel=self.num_classes,ratio=2)
        self.bnF = nn.BatchNorm2d(self.num_classes, affine=False)
        self.softmax = nn.Softmax(dim=1)

        

    def forward(self,x):
            
        """ conv_layer1"""
        x = self.bn1_1(x)
        x = self.conv1_1(x)
        x = self.bn1_2(x)
        x = self.relu(x)
            
        x = self.conv1_2(x)
        x = self.bn1_3(x)
        x = self.relu(x)
        x = self.pool1(x)

        """ conv_layer2"""
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x = self.pool2(x)

        """ conv_layer3"""
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.d1(x)

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.d2(x)

        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.relu(x)
        x = self.d3(x)

        x = self.conv3_4(x)
        x = self.bn3_4(x)
        x = self.relu(x)
        x = self.pool3(x)

        """ resnet_layer """
        x = self.convR_1(x)
        x = self.bnR(x)
        x = self.relu(x)

        """ channel_attention"""
        x = self.bnF(x)
        x = self.channel_attn(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        x = torch.squeeze(x)
        

        return x 










        


