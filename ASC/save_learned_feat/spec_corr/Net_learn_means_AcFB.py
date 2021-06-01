import torch
import torch.nn as nn
import scipy.signal
import numpy as np
import torch.nn.functional as F
import random
import scipy.io as sio


class channel_attention(nn.Module):
    def __init__(self,input_channel = 10, ratio=8):
        super(channel_attention,self).__init__()
        
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

class AcFB(nn.Module):
    def __init__(self):
        super(AcFB, self).__init__()
        """ returns the learned spectrograms"""
        
        self.ngf = 80
        self.num_classes = 10

        self.filt_h = int(16 * 44.1)     # = 705 
        self.padding = int((16 * 44.1)//2) # + 1
        self.win_length = 2048 
        self.patch_length = 11
        
        
        self.len_after_conv = 1344   #1102  #=win_length?
        self.hamming_window = torch.from_numpy(scipy.signal.hamming(self.win_length)).float().cuda()
        
        means_step1 = sio.loadmat('/home/debottamd/Hu-rep/learn-fbank/learn-means-huBaseline-joint/with-mixup-ngf-80-acfb0.01/filt/means_epoch-14_acc-65.66.mat')
        means_sorted, mean_idx = torch.sort(torch.from_numpy(np.asarray(means_step1['data'],dtype=np.float32).reshape(self.ngf)))
        self.means = (means_sorted.float().cuda())   


        t = range(-self.filt_h//2, self.filt_h//2)
        self.temp = torch.from_numpy((np.reshape(t, [self.filt_h, ]))).float().cuda() + 1
        
        self.avg_pool_layer = torch.nn.AvgPool2d((self.len_after_conv, 1), stride=(1,1))
        self.kernels = (torch.zeros([self.ngf, self.filt_h]).cuda())
        for i in range(self.ngf):
            self.kernels[i, :] = torch.cos(np.pi * torch.sigmoid(self.means[i]) * self.temp) * torch.exp(- (((self.temp)**2)/(2*(((1/(torch.sigmoid(self.means[i])+1e-3))*10)**2 + 1e-5))))
        
        self.kernels = (torch.reshape(self.kernels, (self.kernels.shape[0], 1, self.kernels.shape[1], 1))) 
                                                                                   

    def forward(self, x):
        # x = B,C,H,W

        patch_length = x.shape[3] # no of time frames
        assert patch_length == 431

        x = x.permute(0,1,3,2) # B,1,431,2048=B,1,W,H check
        
        x = x * self.hamming_window
        x = x.permute(0,2,3,1) # B, W, H,C
        
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size * x.shape[1],1,self.win_length,1)) # B*W, 1, H, 1
        
        x = F.conv2d(x, self.kernels) 

        # here x = B*w, ngf, H, 1 ; w = patch_length, H=len_after_conv=win_length
        x = torch.reshape(x, (batch_size, patch_length, self.ngf, self.len_after_conv)).permute(0,2,3,1)  # B, ngf, H, W
        x = torch.abs(x)**2

        x = torch.log(self.avg_pool_layer(x) + 1e-3)
        x = x.permute(0,2,1,3) # B, 1, ngf, W
        assert x.shape[2] == self.ngf
        assert x.shape[3] == patch_length
        return x


class Net_learn_means(nn.Module):
    def __init__(self,num_classes=10, input_shape=[1,128,431],
                num_filters=[24,48,96], wd_in=1e-3):

        """ Similar to model_fcnn architecture except 
        added AcFB conv layer at the top and 3 of the 4 conv2d of conv_layer3 removed"""
        
        super(Net_learn_means, self).__init__()

        #self.acfb = AcFB()

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

        """ conv_layer3 """
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
