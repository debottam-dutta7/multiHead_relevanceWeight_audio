
from __future__ import print_function

import argparse
import csv
import os
import time
import numpy as np
import random
import torch
import librosa 
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import scipy.io as sio

from utils_torch_learn_means import *
#from Net_raw_AcFB_taslp_wo_RelWt import Net
from Net_learn_means_AcFB import AcFB
import soundfile as sf

#OUTPATH = '/home/debottamd/Hu-rep/learned-features-with-base-specs/'
OUTPATH = '/home/debottamd/Hu-rep/learned-features-huBaseline-joint-f80/'
train_csv = '../../asc_setup_files/fold1_train_add_time_pitch_noise.csv'
val_csv = '../../asc_setup_files/fold1_evaluate_absolute_path.csv'

train_label_info, train_labels = load_file_absolute(train_csv)
val_label_info, val_labels = load_file_absolute(val_csv)

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)



num_audio_channels = 1
num_freq_bin = 80
win_len = 2048
hop_len = 1024
num_classes = 10
max_lr = 0.1
acfb_lr = 0.01
batch_size = 32 # 32-->16
num_epochs = 50
#mixup_alpha = 0.4
#crop_length = 400
seed = 0
augment = True
num_files_load = 30
lrStep_epoch = 5
current_lr = max_lr

use_cuda = True

#------------------------ RESUME -----------------
resume = False
resume_train_idx = 0
resume_val_idx = 0

if resume:
    train_label_info = train_label_info[resume_train_idx:]
    train_labels = train_labels[resume_train_idx:]
    
    val_label_info = val_label_info[resume_val_idx:]
    val_labels = val_labels[resume_val_idx:]
    print('Resuming ======> ')


if seed != 0:
    torch.manual_seed(seed)

""" Data """
print('===> Preparing data...')

""" Model """
#net = Net()
net = AcFB()


def extract_segments(wavfile, labelfile, hop_length=1024, win_length=2048, splice=0):
    
    win_length = win_len
    hop_length = hop_len

    utt_wav_path_str = wavfile
    #sig,fs = sf.read(utt_wav_path_str) 
    #sig = sig[:,0].astype(np.float32)
    sig,fs = librosa.load(utt_wav_path_str,sr=None) 
    sound_clip = sig / (max(abs(sig)) + 1e-4)
    data_segments = enframe(sound_clip, win_length, hop_length)

    label = labelfile

    return data_segments, label

def save_files(label_info, labels):

    net.eval()
    infiles = label_info
    tarfiles = labels

    for i in range(len(infiles)):
        inp_features, inp_labels = extract_segments(infiles[i][0], tarfiles[i],hop_length=hop_len, win_length=win_len)
        inp_features = np.expand_dims(inp_features,axis=0)
        inp_features = np.expand_dims(inp_features, axis=0) # 1,1,1102,998
        
        
        inputs = torch.from_numpy(inp_features)
        inputs = inputs.permute(0,1,3,2)  # 1,1,2048,431
        if use_cuda:
            inputs = inputs.float().cuda()
            
        
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
        
        filename = OUTPATH + infiles[i][0].split('/')[-1].split('.')[0] + '.mat'
        print('saving:',filename)
        sio.savemat(filename, mdict={'data':outputs.data.cpu().numpy()})
        
    sys.stdout.flush()
    

save_files(train_label_info, train_labels)
print('Done saving fold1_train_add_time_pitch_noise data....')

save_files(val_label_info, val_labels)
print('Done saving fold1_evaluate data...')
print('\n')
print('Done saving data.')
    


