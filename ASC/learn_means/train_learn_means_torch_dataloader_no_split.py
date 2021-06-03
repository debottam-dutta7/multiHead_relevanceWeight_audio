"""
train script for learning means using Hu's Baseline jointly.  
"""

from __future__ import print_function

import argparse
import csv
import os
import time
import numpy as np
import random
import torch 
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils_torch_learn_means_xpad import *
from Net_learn_means_AcFB import AcFB, Net_learn_means
import soundfile as sf
import librosa

experiment = 'exp_LM_baselineFull_joint_mixup_loss_ngf96_init_mel'

train_csv = '../asc_setup_files/fold1_train_add_time_pitch_noise_scorr.csv'
val_csv = '../asc_setup_files/fold1_evaluate_absolute_path.csv'
audio_path ='/home/data/DCASE2020/TAU-urban-acoustic-scenes-2020-mobile-development/'

num_audio_channels = 1
num_freq_bin = 80
win_len = 2048
hop_len = 1024
num_classes = 10
max_lr = 0.1
acfb_lr = 0.01
batch_size = 8 # 32-->16
num_epochs = 50
mixup_alpha = 0.4
crop_length = 400
seed = 1
augment = True
num_files_load = 512
#lrStep_epoch = 5
current_lr = max_lr
splice = 10

"""----------------------------------------------------------------------------------------"""
optim_method = 'sgd' # any one of 'sgd' or 'adam'
use_scheduler = True

use_cuda = torch.cuda.is_available()
best_acc = 0       # best test accuracy
start_epoch = 0    #start from epoch 0 or last checkpoint epoch

""" ====================================== Resuming =========================================="""
last_model_path = None  # for resuming replace with last checkpoint model path else None
resume = True if last_model_path is not None else False



if seed != 0:
    torch.manual_seed(seed)

""" Data """
print('===> Preparing data...')

"""============ load train and val lines==========="""
#train_label_info, train_labels = load_file_absolute(train_csv)
trainset = CustomDataset(audio_path,train_csv, file_type='.wav')
trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch_size,
                                        shuffle=True, num_workers=4)

testset = CustomDataset(audio_path, val_csv, file_type='.wav')
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,
                                            shuffle=False, num_workers=4)
iters = len(trainloader)
print("len of trainloader: %d batches" %iters)



""" Model """
net = Net_learn_means(num_classes, input_shape=[3*num_audio_channels, num_freq_bin, 431], 
               num_filters=[48, 96, 192], wd_in=0)
#net = Net()
acfb = AcFB()
criterion = nn.CrossEntropyLoss()

if use_cuda:
    net.cuda()
    acfb.cuda()
    #net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    #cudnn.benchmark = True  ## Enabling makes it faster training if input is of fixed size
    print('Using CUDA...')

if optim_method == 'sgd':
    optimizer = torch.optim.SGD([ {'params':acfb.parameters(), 'lr':acfb_lr},
                                {'params':net.parameters(),'lr': max_lr},
                                ], momentum=0.9, weight_decay=1e-6, nesterov=False)
    
elif optim_method == 'adam':
    optimizer = torch.optim.Adam([ {'params':acfb.parameters(), 'lr':acfb_lr},
                                {'params':net.parameters(),'lr': max_lr},
                                ], momentum=0.9, weight_decay=1e-6, nesterov=False)
    
if use_scheduler:
    scheduler = CosineAnnealingWarmRestarts(optimizer,1,
                                            T_mult=2,
                                            eta_min=max_lr*1e-4
                                            #last_epoch = -1
                                            #verbose=True
                                            )

if resume:
    print('====> Resuming from checkpoint...')
    checkpoint = torch.load(last_model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    acfb.load_state_dict(checkpoint['acfb_state_dict'])
    acc = checkpoint['acc']
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch'] + 1
    
    if use_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
    
    

else:
    print('===> Building model...')
    

if not os.path.isdir('results'):
    os.mkdir('results')

logname=('results/log_' + experiment+'_' + net.__class__.__name__+'_'+str(seed) + '.csv')



print("====================== params ======================\n")
print("num_audio_channels:{}, num_freq_bin:{}, win_len:{}, hop_len:{},num_classes:{},".format(num_audio_channels,num_freq_bin,win_len,hop_len,num_classes))
print("max_lr:{}, batch_size:{}, num_epochs:{}, optimizer:{}, use_scheduler:{}, use_cuda:{},num_files_load:{},splice:{}".format(max_lr,batch_size,num_epochs,optim_method,use_scheduler,use_cuda, num_files_load,splice))

def adjust_learning_rate(optimizer, lr):
    """Updates learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def extract_segments(inlines, tarlines, hop_length=1024, win_length=2048, splice=10):
    segments_all = []
    labels_all = []
    hop_length = hop_len
    win_length = win_len

    for i in range(np.alen(tarlines)):
        fn = inlines[i]
        # print(fn)
        utt_wav_path = inlines[i][0]
        #utt_wav_path_str = ''.join(utt_wav_path)
        #sig, fs = sf.read(utt_wav_path)
        sig,fs = librosa.load(utt_wav_path,sr=None)
        assert fs == 44100
        sound_clip = sig / (max(abs(sig)) + 1e-4)
        data_segments = enframe(sound_clip, win_length, hop_length)
        num_frames = data_segments.shape[0]
        #data_segments = torch.from_numpy(data_segments).reshape(1,1,num_frames, win_length)

        label = tarlines[i]
        segments_all.append(data_segments)
        labels_all.append(label)

    return np.array(segments_all), np.array(labels_all)

def mixup_data(x, y, alpha=0.4, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def frequency_masking(mel_spectrogram, frequency_masking_para=13, frequency_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0+f),:] = 0
    return mel_spectrogram

def time_masking(mel_spectrogram, time_masking_para=40, time_mask_num=1):
    fbank_size = mel_spectrogram.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0+t)] = 0
    return mel_spectrogram

def data_generation(inputs):
    # inputs = B,H,W,C
    
    for j in range(inputs.shape[0]):
        """ Spectrum Augmentation"""
        for c in range(inputs.shape[3]):
            inputs[j,:,:,c] = frequency_masking(inputs[j,:,:,c])
            inputs[j,:,:,c] = time_masking(inputs[j,:,:,c])
        
        """ Random Cropping"""
        StartLoc = np.random.randint(0,inputs.shape[2] - crop_length)
        inputs[j,:,0:crop_length,:] = inputs[j,:,StartLoc:StartLoc+crop_length,:]
    inputs = inputs[:,:,0:crop_length,:]

    return inputs

def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out


def train(epoch):
    print('\nEpoch: %d' %epoch)
    net.train()
    acfb.train()

    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    batches_seen = 0
    

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == len(trainloader)-1:
            continue;

        if use_cuda:
            inputs, targets = inputs.float().cuda(), targets.cuda()

        inputs = torch.unsqueeze(inputs, 1) 
        inputs = inputs.permute(0,1,3,2) # B,1,2048,431

            
        # ACFB (spec: B, s,t, c)
        spec = acfb(inputs)

        # SCALE 
        for i in range(spec.shape[0]):
            spec[i,:,:,:] = (spec[i,:,:,:].clone() - torch.min(spec[i,:,:,:].clone()))/(torch.max(spec[i,:,:,:].clone())-torch.min(spec[i,:,:,:].clone()))

        spec = spec.permute(0,2,3,1) # B,s,t,1
        spec_deltas = deltas(spec)
        spec_deltas_deltas = deltas(spec_deltas)
        base_inputs = torch.cat((spec[:,:,4:-4,:],spec_deltas[:,:,2:-2,:],spec_deltas_deltas),3)

        # SPEC AUG, RANDOM CROP
        base_inputs = data_generation(base_inputs)

        # MIX-UP
        base_inputs, targets_a, targets_b, lam = mixup_data(base_inputs, targets,
                                                       mixup_alpha, use_cuda)
            
        base_inputs = base_inputs.permute(0,3,1,2)
        outputs = net(base_inputs)
        
        loss = mixup_criterion(criterion, outputs, targets_a.long(), targets_b.long(), lam)
        
        with torch.no_grad():
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())           

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if use_scheduler:
            scheduler.step(epoch + batch_idx / iters) # iters = len(trainloader)

        batches_seen += 1
        progress_bar(batches_seen-1, len(trainloader),
                        'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                        %(train_loss/(batches_seen),reg_loss/(batches_seen),
                        100.*correct/total,correct,total))
            
    return (train_loss/batches_seen, reg_loss/batches_seen, 100.*correct/total)

def test(epoch):
    global best_acc
    
    #set to evaluation mode
    net.eval()
    acfb.eval()
    test_loss = 0
    correct = 0
    total = 0
    batches_seen = 0

    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == len(testloader)-1:
            continue

        if use_cuda:
            inputs, targets = inputs.float().cuda(), targets.cuda()
        inputs = torch.unsqueeze(inputs, 1)
        inputs = inputs.permute(0,1,3,2)

        # ACFB (spec: B, s,t, c)
        spec = acfb(inputs)
            
        # SCALE
        for i in range(spec.shape[0]):
            spec[i,:,:,:] = (spec[i,:,:,:].clone() - torch.min(spec[i,:,:,:].clone()))/(torch.max(spec[i,:,:,:].clone())-torch.min(spec[i,:,:,:].clone()))  


        spec = spec.permute(0,2,3,1)
        spec_deltas = deltas(spec)
        spec_deltas_deltas = deltas(spec_deltas)
        base_inputs = torch.cat((spec[:,:,4:-4,:],spec_deltas[:,:,2:-2,:],spec_deltas_deltas),3)
        base_inputs = base_inputs.permute(0,3,1,2)
        with torch.no_grad():
            outputs = net(base_inputs)
            loss = criterion(outputs, targets.long())
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        #print('sample: ',targets[0:10])
        #print(predicted[:10])
        #print('\n')
        batches_seen += 1

        progress_bar(batches_seen-1, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batches_seen), 100.*correct/total,
                        correct, total))
           
    print('sampling few outputs:\n')
    print('targets: ', targets)
    print('predicted: ', predicted)
    acc = 100.*correct/total

    save_checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc

    return (test_loss/batches_seen, 100.*correct/total)

def save_checkpoint(acc, epoch):
    """ Save checkpoint"""
    if use_scheduler:
        state={
            'model_state_dict': net.state_dict(),
            'acfb_state_dict': acfb.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc,
            'best_acc': best_acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state(),
            'scheduler_state_dict': scheduler.state_dict()

        }
    else:
        state={
        'model_state_dict': net.state_dict(),
        'acfb_state_dict': acfb.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acc': acc,
        'best_acc': best_acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    model_path = './checkpoint' + '/model_epoch_' + str(epoch) + '_'+'{0:.2f}'.format(acc)+'.pth'
    print('Saving model...: ' + model_path)
    
    torch.save(state, model_path) 

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch','train loss', 'reg loss', 
                'train acc', 'test loss', 'test acc'])


for epoch in range(start_epoch, num_epochs):
    start_time = time.time()
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    cur_time = time.time()
    time_per_epoch = cur_time - start_time
    print("Epoch {}/{} took {:.2f} secs ".format(epoch+1,num_epochs,time_per_epoch))
    print('Best acc: ', best_acc)

    print("LR updated to: {}".format(scheduler.get_last_lr()))
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc.detach().numpy(), 
            test_loss, test_acc.detach().numpy()])



