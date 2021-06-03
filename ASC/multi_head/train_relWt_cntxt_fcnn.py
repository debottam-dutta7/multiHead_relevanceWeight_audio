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
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from utils_torch import CustomDataset, progress_bar
#from Net_fcnn_channel_attn import *
from Net_fcnn_2head_sigmoid_relWt_cntxt import *

experiment = 'relWt_2head_split_freq_skipConnect_catAfterAug_alignedAll'

train_csv = '../asc_setup_files/fold1_train_add_time_pitch_noise_scorr.csv'
val_csv = '../asc_setup_files/fold1_evaluate_absolute_path.csv'

# saved learned feature path
feat_path = '/home/debottamd/Hu-rep/learned-features-huBaseline-joint-f80/'


num_audio_channels = 1
num_freq_bin = 80
num_classes = 10
max_lr = 0.1
batch_size = 32
num_epochs = 130
mixup_alpha = 0.4
crop_length = 400
splice = 10
seed = 1
augment = True

use_cuda = torch.cuda.is_available()
best_acc = 0       # best test accuracy
start_epoch = 0    #start from epoch 0 or last checkpoint epoch
last_model_path = None  # for resuming replace with last checkpoint model
resume = True if last_model_path is not None else False


last_saved_epoch = 0

if seed != 0:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    

""" Data """
print('===> Preparing data...')


trainset = CustomDataset(feat_path, train_csv, 'mat')
trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch_size,
                                        shuffle=True, num_workers=4)
testset = CustomDataset(feat_path, val_csv, file_type='mat')
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,
                                            shuffle=False, num_workers=4)
iters = len(trainloader)
print("len of trainloader %d" %iters)

""" Model """
net = model_fcnn(num_classes, input_shape=[3*num_audio_channels,num_freq_bin,400], 
                num_filters=[48, 96, 192], wd_in=0) # num_filters reduced by half

acfb_relWt_net1 = Relevance_weights_acousticFB1(ngf=num_freq_bin//2, splice=splice)
acfb_relWt_net2 = Relevance_weights_acousticFB1(ngf=num_freq_bin//2, splice=splice)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params':net.parameters(), 'lr': max_lr,},
                            {'params':acfb_relWt_net1.parameters(), 'lr': max_lr},
                            {'params':acfb_relWt_net2.parameters(), 'lr': max_lr},
                            ], momentum=0.9, weight_decay=1e-6, nesterov=False)

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
    acfb_relWt_net1.load_state_dict(checkpoint['acfb_relWt_net1_state_dict'])
    acfb_relWt_net2.load_state_dict(checkpoint['acfb_relWt_net2_state_dict'])
    acc = checkpoint['acc']
    best_acc = checkpoint['best_acc']
    if last_model_path is not None:
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

else:
    print('===> Building model...')
    

if not os.path.isdir('results'):
    os.mkdir('results')

logname=('results/log_'+experiment+'_' + net.__class__.__name__+'_'+str(seed) + '.csv')

if use_cuda:
    net.cuda()
    acfb_relWt_net1.cuda()
    acfb_relWt_net2.cuda()
    criterion.cuda()
    #net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    #cudnn.benchmark = True  ## Enabling makes it faster training if input is of fixed size
    print('Using CUDA...')

print("====================== params ======================\n") 
print("num_audio_channels:{}, num_freq_bin:{}, num_classes:{},".format(num_audio_channels,num_freq_bin,num_classes)) 
print("max_lr:{}, batch_size:{}, num_epochs:{}, use_cuda:{}, ".format(max_lr,batch_size,num_epochs,use_cuda))


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


def frequency_masking_aligned(mel_spectrogram1, mel_spectrogram2, frequency_masking_para=13, frequency_mask_num=1):
    """  
    mel_spectrogram1: (ngf,t,c)
    mel_spectrogram1: (ngf,t,c)
    """
    fbank_size = mel_spectrogram1.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram1[f0:(f0+f),:,:] = 0
        mel_spectrogram2[f0:(f0+f),:,:] = 0
    return mel_spectrogram1, mel_spectrogram2

def time_masking_aligned(mel_spectrogram1, mel_spectrogram2, time_masking_para=40, time_mask_num=1):
    """  
    mel_spectrogram1: (ngf,t,c)
    mel_spectrogram1: (ngf,t,c)
    """

    fbank_size = mel_spectrogram1.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram1[:, t0:(t0+t),:] = 0
        mel_spectrogram2[:, t0:(t0+t),:] = 0
    return mel_spectrogram1, mel_spectrogram2

def data_generation_2head(inputs1,inputs2):
    # inputs = B,H,W,C
    
    for j in range(inputs1.shape[0]):
        """ Spectrum Augmentation"""
        
        inputs1[j,:,:,:], inputs2[j,:,:,:] = frequency_masking_aligned(inputs1[j,:,:,:],inputs2[j,:,:,:])
        inputs1[j,:,:,:], inputs2[j,:,:,:] = time_masking_aligned(inputs1[j,:,:,:],inputs2[j,:,:,:])
        
        """ Random Cropping"""
        StartLoc = np.random.randint(0,inputs1.shape[2] - crop_length)
        inputs1[j,:,0:crop_length,:] = inputs1[j,:,StartLoc:StartLoc+crop_length,:]
        inputs2[j,:,0:crop_length,:] = inputs2[j,:,StartLoc:StartLoc+crop_length,:]
    inputs1 = inputs1[:,:,0:crop_length,:]
    inputs2 = inputs2[:,:,0:crop_length,:]

    return inputs1, inputs2


def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out  


def train(epoch):
    print('\nEpoch: %d' %epoch)
    net.train()
    acfb_relWt_net1.train()
    acfb_relWt_net2.train()

    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == len(trainloader)-1:
            continue

        if use_cuda:
            inputs, targets = inputs.cuda().float(), targets.cuda()
       
        inputs = inputs.unsqueeze(3) 
        
        # REL. WEIGHTING
        inputs = inputs.permute(0,3,1,2) # B,1,ngf,t
        inputs1 = inputs[:,:,0:num_freq_bin//2,:]
        inputs2 = inputs[:,:,num_freq_bin//2:,:]

    

        filter_wts_acousticFB1 = acfb_relWt_net1(inputs1)  # B, t, ngf
        filter_wts_acousticFB2 = acfb_relWt_net2(inputs2)  # B, t, ngf

        filter_wts_acousticFB1 = filter_wts_acousticFB1.unsqueeze(1).permute(0,1,3,2).contiguous() # B,1,f,t
        filter_wts_acousticFB2 = filter_wts_acousticFB2.unsqueeze(1).permute(0,1,3,2).contiguous() # B,1,f,t

        assert filter_wts_acousticFB1.shape == inputs1.shape
        inputs1 = filter_wts_acousticFB1 * inputs1
        inputs1 = inputs1.permute(0,2,3,1)

        assert filter_wts_acousticFB2.shape == inputs2.shape
        inputs2 = filter_wts_acousticFB2 * inputs2
        inputs2 = inputs2.permute(0,2,3,1)


        # SKIP CONNECTION
        inputs = inputs.permute(0,2,3,1) # B,f,t,1

        inputs1 = inputs1 + inputs[:,0:num_freq_bin//2,:,:]
        inputs2 = inputs2 + inputs[:,num_freq_bin//2:,:,:]




        # SCALE
        for i in range(inputs1.shape[0]):
            inputs1[i,:,:,:] = (inputs1[i,:,:,:].clone() - torch.min(inputs1[i,:,:,:].clone()))/(torch.max(inputs1[i,:,:,:].clone())-torch.min(inputs1[i,:,:,:].clone()))
            inputs2[i,:,:,:] = (inputs2[i,:,:,:].clone() - torch.min(inputs2[i,:,:,:].clone()))/(torch.max(inputs2[i,:,:,:].clone())-torch.min(inputs2[i,:,:,:].clone()))


        inputs_deltas1 = deltas(inputs1)
        inputs_deltas_deltas1 = deltas(inputs_deltas1)
        inputs1 = torch.cat((inputs1[:,:,4:-4,:],inputs_deltas1[:,:,2:-2,:],inputs_deltas_deltas1),3)

        
        inputs_deltas2 = deltas(inputs2)
        inputs_deltas_deltas2 = deltas(inputs_deltas2)
        inputs2 = torch.cat((inputs2[:,:,4:-4,:],inputs_deltas2[:,:,2:-2,:],inputs_deltas_deltas2),3)
        
        
        inputs1, inputs2 = data_generation_2head(inputs1, inputs2)
        


        # STACK THE TWO WEIGHTED SPECTROGRAMS AND THEIR DELTAS ALONG THE SPLITTED DIM
        inputs = torch.cat((inputs1,inputs2),1)
        

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       mixup_alpha, use_cuda)
        assert inputs.shape[3] == 3
        inputs = inputs.permute(0,3,1,2)  # change to N,C,H,W 
        assert inputs.shape[1] == 3 
        
        outputs = net(inputs)
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
        scheduler.step(epoch + batch_idx / iters) # iters = len(trainloader)
        #print("LR updated to: {}".format(scheduler.get_last_lr()))

        progress_bar(batch_idx, len(trainloader),
                        'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                        %(train_loss/(batch_idx+1),reg_loss/(batch_idx+1),
                        100.*correct/total,correct,total))
       
        return (train_loss/(batch_idx+1), reg_loss/(batch_idx+1), 100.*correct/total)

def test(epoch):
    global best_acc
    global last_saved_epoch
    
    #set to evaluation mode
    net.eval()
    acfb_relWt_net1.eval()
    acfb_relWt_net2.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == len(testloader)-1:
            continue

        if use_cuda:
            inputs, targets = inputs.cuda().float(), targets.cuda()
        
       
        inputs = inputs.unsqueeze(3) 

        with torch.no_grad():

            # REL. WEIGHTING
            #inputs = inputs.permute(0,3,1,2) # B,1,f,t

            inputs = inputs.permute(0,3,1,2) # B,1,ngf,t
            inputs1 = inputs[:,:,0:num_freq_bin//2,:]
            inputs2 = inputs[:,:,num_freq_bin//2:,:]

    

            filter_wts_acousticFB1 = acfb_relWt_net1(inputs1)  # B, ngf
            filter_wts_acousticFB2 = acfb_relWt_net2(inputs2)  # B, ngf
            
            filter_wts_acousticFB1 = filter_wts_acousticFB1.unsqueeze(1).permute(0,1,3,2).contiguous() # B,1,f,t
            filter_wts_acousticFB2 = filter_wts_acousticFB2.unsqueeze(1).permute(0,1,3,2).contiguous() # B,1,f,t

            assert filter_wts_acousticFB1.shape == inputs1.shape
            inputs1 = filter_wts_acousticFB1 * inputs1
            inputs1 = inputs1.permute(0,2,3,1)

            assert filter_wts_acousticFB2.shape == inputs2.shape
            inputs2 = filter_wts_acousticFB2 * inputs2
            inputs2 = inputs2.permute(0,2,3,1)


            # SKIP CONNECTION
            inputs = inputs.permute(0,2,3,1) # B,f,t,1

            inputs1 = inputs1 + inputs[:,0:num_freq_bin//2,:,:]
            inputs2 = inputs2 + inputs[:,num_freq_bin//2:,:,:]
            

            # SCALE
            for i in range(inputs1.shape[0]):
                inputs1[i,:,:,:] = (inputs1[i,:,:,:].clone() - torch.min(inputs1[i,:,:,:].clone()))/(torch.max(inputs1[i,:,:,:].clone())-torch.min(inputs1[i,:,:,:].clone()))
                inputs2[i,:,:,:] = (inputs2[i,:,:,:].clone() - torch.min(inputs2[i,:,:,:].clone()))/(torch.max(inputs2[i,:,:,:].clone())-torch.min(inputs2[i,:,:,:].clone()))


            inputs_deltas1 = deltas(inputs1)
            inputs_deltas_deltas1 = deltas(inputs_deltas1)
            inputs1 = torch.cat((inputs1[:,:,4:-4,:],inputs_deltas1[:,:,2:-2,:],inputs_deltas_deltas1),3)

        
            inputs_deltas2 = deltas(inputs2)
            inputs_deltas_deltas2 = deltas(inputs_deltas2)
            inputs2 = torch.cat((inputs2[:,:,4:-4,:],inputs_deltas2[:,:,2:-2,:],inputs_deltas_deltas2),3)
    
                            
            # STACK THE TWO WEIGHTED SPECTROGRAMS AND THEIR DELTAS ALONG THE SPLITTED DIM
            inputs = torch.cat((inputs1,inputs2),1)
            

        
            assert inputs.shape[3] == 3
            inputs = inputs.permute(0,3,1,2)  # change to N,C,H,W 
            assert inputs.shape[1] == 3 

        
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
        
    acc = 100. * correct/total
    if epoch == start_epoch + num_epochs -1 or acc > best_acc or (epoch - last_saved_epoch)>=5:   # save when best acc or every 5 epoch
        save_checkpoint(acc, epoch)
        last_saved_epoch = epoch
    if acc > best_acc:
        best_acc = acc
    #print(best_acc)

    return (test_loss/(batch_idx+1), 100.*correct/total)

def save_checkpoint(acc, epoch):
    """ Save checkpoint"""
    state={
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'acfb_relWt_net1_state_dict': acfb_relWt_net1.state_dict(),
        'acfb_relWt_net2_state_dict': acfb_relWt_net2.state_dict(),
        'acc': acc,
        'best_acc': best_acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'scheduler_state_dict': scheduler.state_dict()

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
    print("LR updated to: {}".format(scheduler.get_last_lr())) 
    print('Best_acc: ',best_acc)
    print('last_saved_epoch: ', last_saved_epoch)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc.detach().numpy(), 
            test_loss, test_acc.detach().numpy()])



