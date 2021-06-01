import numpy as np
import torch
from torch.autograd import Variable

import glob
import sys
import os
import time
import random
import scipy.io as sio
import scipy
import torch
import torch.nn.functional as F

from Net_CNN2D_LSTM_splice10_cuda import Net

OUTDIR = './classifier_models_10foldCV/'

def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '=' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def normalize(features):
    mean = features.mean(axis=0)
    # mean = (mean.mean(axis=0))
    var = features.std(axis=0)
    # var = (var.std(axis=0))
    return mean, var


def build_generator(param):
    if param['G_type'] == 'raw_classifier':
        # return VAE(param=param)
        return Net()
    else:
        print('Unkown Generator Type')
        return None


def num_of_params(module):
    total_params = 0

    for param in module.parameters():
        temp = 1
        for s in param.size():
            temp *= s
        total_params += temp
    return total_params


def adjust_learning_rate(optimizer, lr):
    """Updates learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def enframe(x, winlen, hoplen):
    '''
    receives a 1D numpy array and divides it into frames.
    outputs a numpy matrix with the frames on the rows.
    '''
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + np.int(np.floor((len(x) - winlen) / float(hoplen)))
    xf = np.zeros((n_frames, winlen))
    for ii in range(n_frames):
        xf[ii] = x[ii * hoplen : ii * hoplen + winlen]
    return xf


def extract_segments(wavfiles, labelfiles, hop_length=int(0.010*44100), win_length=int(0.025*44100), n_bands = 80, splice=25):
    segments_all = []
    labels_all = []
    hop_length=int(0.010*44100)
    win_length=int(0.025*44100)
    for i in range(np.alen(labelfiles)):
        fn = labelfiles[i]
        file_name = fn.split('.')[0]
        feat_path = ['/data1/MSR_IL_2018/purvia/Purvi_VAE_rawWav/Purvi_gaussKernel_UrbanSound8k/raw_features_save/UrbanSound8k/' + file_name + '.mat']
        feat_path_str = ''.join(feat_path)
        data_segments = torch.from_numpy(sio.loadmat(feat_path_str)['data']).permute(1,2,0,3)
        num_frames = data_segments.shape[2]
        data_segments = F.unfold(data_segments, (2*splice+1, n_bands), padding=(splice,0))
        data_segments = data_segments.permute(0,2,1).reshape(1,num_frames,2*splice+1,n_bands).permute(0,1,3,2).squeeze().numpy()

        labels = np.array(fn.split('-')[1])
        data_segments = data_segments[24:num_frames-24, :, :]
        segments_all.append(data_segments)
        labels_all.append(labels)

    segments_all = np.asarray(segments_all)
    labels_all = np.asarray(labels_all)
    mat_data = []
    mat_label = []
    for i in range(np.alen(segments_all)):
        num_frames = np.alen(segments_all[i])
        for j in range(num_frames):
            mat_data.append(segments_all[i][j])
            # mat_label.append(labels_all[i])
    
    for i in range(np.alen(labels_all)):
        num_frames = np.alen(segments_all[i])
        for j in range(num_frames):
            mat_label.append(labels_all[i])
  
    return np.array(mat_data), np.array(mat_label)

def batch_generator(inp_features, tar_features, param, minibatch_index):
    num_of_total_segments = np.size(inp_features, axis=0)
    data_inp = np.ndarray(shape=(param['batch_size'],  np.size(inp_features, axis=1), np.size(inp_features, axis=2), np.size(inp_features, axis=3)), dtype=np.float32)
    data_tar = np.ndarray(shape=(param['batch_size']), dtype=np.int64)
    for i in range(param['batch_size']):
        # temp = random.randint(1, num_of_total_segments)-1
        temp = minibatch_index * param['batch_size'] + i
        data_inp[i] = (inp_features[temp, :, :, :])
        data_tar[i] = (tar_features[temp])

    # Convert to torch tensors and share memory before returning
    return torch.from_numpy(data_inp).share_memory_(), torch.from_numpy(data_tar).share_memory_()


def split_data_train_val(inp_features, inp_targets, train_percent=0.9):
    num_features = np.size(inp_features, axis=0)
    shuf_indices = np.random.permutation(np.arange(num_features))
    # inp_features_shuf = np.random.permutation(inp_features)
    inp_features_shuf = inp_features[shuf_indices[0:int(np.floor(num_features*train_percent))], :]
    inp_targets_shuf = inp_targets[shuf_indices[0:int(np.floor(num_features*train_percent))]]
    return inp_features_shuf, inp_targets_shuf


def main(param):
    G_net = build_generator(param=param)
    
    # Load pre-trained weights if available
    if param['model_G_net'] is not None:
        print ('Loading param file: {}'.format(param['model_G_net']))
        G_net.load_state_dict(torch.load(param['model_G_net']))

    reference = param['type'] + str(time.strftime("_%Y%m%d-%H%M%S"))

    N_param_G = num_of_params(G_net)
    print ('Total number of parameters: G:{}'.format(N_param_G))
    CE_criterion = torch.nn.CrossEntropyLoss()
    if param['cuda']:
        G_net.cuda()
        CE_criterion.cuda()

    G_solver = torch.optim.Adam(G_net.parameters(), lr=param['lr_G'])
    training_error_G_CE = np.zeros(shape=(param['num_epochs'],))
    training_error_G = np.zeros(shape=(param['num_epochs'],))

    training_error_G_CE_val = np.zeros(shape=(param['num_epochs'],))
    training_error_G_val = np.zeros(shape=(param['num_epochs'],))

    epoch = 0
    current_lr_G = param['lr_G']

    inList_full = '/home/purvia/purvia/extra/UrbanSound_classification/UrbanSound8K_rewrite/UrbanSound8k_list_I2-10-E1_shuf.txt'  
    with open(inList_full) as fin_full:
        infiles_full = fin_full.read().splitlines()

    num_files_load = param['files_load_split']
    num_split = int(len(infiles_full)/num_files_load)

    while epoch < param['num_epochs']:
        print('epoch no.:', epoch)
        start_time = time.time()

        # For accuracy
        correct = 0
        total = 0
        num_classes = 10
        confusion_matrix = torch.zeros(num_classes, num_classes)

        # For validation accuracy
        correct_val = 0
        total_val = 0

        for i in range(num_split):
            if (i+1)*num_files_load <= len(infiles_full):
                infiles = infiles_full[i*num_files_load:(i+1)*num_files_load]
            else: 
                infiles = infiles_full[i*num_files_load:len(infiles_full)]

            tarfiles = infiles

            inp_features, inp_labels = extract_segments(infiles, tarfiles, win_length=param['win_length'], hop_length=param['hop_len'])
        
            mean, var = normalize(inp_features)
            statistics = {'mean': mean, 'var': var}
            inp_features = (inp_features - statistics['mean']) / (statistics['var'] + np.finfo(np.float32).eps)

            inp_features, inp_labels = split_data_train_val(inp_features, inp_labels, train_percent=1)

            inp_features = np.expand_dims(inp_features, axis=1)
            #print('inp_features, inp_labels ', np.shape(inp_features), np.shape(inp_labels))

            num_minibatches_per_epoch = np.size(inp_features, axis=0) // param['batch_size']
            
        
            minibatch_process_time = 0
            G_net.train()
            minibatches_seen = 0

            while minibatches_seen < num_minibatches_per_epoch:
                inputs, targets = batch_generator(inp_features, inp_labels, param, minibatches_seen)
                mbatch_time = time.time()

                # Send to GPU
                if param['cuda']:
                    inputs = inputs.float().cuda()
                    targets = targets.cuda()

                # train generator
                G_net.zero_grad()
                fake = G_net(Variable(inputs))
                errCE = CE_criterion(fake, Variable(targets))
                errG = 1.0 * errCE
                errG.backward()
                G_solver.step()

                training_error_G_CE[epoch] += errCE.item()
                training_error_G[epoch] += errG.item()
                minibatches_seen += 1
                print_progress(minibatches_seen, num_minibatches_per_epoch*num_split)

                with torch.no_grad():
                    _, predicted = torch.max(fake.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
        # VALIDATION
        
        print("Computing validation loss:\t\t ")
        if param['validate']:
            inList = '/home/purvia/purvia/extra/UrbanSound_classification/UrbanSound8K_rewrite/UrbanSound8k_list_fold1.txt'
            with open(inList) as fin:
                infiles = fin.read().splitlines()

                tarfiles = infiles

            val_features, val_labels = extract_segments(infiles, tarfiles, win_length=param['win_length'], hop_length=param['hop_len'])

            mean, var = normalize(val_features)
            statistics = {'mean': mean, 'var': var}
            val_features = (val_features - statistics['mean']) / (statistics['var'] + np.finfo(np.float32).eps)

            val_features = np.expand_dims(val_features, axis=1)

            G_net.eval()  # Set model to evaluate mode
            minibatches_seen_val = 0

            num_minibatches_per_epoch_val = np.size(val_features, axis=0) // param['batch_size']
            # num_minibatches_per_epoch_val = 5 # debug
            while minibatches_seen_val < num_minibatches_per_epoch_val:
                val_inputs, val_targets = batch_generator(val_features, val_labels, param, minibatches_seen_val)
                if param['cuda']:
                    val_inputs = val_inputs.float().cuda()
                    val_targets = val_targets.cuda()

                val_output = G_net(Variable(val_inputs))
                errCE_val = CE_criterion(val_output, Variable(val_targets))
                errG_val = 1.0 * errCE_val
                training_error_G_CE_val[epoch] += errCE_val.item()
                training_error_G_val[epoch] += errG_val.item()
                minibatches_seen_val += 1
    
                with torch.no_grad():
                    _, predicted_val = torch.max(val_output.data, 1)
                    total_val += val_targets.size(0)
                    correct_val += (predicted_val == val_targets).sum().item()
                    for t, p in zip(val_targets.view(-1), predicted_val.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                          
        training_error_G_CE_val[epoch] /= num_minibatches_per_epoch_val
        training_error_G_val[epoch] /= num_minibatches_per_epoch_val
        print('Done validation\t\t')
            
        print('Training Accuracy of the network: %d %%' % ( 100 * correct / total))

        print('Validation Accuracy of the network: %d %%' % ( 100 * correct_val / total_val))

        training_error_G_CE[epoch] /= num_minibatches_per_epoch * num_split
        training_error_G[epoch] /= num_minibatches_per_epoch * num_split
        time_for_1_epoch = time.time() - start_time

        # Print Training Error every epoch
        print("Epoch {} of {} took {:.3f}s (training time {:.2f}s, Data prep {:.2f}s".format(epoch + 1, param['num_epochs'], time_for_1_epoch, minibatch_process_time,
        time_for_1_epoch - minibatch_process_time))
        print("training loss:\t\tCE:{:.4f} \t\tG: {:.4f}".format(training_error_G_CE[epoch], training_error_G[epoch]))
        
        if param['validate']:
            print("validation loss:\t\tCE_val:{:.4f} \t\tG_val: {:.4f}".format(training_error_G_CE_val[epoch], training_error_G_val[epoch]))

        sys.stdout.flush()
        print('Saving current model ...')
        if not os.path.exists(OUTDIR + reference):
            os.makedirs(OUTDIR + reference)
        model_path_G = OUTDIR + reference + '/model_G_' + str(epoch) + '.pth'
        model_path_val = OUTDIR + reference

        if (epoch + 1) % 1 == 0:
            torch.save(G_net.state_dict(), model_path_G)
            sio.savemat(model_path_val + '/confusion_matrix_CNN_LSTM_splice25_' + str(epoch) + '.mat', mdict={'data': confusion_matrix.numpy()})

        # Update learning late as per learning rate schedule
        if (epoch + 1) % param['LearningRateEpoch_G'] == 0 and epoch != 0:
            if current_lr_G >= 1e-5:
                current_lr_G *= 0.1
                adjust_learning_rate(G_solver, (current_lr_G))
                print ('===================Updated G Learning Rate to {}====================='.format(current_lr_G))

        # Update Epoch
        epoch += 1
        sio.savemat(OUTDIR + reference + '/training_error_G_CE.mat', mdict={'data': training_error_G_CE})
        sio.savemat(OUTDIR + reference + '/training_error_G.mat', mdict={'data': training_error_G})

        if param['validate']:
            sio.savemat(OUTDIR + reference + '/training_error_G_CE_val.mat', mdict={'data': training_error_G_CE_val})
            sio.savemat(OUTDIR + reference + '/training_error_G_val.mat', mdict={'data': training_error_G_val})

    return model_path_G


if __name__ == '__main__':
    Param = {
        'type': 'raw_gaussianKernel_CNN_LSTM_Fs44_pl51_T1_classifier',
        'hop_len': int(10*44.1),
        'win_length': int(25*44.1),
        'G_type': 'raw_classifier',
        'num_classes': 10,
        'lr_G': 0.001,
        'LearningRateEpoch_G': 10,
        'num_epochs': 30,
        'cuda': True,
        'batch_size': 32,
        'validate': True,
        'files_load_split': 55,
        'model_G_net': None,
    }

model_pah_G = main(param=Param)
