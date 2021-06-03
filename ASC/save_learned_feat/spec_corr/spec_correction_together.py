import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import pandas as pd
import librosa
import soundfile as sound
import random
from sklearn.preprocessing import StandardScaler


from utils_torch_learn_means import *
import pickle
import torch 
import torch.nn.functional as F
import scipy.io as sio
import scipy

#from Net_raw_AcFB_taslp_wo_RelWt import Net
from Net_learn_means_AcFB import AcFB

#file_path = 'data_2020/'
file_path = '/home/data/DCASE2020/TAU-urban-acoustic-scenes-2020-mobile-development/'
csv_file = '../../asc_setup_files/fold1_train.csv'
val_csv_file = '../../asc_setup_files/fold1_evaluate.csv'
device_a_csv = '../../asc_setup_files/fold1_train_a_2003.csv'

# Path to store the saved features
OUTPATH = '/home/debottamd/Hu-rep/learned-features-huBaseline-joint-f80' 

feature_type = 'mat'

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)

use_cuda = True


sr = 44100
num_audio_channels = 1
num_channel = 1
duration = 10

num_freq_bin = 80
win_length = 2048
num_fft = win_length
hop_length = 1024
num_time_bin = int(np.ceil(duration * sr / hop_length)) 



dev_train_df = pd.read_csv(csv_file,sep='\t', encoding='ASCII')
dev_val_df = pd.read_csv(val_csv_file,sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
wavpaths_val = dev_val_df['filename'].tolist()
y_train_labels =  dev_train_df['scene_label'].astype('category').cat.codes.values
y_val_labels =  dev_val_df['scene_label'].astype('category').cat.codes.values


trainf = [x[6:-4] for x in wavpaths_train]
train_subset = []

for idx in ['-a', '-b', '-c']:
    train_subset.append([x[:-1] for x in trainf if (x.endswith(idx))])

for idx in ['-s1', '-s2', '-s3']:
    train_subset.append([x[:-2] for x in trainf if (x.endswith(idx))])


train_sets=[]
for idx in range(len(train_subset)):
    train_sets.append(set(train_subset[idx]))

paired_wavs = []
# paired waves in subsets a, b, and c, s1, s2, and s3
for j in range(1, len(train_sets)):
    # paired waves in subsets b, c, and [s1, s2, s3] -- s4, s5, and s6 are not in the training partition of the data
    paired_wavs.append(train_sets[0] &  train_sets[j])

num_paired_wav = [ len(x) for x in paired_wavs]
min_paired_wav = 150

waves30 = []
wav_idxs = random.sample(range(min(num_paired_wav)), min_paired_wav)
for wavs in paired_wavs:
    temp = [list(wavs)[i] for i in wav_idxs]
    waves30.append(temp)


nbins_stft = int(np.ceil(num_fft/2.0)+1)
STFT_all = np.zeros((len(waves30)*min_paired_wav,nbins_stft,num_time_bin),'float32')
#STFT_all = np.zeros((min_paired_wav,nbins_stft,num_time_bin),'float32')
#STFT_sets = []
for group, x in zip(waves30, ['b', 'c', 's1','s2','s3']):
    i = 0
    for sc in group:
        wav_a = 'audio/' + sc + 'a.wav'
        wav_x = 'audio/' + sc + x + '.wav'
        stereo_a, fs = sound.read(file_path + wav_a, stop=duration * sr)
        stereo_x, fs = sound.read(file_path + wav_x, stop=duration * sr)
        # compute STFT of the paired signals
        STFT_a = librosa.stft(stereo_a, n_fft=num_fft, hop_length=hop_length)
        STFT_x = librosa.stft(stereo_x, n_fft=num_fft, hop_length=hop_length)
        # compute average value per each bin
        STFT_ref = np.abs(STFT_x)
        STFT_corr_coeff = STFT_ref/np.abs(STFT_a)
        # stack averaged values
        STFT_all[i,:,:] = STFT_corr_coeff
        i=i+1
    #STFT_sets.append(STFT_all)



STFT_hstak = np.hstack(STFT_all)
STFT_corr_coeff = np.expand_dims(np.mean(STFT_hstak,axis=1),-1)

data_df = pd.read_csv(device_a_csv, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()



""" Model """
#net = Net()
net = AcFB()
net.eval()



#device_list = ['b', 'c', 's1','s2','s3']
for d in range(1):
    for i in range(len(wavpath)):
        stereo, fs = sound.read(file_path + 'audio/'+ wavpath[i], stop=duration*sr)
        #stereo, fs = librosa.load(file_path + 'audio/'+ wavpath[i], sr=None)
        STFT = librosa.stft(stereo, n_fft=num_fft, hop_length=hop_length)
        
        STFT_corr = STFT * STFT_corr_coeff
        
        
        audio = librosa.istft(STFT_corr, hop_length=hop_length,win_length=win_length, length=len(stereo))
        audio = audio / (max(abs(audio)) + 1e-4)
        assert len(audio) == len(stereo)

        data_segments = enframe(audio, win_length, hop_length)
        inputs = torch.from_numpy(data_segments)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.unsqueeze(0)  
        inputs = inputs.permute(0,1,3,2)
        if use_cuda:
            inputs = inputs.float().cuda()
        
        outputs = net(inputs)
        outputs = torch.squeeze(outputs)
       

        filename = OUTPATH + '/'+ wavpath[i][0:-4] + '-all'  + '.mat'

        print('saving:',filename)
        sio.savemat(filename, mdict={'data':outputs.data.cpu().numpy()})

        
        
    



