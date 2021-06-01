import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
from scipy.io.wavfile import write
import time


overwrite = True

file_path = '/home/data/DCASE2020/TAU-urban-acoustic-scenes-2020-mobile-development/'

csv_file = '/home/debottamd/DCASE-Hu-fcnn_baseline/task1a/10class/evaluation_setup/fold1_train_full.csv'

output_path = './aug_wavfiles'

feature_type = 'wav'

sr = 44100
duration = 10
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()

t0 = time.time()
for i in range(len(wavpath)):
    
    stereo, fs = sound.read(file_path + wavpath[i], stop=duration*sr)
    noise = np.random.normal(0,1,len(stereo))
    augmented_data = np.where(stereo != 0.0, stereo.astype('float64') + 0.01 * noise, 0.0).astype(np.float32)
    stereo = augmented_data 
    

    cur_file_name = output_path + wavpath[i][5:-4] + '_noise.' + 'wav'
    #pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    write(cur_file_name, fs, stereo)
        
t = time.time()
print("Took {} secs".format(t-t0))        

