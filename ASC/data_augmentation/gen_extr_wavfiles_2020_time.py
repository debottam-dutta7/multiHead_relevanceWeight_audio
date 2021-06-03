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

# Path to audio files of the format: file_path + 'audio/' + 'example.wav'
file_path = '/home/data/DCASE2020/TAU-urban-acoustic-scenes-2020-mobile-development/'
csv_file = '../asc_setup_files/fold1_train_full.csv'
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
    time_factor = np.random.uniform(0.5, 2)
    length = len(stereo)
    y_stretch = librosa.effects.time_stretch(stereo, time_factor)
    if len(y_stretch) < length:
        y_stretch = np.concatenate((y_stretch, y_stretch))
        y_stretch = y_stretch[0:length]
    else:
        y_stretch = y_stretch[0:length]
    stereo = y_stretch

    cur_file_name = output_path + wavpath[i][5:-4] + '_time.' + feature_type

    write(cur_file_name, fs, stereo)
    print("{} th file done.".format(i))

t = time.time()
print("Took {} secs".format(t-t0)) 
        
        

