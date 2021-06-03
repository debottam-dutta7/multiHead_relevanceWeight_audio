import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
from itertools import islice
import random
from scipy.io.wavfile import write


overwrite = True

# Path to audio files of the format: file_path + 'audio/' + 'example.wav'
file_path = '/home/data/DCASE2020/TAU-urban-acoustic-scenes-2020-mobile-development/'
csv_file = '../asc_setup_files/fold1_train_full.csv'
output_path = './aug_wavfiles'

feature_type = 'wav'
folder_name = "/home/data/DCASE2020/TAU-urban-acoustic-scenes-2020-mobile-development/audio/"

sr = 44100
duration = 10
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep='\t', encoding='ASCII')
wavpath = data_df['filename'].tolist()
label_dict = dict(airport=0, bus=1, metro=2, metro_station=3, park=4, public_square=5, shopping_mall=6, street_pedestrian=7, street_traffic=8, tram=9)


def class_sort():
    class_list = []
    for i in range(10):
        ap = []
        class_list.append(ap)
    with open(csv_file, 'r') as csv_r:
        # reader = csv.reader(csv_r)
        for line in islice(csv_r, 1, None):
            file_name = line.split('\t')[0].split('/')[1] #  'audio/' removed
            label = line.split('\t')[1].split('\n')[0]
            class_list[label_dict[label]].append(file_name)

    return class_list


def data_add():
    sample_rate = 44100
    class_list = class_sort()
    for label in class_list:
        length = len(label)
        print(length)
        for file in label:
            y, sr = librosa.load(folder_name + file, mono=True, sr=sample_rate)
            num = random.randint(0, length - 1)
            while file == label[num]:
                num = random.randint(0, length - 1)
            f1, f2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
            y2, _ = librosa.load(folder_name + label[num], mono=True, sr=sample_rate)
            stereo = y * f1 + y2 * f2

            cur_file_name = output_path + '/' + file.split('.')[0] + '_add.' + 'wav'

            write(cur_file_name, sample_rate, stereo)



if __name__ == "__main__":
    data_add()
        
        

