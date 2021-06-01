import os
import sys
import time
import math
import torch
import numpy as np
import pandas as pd 
import pickle
import scipy.io as sio


def load_file(csv_path):
    """ Taken from load_data_2020 of utils.py for Hu baseline"""

    with open(csv_path,'r') as f:
        lines = f.read().split('\n')
        for idx,elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        # remove first line
        lines = lines[1:]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info = np.array(lines)

        data_df = pd.read_csv(csv_path,sep='\t',encoding='ASCII')
        ClassNames = np.unique(data_df['scene_label'])
        labels = data_df['scene_label'].astype('category').cat.codes.values
    return label_info, labels
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,feat_path,csv_path,file_type):
        self.feat_path = feat_path
        self.file_type = file_type
        self.csv_path = csv_path
        self.label_info, self.labels = load_file(self.csv_path)

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filepath = self.feat_path + '/' + self.label_info[idx][0] + '.mat'

        x = sio.loadmat(filepath)['data']
        target = self.labels[idx]

        return x, target
        
        
        



#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 70.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

