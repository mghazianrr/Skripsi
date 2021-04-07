# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:48:17 2020

@author: Ghazian
"""
import os
import mne

#delay = [120]
#time = [180, 120, 180, 120, 180, 120, 180]

delay = [150]
time = [140, 160, 140, 160, 140, 160, 140]

#delay2 = [120]
#time2 = [180, 120, 180, 120, 180, 120, 180]

#Jenis emosi dan stimulus
emo1 = ['sedih', 'tenang', 'takut', 'senang']
a = 'Z1'

#Nama file untuk dibuka
name = ['1', '2', '3', '4', '13', '20', '27']

#Fungsi preprocessing
def preprocessing(raw):
    raw = raw.filter(l_freq = 1, h_freq = 35, l_trans_bandwidth = 1, 
               h_trans_bandwidth = 1, fir_design='firwin')
    return raw   

#Fungsi pembagi file berdasarkan emosi dan stimulus
def split_emotion (delay, time, stim, emo):
    for i in range (len(name)):
        fname = name[i]
        path = os.path.join (fname + '.BDF')
        raw = mne.io.read_raw_bdf(path, preload = True)
        raw = preprocessing(raw)
        raw.drop_channels(['Add_lead1', 'Add_lead2'])
        a = 0
        b = 0
        fid = 0
        for index in range (len(time)):
            b = b + time[index]
            if index % 2 == 0:
                save_path = os.path.join ('C:\MNE\Data\Ground Truth\Emosi Cut', emo[fid], fname+'_' + '_raw.fif')  
                raw.save(save_path, tmin = delay[0] + a, tmax = delay[0] + b, 
                         overwrite = True)
                fid = fid + 1
            a = b
split_emotion (delay, time, a, emo1)