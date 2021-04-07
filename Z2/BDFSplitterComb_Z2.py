# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:48:17 2020

@author: Ghazian
"""
import os
import mne

#def split_emotion ():

#delay = [120]
#time = [180, 120, 180, 120, 180, 120, 180]

delay = [150]
time = [140, 160, 140, 160, 140, 160, 140]

#delay2 = [120]
#time2 = [180, 120, 180, 120, 180, 120, 180]

#ind = ['1', '0', '2', '3']
emo1 = ['takut', 'sedih', 'senang', 'tenang']


a = 'Z2' 
name = ['5', '6', '7', '8', '14', '17', '28']
def preprocessing(raw):                                                        #definisikan preprocesing
    raw.filter(l_freq = 1, h_freq = 35, l_trans_bandwidth = 1, 
               h_trans_bandwidth = 1, fir_design='firwin')
    return raw   
                                                                               #kembalikan raw
def split_emotion (delay, time, stim, emo):
    for i in range (7):                                                        #jumlah data berapa?
        fname = name[i]
        path = os.path.join (fname + '.BDF')                             #mengulang file dalam 1 folder
    #print (fname)
        raw = mne.io.read_raw_bdf(path, preload = True)                        #baca raw
        raw = preprocessing(raw)                                               #kembalikan hasil preprocessing ke raw
        raw.drop_channels(['Add_lead1', 'Add_lead2'])                          #mark bad channels, drop channels
    
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
    #        return channel
split_emotion (delay, time, a, emo1)