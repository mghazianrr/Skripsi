# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:48:17 2020

@author: Ghazian
"""
import os
import mne

#def split_emotion ():

delay = [120]
time = [180, 120, 180, 120, 180, 120, 180]

#delay2 = [120]
#time2 = [180, 120, 180, 120, 180, 120, 180]

#ind = ['1', '0', '2', '3']
emo1 = ['sedih', 'tenang', 'takut', 'senang']
emo2 = ['takut', 'sedih', 'senang', 'tenang']
emo3 = ['tenang', 'takut', 'senang', 'sedih']

a = 'Z1' 
b = 'Z2'
c = 'Z3'

def preprocessing(raw):                                                        #definisikan preprocesing
    raw.filter(l_freq = 0.5, h_freq = 35.5, l_trans_bandwidth = 0.5, 
               h_trans_bandwidth = 0.5, fir_design='firwin')
    return raw   
                                                                               #kembalikan raw
def split_emotion (delay, time, stim, emo):
    for i in range (4):                                                        #jumlah data berapa?
        fname = str(i+1)
        path = os.path.join (stim, fname + '.BDF')                             #mengulang file dalam 1 folder
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
                save_path = os.path.join ('Emosi', emo[fid], fname+'_' + stim + '_raw.fif')  
                raw.save(save_path, tmin = delay[0] + a, tmax = delay[0] + b, 
                         overwrite = True)
                fid = fid + 1
            a = b
    #        return channel
split_emotion (delay, time, a, emo1)
split_emotion (delay, time, b, emo2)