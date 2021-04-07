# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 09:23:53 2020

@author: Ghazian
"""
import os
import mne

jml = 27
for i in range (jml):
    i = str(i + 1)
    path = os.path.join (i + '__raw.fif')
    raw = mne.io.read_raw_fif(path, preload = True)
    start_seconds = 0 #detik mulai
    run_time = raw.times # list waktu data
    stop_seconds = run_time[-1] #waktu data terakhir
    interval = (stop_seconds-start_seconds)/4
    a = 0
    b = 0
    fid = 0
    for index in range(4):
        b = b + interval
        save_path = os.path.join ('Segmented', i + str(index) + '_raw.fif')  
        raw.save(save_path, tmin = a, tmax = b, overwrite = True)
        a = b