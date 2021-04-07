import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.integrate import simps
from scipy.signal import find_peaks, argrelmax
from sklearn import svm
import pandas as pd
import mne
import csv
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, 
                               corrmap)

Fp1_delta_psd = []
Fp1_theta_psd = []
Fp1_alpha_psd = []
Fp1_beta_psd = []
Fp1_gamma_psd = []

Fp2_delta_psd = []
Fp2_theta_psd = []
Fp2_alpha_psd = []
Fp2_beta_psd = []
Fp2_gamma_psd = []

F3_delta_psd = []
F3_theta_psd = []
F3_alpha_psd = []
F3_beta_psd = []
F3_gamma_psd = []

F4_delta_psd = []
F4_theta_psd = []
F4_alpha_psd = []
F4_beta_psd = []
F4_gamma_psd = []

C3_delta_psd = []
C3_theta_psd = []
C3_alpha_psd = []
C3_beta_psd = []
C3_gamma_psd = []

C4_delta_psd = []
C4_theta_psd = []
C4_alpha_psd = []
C4_beta_psd = []
C4_gamma_psd = []

P3_delta_psd = []
P3_theta_psd = []
P3_alpha_psd = []
P3_beta_psd = []
P3_gamma_psd = []

P4_delta_psd = []
P4_theta_psd = []
P4_alpha_psd = []
P4_beta_psd = []
P4_gamma_psd = []

O1_delta_psd = []
O1_theta_psd = []
O1_alpha_psd = []
O1_beta_psd = []
O1_gamma_psd = []

O2_delta_psd = []
O2_theta_psd = []
O2_alpha_psd = []
O2_beta_psd = []
O2_gamma_psd = []

F7_delta_psd = []
F7_theta_psd = []
F7_alpha_psd = []
F7_beta_psd = []
F7_gamma_psd = []

F8_delta_psd = []
F8_theta_psd = []
F8_alpha_psd = []
F8_beta_psd = []
F8_gamma_psd = []

T3_delta_psd = []
T3_theta_psd = []
T3_alpha_psd = []
T3_beta_psd = []
T3_gamma_psd = []

T4_delta_psd = []
T4_theta_psd = []
T4_alpha_psd = []
T4_beta_psd = []
T4_gamma_psd = []

T5_delta_psd = []
T5_theta_psd = []
T5_alpha_psd = []
T5_beta_psd = []
T5_gamma_psd = []

T6_delta_psd = []
T6_theta_psd = []
T6_alpha_psd = []
T6_beta_psd = []
T6_gamma_psd = []

Fp1_delta_e = []
Fp1_theta_e = []
Fp1_alpha_e = []
Fp1_beta_e = []
Fp1_gamma_e = []

Fp2_delta_e = []
Fp2_theta_e = []
Fp2_alpha_e = []
Fp2_beta_e = []
Fp2_gamma_e = []

F3_delta_e = []
F3_theta_e = []
F3_alpha_e = []
F3_beta_e = []
F3_gamma_e = []

F4_delta_e = []
F4_theta_e = []
F4_alpha_e = []
F4_beta_e = []
F4_gamma_e = []

C3_delta_e = []
C3_theta_e = []
C3_alpha_e = []
C3_beta_e = []
C3_gamma_e = []

C4_delta_e = []
C4_theta_e = []
C4_alpha_e = []
C4_beta_e = []
C4_gamma_e = []

P3_delta_e = []
P3_theta_e = []
P3_alpha_e = []
P3_beta_e = []
P3_gamma_e = []

P4_delta_e = []
P4_theta_e = []
P4_alpha_e = []
P4_beta_e = []
P4_gamma_e = []

O1_delta_e = []
O1_theta_e = []
O1_alpha_e = []
O1_beta_e = []
O1_gamma_e = []

O2_delta_e = []
O2_theta_e = []
O2_alpha_e = []
O2_beta_e = []
O2_gamma_e = []

F7_delta_e = []
F7_theta_e = []
F7_alpha_e = []
F7_beta_e = []
F7_gamma_e = []

F8_delta_e = []
F8_theta_e = []
F8_alpha_e = []
F8_beta_e = []
F8_gamma_e = []

T3_delta_e = []
T3_theta_e = []
T3_alpha_e = []
T3_beta_e = []
T3_gamma_e = []

T4_delta_e = []
T4_theta_e = []
T4_alpha_e = []
T4_beta_e = []
T4_gamma_e = []

T5_delta_e = []
T5_theta_e = []
T5_alpha_e = []
T5_beta_e = []
T5_gamma_e = []

T6_delta_e = []
T6_theta_e = []
T6_alpha_e = []
T6_beta_e = []
T6_gamma_e = []

Fp1 = []
Fp2 = []
F3 = []
F4 = []
C3 = []
C4 = []
P3 = []
P4 = []
O1 = []
O2 = []
F7 = []
F8 = []
T3 = []
T4 = []
T5 = []
T6 = []

Ch = ['Fp1-A1', 'Fp2-A2', 'F3-A1', 'F4-A2', 'C3-A1', 'C4-A2', 'P3-A1', 'P4-A2', 
      'O1-A1', 'O2-A2', 'F7-A1', 'F8-A2', 'T3-A1', 'T4-A2', 'T5-A1', 'T6-A2']

ChN = [Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, 
       F8, T3, T4, T5, T6]

psd_var = [Fp1_delta_psd, Fp1_theta_psd, Fp1_alpha_psd,	Fp1_beta_psd,          #
           Fp1_gamma_psd, Fp2_delta_psd, Fp2_theta_psd, Fp2_alpha_psd, 
           Fp2_beta_psd, Fp2_gamma_psd, F3_delta_psd, F3_theta_psd, 
           F3_alpha_psd, F3_beta_psd, F3_gamma_psd, F4_delta_psd, 
           F4_theta_psd, F4_alpha_psd, F4_beta_psd, F4_gamma_psd, 
           C3_delta_psd, C3_theta_psd, C3_alpha_psd, C3_beta_psd, 
           C3_gamma_psd, C4_delta_psd, C4_theta_psd, C4_alpha_psd, 
           C4_beta_psd, C4_gamma_psd, P3_delta_psd, P3_theta_psd, 
           P3_alpha_psd, P3_beta_psd, P3_gamma_psd, P4_delta_psd, P4_theta_psd,
           P4_alpha_psd, P4_beta_psd, P4_gamma_psd, O1_delta_psd, O1_theta_psd, 
           O1_alpha_psd, O1_beta_psd, O1_gamma_psd, O2_delta_psd, O2_theta_psd, 
           O2_alpha_psd, O2_beta_psd, O2_gamma_psd, F7_delta_psd, F7_theta_psd, 
           F7_alpha_psd, F7_beta_psd, F7_gamma_psd, F8_delta_psd, F8_theta_psd, 
           F8_alpha_psd, F8_beta_psd, F8_gamma_psd, T3_delta_psd, T3_theta_psd, 
           T3_alpha_psd, T3_beta_psd, T3_gamma_psd, T4_delta_psd, T4_theta_psd, 
           T4_alpha_psd, T4_beta_psd, T4_gamma_psd, T5_delta_psd, T5_theta_psd,	
           T5_alpha_psd, T5_beta_psd, T5_gamma_psd, T6_delta_psd, T6_theta_psd, 
           T6_alpha_psd, T6_beta_psd, T6_gamma_psd]

e_var = [Fp1_delta_e, Fp1_theta_e, Fp1_alpha_e, Fp1_beta_e, 
          Fp1_gamma_e, Fp2_delta_e, Fp2_theta_e, Fp2_alpha_e, 
          Fp2_beta_e, Fp2_gamma_e, F3_delta_e, F3_theta_e, 
          F3_alpha_e, F3_beta_e, F3_gamma_e, F4_delta_e, 
          F4_theta_e, F4_alpha_e, F4_beta_e, F4_gamma_e, 
          C3_delta_e, C3_theta_e, C3_alpha_e, C3_beta_e, 
          C3_gamma_e, C4_delta_e, C4_theta_e, C4_alpha_e, 
          C4_beta_e, C4_gamma_e, P3_delta_e, P3_theta_e, 
          P3_alpha_e, P3_beta_e, P3_gamma_e, P4_delta_e, 
          P4_theta_e, P4_alpha_e, P4_beta_e, P4_gamma_e, 
          O1_delta_e, O1_theta_e, O1_alpha_e, O1_beta_e, 
          O1_gamma_e, O2_delta_e, O2_theta_e, O2_alpha_e, 
          O2_beta_e, O2_gamma_e, F7_delta_e, F7_theta_e, 
          F7_alpha_e, F7_beta_e, F7_gamma_e, F8_delta_e, 
          F8_theta_e, F8_alpha_e, F8_beta_e, F8_gamma_e, 
          T3_delta_e, T3_theta_e, T3_alpha_e, T3_beta_e, 
          T3_gamma_e, T4_delta_e, T4_theta_e, T4_alpha_e, 
          T4_beta_e, T4_gamma_e, T5_delta_e, T5_theta_e, T5_alpha_e, 
          T5_beta_e, T5_gamma_e, T6_delta_e, T6_theta_e, T6_alpha_e, 
          T6_beta_e, T6_gamma_e]

p_var = [Fp1_delta_p, Fp1_theta_p, Fp1_alpha_p, Fp1_beta_p, 
          Fp1_gamma_p, Fp2_delta_p, Fp2_theta_p, Fp2_alpha_p, 
          Fp2_beta_p, Fp2_gamma_p, F3_delta_p, F3_theta_p, 
          F3_alpha_p, F3_beta_p, F3_gamma_p, F4_delta_p, 
          F4_theta_p, F4_alpha_p, F4_beta_p, F4_gamma_p, 
          C3_delta_p, C3_theta_p, C3_alpha_p, C3_beta_p, 
          C3_gamma_p, C4_delta_p, C4_theta_p, C4_alpha_p, 
          C4_beta_p, C4_gamma_p, P3_delta_p, P3_theta_p, 
          P3_alpha_p, P3_beta_p, P3_gamma_p, P4_delta_p, 
          P4_theta_p, P4_alpha_p, P4_beta_p, P4_gamma_p, 
          O1_delta_p, O1_theta_p, O1_alpha_p, O1_beta_p, 
          O1_gamma_p, O2_delta_p, O2_theta_p, O2_alpha_p, 
          O2_beta_p, O2_gamma_p, F7_delta_p, F7_theta_p, 
          F7_alpha_p, F7_beta_p, F7_gamma_p, F8_delta_p, 
          F8_theta_p, F8_alpha_p, F8_beta_p, F8_gamma_p, 
          T3_delta_p, T3_theta_p, T3_alpha_p, T3_beta_p, 
          T3_gamma_p, T4_delta_p, T4_theta_p, T4_alpha_p, 
          T4_beta_p, T4_gamma_p, T5_delta_p, T5_theta_p, T5_alpha_p, 
          T5_beta_p, T5_gamma_p, T6_delta_p, T6_theta_p, T6_alpha_p, 
          T6_beta_p, T6_gamma_p]


dasmi12_var = [Fp12_delta, Fp12_theta, Fp12_alpha, Fp12_beta, Fp12_gamma,
               F34_delta, F34_theta, F34_alpha, F34_beta, F34_gamma,
               C34_delta, C34_theta, C34_alpha, C34_beta, C34_gamma,
               P34_delta, P34_theta, P34_alpha, P34_beta, P34_gamma,
               O12_delta, O12_theta, O12_alpha, O12_beta, O12_gamma,
               F78_delta, F78_theta, F78_alpha, F78_beta, F78_gamma,
               T34_delta, T34_theta, T34_alpha, T34_beta, T34_gamma,
               T56_delta, T56_theta, T56_alpha, T56_beta, T56_gamma,]

freq_range = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 35)]

def preprocessing(raw): #definisikan preprocesing
#    tapis = mne.filter.create_filter(raw.get_data()[0], sampling_freq,
#                                                  l_freq = 0.1, 
#                                                  h_freq = 49.5, 
#                                                  l_trans_bandwidth = 0.1, 
#                                                  h_trans_bandwidth = 0.5, 
#                                                  fir_design='firwin')
#   raw = np.convolve(tapis, raw)
    raw.filter(l_freq = 0.5, h_freq = 35.5, l_trans_bandwidth = 0.5, 
               h_trans_bandwidth = 0.5, fir_design='firwin')
    return raw #kembalikan raw

def separate (ChName): #definisikan fungsi pemisah per kanal
    channel = raw[ChName, start_sample:stop_sample] #pisahkan per kanal
    return channel #kembalikan channel
    print (channel)

def feature_extraction (n, low, high):
    win = 4 * sampling_freq
    f_values, psd_values = signal.welch(n, fs = sampling_freq, nperseg = win, 
                                        scaling='density')
#    _, e_values = signal.welch(n, fs = sampling_freq, nperseg = win,
#                               scaling='spectrum')
    idx_delta = np.logical_and(f_values >= low, f_values <= high)
    freq_res = f_values[1] - f_values[0]
    psd = simps(psd_values[idx_delta], dx = freq_res)
    f_fft = np.fft.fftfreq(len(n), d = 1/100)
    df_fft = f_fft[1] - f_fft[0]
    E_welch = ((1. / (1/100) * (freq_res / df_fft) * 
                np.sum(psd_values[idx_delta])))
#    energy = sum(abs(n)**2.0)/sampling_freq
#    print (f_values)
#    print (psd_values)
#    print (energy)
#    puncak, _ = find_peaks (psd_values)
#    print (puncak)
#    for p in puncak:
#        x = psd_values[p]
#        y = 0
#        if x > y:
#            y == x
#            f = f_values[p]
#    print(f)
#    print(p)
#    print(energy)
    return psd, E_welch
    
for i in range (3):                                                            #jumlah data berapa?
    fname = os.path.join ('Data EEG Tobias', 'Buka tutup mata', str(i+1)+'.BDF') #mengulang file dalam 1 folder
#print (fname)
    raw = mne.io.read_raw_fif(fname, preload = True)               #baca raw
    info = raw.info                                                            #dapatkan info
    raw.drop_channels(['Add_lead1', 'Add_lead2']) #mark bad channels, drop channels
    #raw.set_eeg_reference(ref_channels='average', projection=False, 
    #                      verbose=None) #set elektroda referensi, ubah average

    sampling_freq = raw.info['sfreq'] #frekuensi sampling
    start_seconds = 0 #detik mulai
    run_time = raw.times # list waktu data
    stop_seconds = run_time[-1] #waktu data terakhir
    start_stop_seconds = np.array([start_seconds, stop_seconds]) #buat array mulai dan selesai
    start_sample, stop_sample = (start_stop_seconds*sampling_freq).astype(int) #sampel ke n mulai dan selesai

#data = raw.get_data()
    raw = preprocessing(raw) #kembalikan hasil preprocessing ke raw
#print(info.keys()) #tampilkan info "keys"
#raw.plot_psd(tmax=np.inf, picks=None, average=True) #plot psd

#ch = info['ch_names'] #info nama chanel

    m = 0
    for a in Ch:
        channel_data = separate(a)[0].flatten()
        ChN[m].append(channel_data)
#    print (ChN[m])
        m = m + 1

#    if m == 15:
#        m = 0
        
#Fp1 = separate('Fp1-A1')[0].flatten()
#Fp2 = separate('Fp2-A2')[0].flatten()
#F3 = separate('F3-A1')[0].flatten()
#F4 = separate('F4-A2')[0].flatten()
#C3 = separate('C3-A1')[0].flatten()
#C4 = separate('C4-A2')[0].flatten()
#P3 = separate('P3-A1')[0].flatten()
#P4 = separate('P4-A2')[0].flatten()
#O1 = separate('O1-A1')[0].flatten()
#O2 = separate('O2-A2')[0].flatten()
#F7 = separate('F7-A1')[0].flatten()
#F8 = separate('F8-A2')[0].flatten()
#T3 = separate('T3-A1')[0].flatten()
#T4 = separate('T4-A2')[0].flatten()
#T5 = separate('T5-A1')[0].flatten()
#T6 = separate('T6-A2')[0].flatten()

#    print (Fp1)
    
    n = 0
    o = 0
    for i in ChN:
        for f in freq_range:
            l_freq, h_freq = f
#            feature_filter = mne.filter.create_filter(np.array(i).flatten(), sampling_freq, 
#                                                      l_freq = l_freq, 
#                                                      h_freq = h_freq, 
#                                                      l_trans_bandwidth = 0.5, 
#                                                      h_trans_bandwidth = 0.5, 
#                                                      fir_design='firwin')
#            j = np.convolve(feature_filter, np.array(i).flatten())
            psd, energy = feature_extraction(np.array(i).flatten(), l_freq, 
                                             h_freq)
            psd_var[n].append(psd)
            e_var[n].append(energy)
            n = n + 1
    #        if n == 79:
    #            n = 0
        o = o + 1

for i in range(len(Fp1_delta_psd)):
    fitur = (Fp1_delta_psd[i], Fp1_theta_psd[i], Fp1_alpha_psd[i], 
                Fp1_beta_psd[i], Fp1_gamma_psd[i], Fp2_delta_psd[i], 
                Fp2_theta_psd[i], Fp2_alpha_psd[i], Fp2_beta_psd[i],
                Fp2_gamma_psd[i], F3_delta_psd[i], F3_theta_psd[i], 
                F3_alpha_psd[i], F3_beta_psd[i], F3_gamma_psd[i], F4_delta_psd[i],
                F4_theta_psd[i], F4_alpha_psd[i], F4_beta_psd[i], F4_gamma_psd[i], 
                C3_delta_psd[i], C3_theta_psd[i], C3_alpha_psd[i], C3_beta_psd[i],
                C3_gamma_psd[i], C4_delta_psd[i], C4_theta_psd[i], C4_alpha_psd[i], 
                C4_beta_psd[i], C4_gamma_psd[i], P3_delta_psd[i], P3_theta_psd[i], 
                P3_alpha_psd[i], P3_beta_psd[i], P3_gamma_psd[i], P4_delta_psd[i], 
                P4_theta_psd[i], P4_alpha_psd[i], P4_beta_psd[i], P4_gamma_psd[i], 
                O1_delta_psd[i], O1_theta_psd[i], O1_alpha_psd[i], O1_beta_psd[i], 
                O1_gamma_psd[i], O2_delta_psd[i], O2_theta_psd[i], O2_alpha_psd[i],
                O2_beta_psd[i], O2_gamma_psd[i], F7_delta_psd[i], F7_theta_psd[i], 
                F7_alpha_psd[i], F7_beta_psd[i], F7_gamma_psd[i], F8_delta_psd[i], 
                F8_theta_psd[i], F8_alpha_psd[i], F8_beta_psd[i], F8_gamma_psd[i], 
                T3_delta_psd[i], T3_theta_psd[i], T3_alpha_psd[i], T3_beta_psd[i], 
                T3_gamma_psd[i], T4_delta_psd[i], T4_theta_psd[i], T4_alpha_psd[i], 
                T4_beta_psd[i], T4_gamma_psd[i], T5_delta_psd[i], T5_theta_psd[i], 
                T5_alpha_psd[i], T5_beta_psd[i], T5_gamma_psd[i], T6_delta_psd[i], 
                T6_theta_psd[i], T6_alpha_psd[i], T6_beta_psd[i], T6_gamma_psd[i], 
                Fp1_delta_e[i], Fp1_theta_e[i], Fp1_alpha_e[i], Fp1_beta_e[i], 
                Fp1_gamma_e[i], Fp2_delta_e[i], Fp2_theta_e[i], Fp2_alpha_e[i], 
                Fp2_beta_e[i], Fp2_gamma_e[i], F3_delta_e[i], F3_theta_e[i], 
                F3_alpha_e[i], F3_beta_e[i], F3_gamma_e[i], F4_delta_e[i], 
                F4_theta_e[i], F4_alpha_e[i], F4_beta_e[i], F4_gamma_e[i], 
                C3_delta_e[i], C3_theta_e[i], C3_alpha_e[i], C3_beta_e[i], 
                C3_gamma_e[i], C4_delta_e[i], C4_theta_e[i], C4_alpha_e[i], 
                C4_beta_e[i], C4_gamma_e[i], P3_delta_e[i], P3_theta_e[i], 
                P3_alpha_e[i], P3_beta_e[i], P3_gamma_e[i], P4_delta_e[i], 
                P4_theta_e[i], P4_alpha_e[i], P4_beta_e[i], P4_gamma_e[i], 
                O1_delta_e[i], O1_theta_e[i], O1_alpha_e[i], O1_beta_e[i], 
                O1_gamma_e[i], O2_delta_e[i], O2_theta_e[i], O2_alpha_e[i], 
                O2_beta_e[i], O2_gamma_e[i], F7_delta_e[i], F7_theta_e[i], 
                F7_alpha_e[i], F7_beta_e[i], F7_gamma_e[i], F8_delta_e[i], 
                F8_theta_e[i], F8_alpha_e[i], F8_beta_e[i], F8_gamma_e[i], 
                T3_delta_e[i], T3_theta_e[i], T3_alpha_e[i], T3_beta_e[i], 
                T3_gamma_e[i], T4_delta_e[i], T4_theta_e[i], T4_alpha_e[i], 
                T4_beta_e[i], T4_gamma_e[i], T5_delta_e[i], T5_theta_e[i], 
                T5_alpha_e[i], T5_beta_e[i], T5_gamma_e[i], T6_delta_e[i], 
                T6_theta_e[i], T6_alpha_e[i], T6_beta_e[i], T6_gamma_e[i], 0)
        
    with open ('fitur_senang.csv', mode='a', newline = '') as fitur_senang:
        fitur_writer = csv.writer(fitur_senang, delimiter = ',')
        fitur_writer.writerow(fitur)
#        fitur = 0
#    print (psd)
#    print (energy)


#        if i == Fp1 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp1_delta_psd.append(psd)
#            Fp1_delta_e.append(energy)
#        elif i == Fp1 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp1_theta_psd.append(psd)
#            Fp1_theta_e.append(energy)
#        elif i == Fp1 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp1_alpha_psd.append(psd)
#            Fp1_alpha_e.append(energy)
#        elif i == Fp1 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp1_beta_psd.append(psd)
#            Fp1_beta_e.append(energy)
#        elif i == Fp1 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp1_gamma_psd.append(psd)
#            Fp1_gamma_e.append(energy)
#        elif i == 1 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp2_delta_psd.append(psd)
#            Fp2_delta_e.append(energy)
#        elif i == 1 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp2_theta_psd.append(psd)
#            Fp2_theta_e.append(energy)
#        elif i == 1 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp2_alpha_psd.append(psd)
#            Fp2_alpha_e.append(energy)
#        elif i == 1 and n == 3 :
#            j = np.convolve(feature_filter, Fp2)
#            psd, energy = feature_extraction(j)
#            Fp2_beta_psd.append(psd)
#            Fp2_beta_e.append(energy)
#        elif i == 1 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            Fp2_gamma_psd.append(psd)
#            Fp2_gamma_e.append(energy)
#        elif i == 2 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F3_delta_psd.append(psd)
#            F3_delta_e.append(energy)
#        elif i == 2 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F3_theta_psd.append(psd)
#            F3_theta_e.append(energy)
#        elif i == 2 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F3_alpha_psd.append(psd)
#            F3_alpha_e.append(energy)
#        elif i == 2 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F3_beta_psd.append(psd)
#            F3_beta_e.append(energy)
#        elif i == 2 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F3_gamma_psd.append(psd)
#            F3_gamma_e.append(energy)
#        elif i == 3 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F4_delta_psd.append(psd)
#            F4_delta_e.append(energy)
#        elif i == 3 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F4_theta_psd.append(psd)
#            F4_theta_e.append(energy)
#        elif i == 3 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F4_alpha_psd.append(psd)
#            F4_alpha_e.append(energy)
#        elif i == 3 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F4_beta_psd.append(psd)
#            F4_beta_e.append(energy)
#        elif i == 3 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F4_gamma_psd.append(psd)
#            F4_gamma_e.append(energy)
#        elif i == 4 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C3_delta_psd.append(psd)
#            C3_delta_e.append(energy)
#        elif i == 4 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C3_theta_psd.append(psd)
#            C3_theta_e.append(energy)
#        elif i == 4 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C3_alpha_psd.append(psd)
#            C3_alpha_e.append(energy)
#        elif i == 4 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C3_beta_psd.append(psd)
#            C3_beta_e.append(energy)
#        elif i == 4 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C3_gamma_psd.append(psd)
#            C3_gamma_e.append(energy)
#        elif i == 5 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C4_delta_psd.append(psd)
#            C4_delta_e.append(energy)
#        elif i == 5 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C4_theta_psd.append(psd)
#            C4_theta_e.append(energy)
#        elif i == 5 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C4_alpha_psd.append(psd)
#            C4_alpha_e.append(energy)
#        elif i == 5 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C4_beta_psd.append(psd)
#            C4_beta_e.append(energy)
#        elif i == 5 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            C4_gamma_psd.append(psd)
#            C4_gamma_e.append(energy)
#        elif i == 6 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P3_delta_psd.append(psd)
#            P3_delta_e.append(energy)
#        elif i == 6 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P3_theta_psd.append(psd)
#            P3_theta_e.append(energy)
#        elif i == 6 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P3_alpha_psd.append(psd)
#            P3_alpha_e.append(energy)
#        elif i == 6 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P3_beta_psd.append(psd)
#            P3_beta_e.append(energy)
#        elif i == 6 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P3_gamma_psd.append(psd)
#            P3_gamma_e.append(energy)
#        elif i == 7 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P4_delta_psd.append(psd)
#            P4_delta_e.append(energy)
#        elif i == 7 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P4_theta_psd.append(psd)
#            P4_theta_e.append(energy)
#        elif i == 7 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P4_alpha_psd.append(psd)
#            P4_alpha_e.append(energy)
#        elif i == 7 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P4_beta_psd.append(psd)
#            P4_beta_e.append(energy)
#        elif i == 7 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            P4_gamma_psd.append(psd)
#            P4_gamma_e.append(energy)
#        elif i == 8 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O1_delta_psd.append(psd)
#            O1_delta_e.append(energy)
#        elif i == 8 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O1_theta_psd.append(psd)
#            O1_theta_e.append(energy)
#        elif i == 8 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O1_alpha_psd.append(psd)
#            O1_alpha_e.append(energy)
#        elif i == 8 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O1_beta_psd.append(psd)
#            O1_beta_e.append(energy)
#        elif i == 8 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O1_gamma_psd.append(psd)
#            O1_gamma_e.append(energy)
#        elif i == 9 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O2_delta_psd.append(psd)
#            O2_delta_e.append(energy)
#        elif i == 9 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O2_theta_psd.append(psd)
#            O2_theta_e.append(energy)
#        elif i == 9 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O2_alpha_psd.append(psd)
#            O2_alpha_e.append(energy)
#        elif i == 9 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O2_beta_psd.append(psd)
#            O2_beta_e.append(energy)
#        elif i == 9 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            O2_gamma_psd.append(psd)
#            O2_gamma_e.append(energy)
#        elif i == 10 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F7_delta_psd.append(psd)
#            F7_delta_e.append(energy)
#        elif i == 10 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F7_theta_psd.append(psd)
#            F7_theta_e.append(energy)
#        elif i == 10 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F7_alpha_psd.append(psd)
#            F7_alpha_e.append(energy)
#        elif i == 10 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F7_beta_psd.append(psd)
#            F7_beta_e.append(energy)
#        elif i == 10 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F7_gamma_psd.append(psd)
#            F7_gamma_e.append(energy)
#        elif i == 11 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F8_delta_psd.append(psd)
#            F8_delta_e.append(energy)
#        elif i == 11 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F8_theta_psd.append(psd)
#            F8_theta_e.append(energy)
#        elif i == 11 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F8_alpha_psd.append(psd)
#            F8_alpha_e.append(energy)
#        elif i == 11 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F8_beta_psd.append(psd)
#            F8_beta_e.append(energy)
#        elif i == 11 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            F8_gamma_psd.append(psd)
#            F8_gamma_e.append(energy)
#        elif i == 12 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T3_delta_psd.append(psd)
#            T3_delta_e.append(energy)
#        elif i == 12 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T3_theta_psd.append(psd)
#            T3_theta_e.append(energy)
#        elif i == 12 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T3_alpha_psd.append(psd)
#            T3_alpha_e.append(energy)
#        elif i == 12 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T3_beta_psd.append(psd)
#            T3_beta_e.append(energy)
#        elif i == 12 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T3_gamma_psd.append(psd)
#            T3_gamma_e.append(energy)
#        elif i == 13 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T4_delta_psd.append(psd)
#            T4_delta_e.append(energy)
#        elif i == 13 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T4_theta_psd.append(psd)
#            T4_theta_e.append(energy)
#        elif i == 13 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T4_alpha_psd.append(psd)
#            T4_alpha_e.append(energy)
#        elif i == 13 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T4_beta_psd.append(psd)
#            T4_beta_e.append(energy)
#        elif i == 13 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T4_gamma_psd.append(psd)
#            T4_gamma_e.append(energy)
#        elif i == 14 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T5_delta_psd.append(psd)
#            T5_delta_e.append(energy)
#        elif i == 14 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T5_theta_psd.append(psd)
#            T5_theta_e.append(energy)
#        elif i == 14 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T5_alpha_psd.append(psd)
#            T5_alpha_e.append(energy)
#        elif i == 14 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T5_beta_psd.append(psd)
#            T5_beta_e.append(energy)
#        elif i == 14 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T5_gamma_psd.append(psd)
#            T5_gamma_e.append(energy)
#        elif i == 15 and n == 0 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T6_delta_psd.append(psd)
#            T6_delta_e.append(energy)
#        elif i == 15 and n == 1 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T6_theta_psd.append(psd)
#            T6_theta_e.append(energy)
#        elif i == 15 and n == 2 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T6_alpha_psd.append(psd)
#            T6_alpha_e.append(energy)
#        elif i == 15 and n == 3 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T6_beta_psd.append(psd)
#            T6_beta_e.append(energy)
#        elif i == 15 and n == 4 :
#            j = np.convolve(feature_filter, i)
#            psd, energy = feature_extraction(j)
#            T6_gamma_psd.append(psd)
#            T6_gamma_e.append(energy)
#        n = n + 1
        

#def normalization ():
    
#raw.plot(start=0, duration=15) #plot raw data

#print(info['ch_names']) #tampilkan nama channel
#layout_from_raw = mne.channels.make_eeg_layout(raw.info)
#layout_from_raw.plot() #plot montage
#print(info)

#print()  # insert a blank line
#raw.copy().pick_types(meg=False, stim=True).plot(start=0, duration=10)

#print (data)

#Marking bad channel (yang kosong/sensor rusak dsb0
#raw.info['bads'] = ['O1-A1','O2-A2']

#Epochs?
#Evoked?

#Filter 1 Hz untuk ICA
#raw.filter(1., None, n_jobs=1, fir_design = 'firwin')

#Preprocessing ICA
#ica = ICA(n_components=n_components, method=method, random_state=random_state)

#Ini buat reject untuk tiap jenis data
#reject = dict(grad=4000e-13, # T / m (gradiometers)
 #             mag=4e-12, # T (magnetometers)
 #             eeg=40e-6, # V (EEG channels)
 #c            eog=250e-6 # V (EOG channels)
#              )#

#reject = dict(grad=4000e-13, eeg=40e-6)
#ica.fit(raw, picks='eeg', decim=decim, reject=reject)