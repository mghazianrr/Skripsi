# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:21:04 2020

@author: Ghazian
"""

import mne
raw = mne.io.read_raw_bdf('01.BDF', preload = True)
raw.plot()
#raw.plot_psd(average = True)