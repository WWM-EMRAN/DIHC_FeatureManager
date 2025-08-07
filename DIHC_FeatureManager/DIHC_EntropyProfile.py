# -*- coding: utf-8 -*-
"""
File Name: DIHC_FeatureExtractor.py
Author: WWM Emran (Emran Ali)
Involvement: HumachLab & Deakin- Innovation in Healthcare (DIHC)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Date: 5/01/2020 8:55 pm
"""


###
import math
import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as sig
import math
import collections
from scipy.stats import entropy as scipyEntropy
from scipy.signal import butter, lfilter, welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
# from scipy.fft import fft
from scipy import fft, fftpack
# from math import log, floor
import math

from antropy import *
import pyeeg

# from numba import jit
# from math import factorial, log
# from sklearn.neighbors import KDTree
# from scipy.signal import periodogram, welch, butter, lfilter

# from utils import _linear_regression, _log_n 
# from utils import _embed

### SRART: My modules ###
from DIHC_FeatureManager import *
from DIHC_FeatureManager.DIHC_FeatureDetails import *
from DIHC_FeatureManager.DIHC_FeatureExtrantor_Helper import *
from DIHC_FeatureManager.DIHC_FeatureDetails import DIHC_FeatureGroup
### END: My modules ###




###
class DIHC_EntropyProfile:

    def __init__(self):
        # self.entropy_profile = None
        return 


    ### Getting all the features
    #############################################################
    def generate_sampEn_profile(self, seg_data): 
        # feature_values = []
        # seg_values = seg_data.copy()
        # final_data = seg_values.copy()
        final_data = seg_data.copy()

        enProf = self.get_sample_entropy_profile(final_data)
        dat = np.asarray(enProf)
        if isinstance(dat[0], (list, np.ndarray)):
            dat2 = [np.float64(item) for sublist in dat for item in sublist]
        else:
            dat2 = [np.float64(item) for item in dat]
        enProf = np.array(dat2)
        enProf = np.asarray(enProf, dtype=np.float64)

        # print(f"enProf: {enProf.shape} {enProf}")

        # dat = np.asarray(enProf)
        # dat2 = [0.0]
        # if len(enProf)>1:
        #     dat2 = [np.float64(item) for sublist in dat for item in sublist]
        # enProf = np.array(dat2)

        entProf_df = pd.DataFrame(enProf, columns=['sampEn_profile'])

        return entProf_df  
    
    

    # ############ Entropy Profiling (Python implementation)
    # def cumulative_histogram_method(self, data, m):
    #     data = np.asarray(data).flatten()
    #     N = len(data)
    #
    #     # Form template matrix for embedding dimension m
    #     tmpltMatM = np.array([data[i:N - m - 1 + i + 1] for i in range(m)]).T
    #     tmpltMatM1 = np.array([data[i:N - m - 1 + i + 1] for i in range(m + 1)]).T
    #
    #     matLenM = tmpltMatM.shape[0]
    #     allDistM = []
    #     allDistM1 = []
    #
    #     for i in range(matLenM):
    #         vecM = tmpltMatM[i]
    #         matM = np.delete(tmpltMatM, i, axis=0)
    #         d = np.max(np.abs(matM - vecM), axis=1)
    #         allDistM.append(np.round(d, 3))
    #
    #         vecM1 = tmpltMatM1[i]
    #         matM1 = np.delete(tmpltMatM1, i, axis=0)
    #         d1 = np.max(np.abs(matM1 - vecM1), axis=1)
    #         allDistM1.append(np.round(d1, 3))
    #
    #     allDistM = np.array(allDistM).T
    #     allDistM1 = np.array(allDistM1).T
    #
    #     D = np.concatenate((allDistM.flatten(), allDistM1.flatten()))
    #     range_vals = np.unique(D)
    #
    #     # Compute cumulative histograms
    #     allHistM = []
    #     allHistM1 = []
    #
    #     for i in range(matLenM):
    #         histM = np.histogram(allDistM[:, i], bins=np.append(range_vals, range_vals[-1]+1))[0]
    #         cumHistM = np.cumsum(histM) / (matLenM - 1)
    #         allHistM.append(cumHistM)
    #
    #         histM1 = np.histogram(allDistM1[:, i], bins=np.append(range_vals, range_vals[-1]+1))[0]
    #         cumHistM1 = np.cumsum(histM1) / (matLenM - 1)
    #         allHistM1.append(cumHistM1)
    #
    #     allHistM = np.array(allHistM).T
    #     allHistM1 = np.array(allHistM1).T
    #
    #     b = np.sum(allHistM, axis=1) / matLenM
    #     a = np.sum(allHistM1, axis=1) / matLenM
    #
    #     return b, a, range_vals
    #
    #
    # #########################################
    # def get_sample_entropy_profile(self, data, m=2):
    #     data = np.asarray(data).flatten()
    #     b, a, r_range = self.cumulative_histogram_method(data, m)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         se_profile = np.log(np.divide(b, a))
    #     se_profile = se_profile[np.isfinite(se_profile)]
    #     return se_profile

    def get_sample_entropy_profile(self, data, m=2):
        data = np.asarray(data).flatten()
        se_profile = compute_entropy_profile_jit(data, m)
        se_profile = se_profile[np.isfinite(se_profile)]
        se_profile = np.asarray(se_profile, dtype=np.float64)
        return se_profile
    ##########################################


