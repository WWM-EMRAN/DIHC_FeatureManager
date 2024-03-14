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

# import matlab
# import matlab.engine
# from utils import _embed

### SRART: My modules ###
from DIHC_FeatureManager import *
from DIHC_FeatureManager.DIHC_FeatureDetails import *
from DIHC_FeatureManager.DIHC_FeatureDetails import DIHC_FeatureGroup
### END: My modules ###




###
class DIHC_EntropyProfile:

    def __init__(self, has_matlab_engine=True, matlab_engine=None, matlab_file_loc=r'./DIHC_FeatureManager/'):
        # self.entropy_profile = None
        self.matlab_engine = matlab_engine
        self.has_matlab_engine = has_matlab_engine
        self.matlab_file_loc = matlab_file_loc
        if matlab_engine is None:
            self.matlab_engine = self.manage_matlab_python_engine()

        return


    def manage_matlab_python_engine(self, existing_eng=None):
        import pkgutil
        import os, sys
        from pathlib import Path

        eggs_loader = pkgutil.find_loader('matlab')
        found = eggs_loader is not None

        mat_bld_path = str(Path.home())
        mat_bld_path = mat_bld_path.replace("\\\\", "\\")
        mat_bld_path = mat_bld_path.replace("\\", "/")
        mat_bld_path += '/matlab_build/lib'

        if existing_eng is None:
            eng = None
            if found:
                import matlab.engine
                eng = matlab.engine.start_matlab()
                eng.cd(self.matlab_file_loc, nargout=0)
                print(f'Starting matlab engine (default)...')
            elif (not found) and (os.path.exists(mat_bld_path)):
                sys.path.append(mat_bld_path)
                import matlab.engine
                eng = matlab.engine.start_matlab()
                print(f'Starting matlab engine (custom)...')
            else:
                print(f'No matlab is installed...')
                exit(0)
            return eng
        else:
            print(f'Quitting matlab engine...')
            existing_eng.quit()
            self.matlab_engine = None
        return


    ### Getting all the features
    #############################################################
    def generate_sampEn_profile(self, seg_data, matlab_eng):
        if self.matlab_engine is None: # and matlab_eng is None:
            if self.has_matlab_engine:
                self.matlab_engine = self.manage_matlab_python_engine()
            else:
                print(f"Does not have the matlab engine setup...")
        else:
            if matlab_eng is not None:
                self.matlab_engine = matlab_eng
            print(f"Already have matlab engine running...")
        # feature_values = []
        # seg_values = seg_data.copy()
        # final_data = seg_values.copy()
        final_data = seg_data.copy()

        enProf = self.get_sample_entropy_profile(final_data)
        dat = np.asarray(enProf)
        dat2 = [0.0]
        if len(enProf)>1:
            dat2 = [np.float64(item) for sublist in dat for item in sublist]
        enProf = np.array(dat2)

        entProf_df = pd.DataFrame(enProf, columns=['sampEn_profile'])


        return entProf_df, self.matlab_engine



    ############ Entropy Profiling
    # Collected from @author: radhagayathri
    def get_sample_entropy_profile(self, data, matlab_eng=None, m=2):
        sig_seg_df_list = data.tolist()

        # Calling Matlab code from Python
        # is_gen = False
        eng = self.matlab_engine
        # if self.matlab_engine is None:
        #     eng = self.manage_matlab_python_engine()
        #     self.matlab_engine = eng
        #     is_gen = True

        # print(f'{len(sig_seg_df_list)} {sig_seg_df_list}')
        ent2 = eng.sampEnProfiling(sig_seg_df_list, m, nargout=1)
        if isinstance(ent2, float):
            ent2 = [ent2]
        ent = list(ent2)
        # print(f'=== {len(ent)} {ent}')

        # if is_gen:
        #     self.manage_matlab_python_engine(existing_eng=eng)

        return ent


