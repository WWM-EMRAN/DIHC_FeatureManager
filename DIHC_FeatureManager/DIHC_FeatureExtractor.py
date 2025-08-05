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
from scipy.spatial.distance import pdist, squareform

# from numba import jit
# from math import factorial, log
# from sklearn.neighbors import KDTree
# from scipy.signal import periodogram, welch, butter, lfilter

# from utils import _linear_regression, _log_n

### SRART: My modules ###
from DIHC_FeatureManager import *
from DIHC_FeatureManager.DIHC_EntropyProfile import *
from DIHC_FeatureManager.DIHC_FeatureDetails import *
from DIHC_FeatureManager.DIHC_FeatureDetails import DIHC_FeatureGroup
### END: My modules ###




###
class DIHC_FeatureExtractor:

    def __init__(self, manage_exceptional_data=0, signal_frequency = 256, sample_per_second=1280, filtering_enabled=False, lowcut=1, highcut=48):
        self.manage_exceptional_data = manage_exceptional_data

        self.td_linear_statistical = DIHC_FeatureDetails.td_linear_statistical
        self.td_nonlinear_entropy = DIHC_FeatureDetails.td_nonlinear_entropy
        self.td_nonlinear_complexity_and_fractal_dimensions = DIHC_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions
        # 'hurstExponent',
        self.td_nonlinear_samp_entropy_profiling = DIHC_FeatureDetails.td_nonlinear_samp_entropy_profiling
        # other means gamma frequency
        self.fd_linear_statistical = DIHC_FeatureDetails.fd_linear_statistical
        self.fd_linear_statistical_binwise = DIHC_FeatureDetails.fd_linear_statistical_binwise
        self.fd_spectral_band_power = DIHC_FeatureDetails.fd_spectral_band_power

        self.band_frequency_list = DIHC_FeatureDetails.band_frequency_list

        self.band = (0, signal_frequency)

        # #All features
        self.feature_list = [DIHC_FeatureDetails.td_linear_statistical, DIHC_FeatureDetails.td_nonlinear_entropy,
                             DIHC_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions,
                             DIHC_FeatureDetails.td_nonlinear_samp_entropy_profiling, DIHC_FeatureDetails.fd_linear_statistical,
                             DIHC_FeatureDetails.fd_linear_statistical_binwise, DIHC_FeatureDetails.fd_spectral_band_power]

        self.signal_frequency = signal_frequency
        self.sample_per_second = sample_per_second
        self.filtering_enabled = filtering_enabled
        self.lowcut = lowcut
        self.highcut = highcut
        if self.filtering_enabled:
            self.lowcut = lowcut
            self.highcut = highcut

        self.fd_data_dict = None
        self.entropy_profile = None 

        return


    def get_new_features_to_calculate(self, feature_types):
        feature_names = None

        # Select features based on type
        if len(feature_types)==0:
            feature_names = [item for sublist in self.feature_list for item in sublist]
        else:
            feature_names = [item for i, sublist in enumerate(self.feature_list) for item in sublist if i in feature_types]

        return feature_names


    ### Getting all the features
    #############################################################
    def generate_features(self, seg_data, feature_names):
        self.fd_data_dict = None
        self.entropy_profile = None 
        feature_values = []

        seg_values = seg_data.copy()
        # seg_values = data_frame_segment[self.channel_name].values.flatten() #np.array(data_frame_segment) # data_frame_segment.values.flatten()
        # seg_values = np.round(seg_values, decimals=20) # ### Don't know why but some features are getting NaN if this is not given, especially SpertralEntropy
        # seg_values = 1000*seg_values

        #check if the feate names are enum or string
        if feature_names is None or len(feature_names)==0:
            print("Extracting all features.")
            feature_names = DIHC_FeatureGroup.all.value
        elif type(feature_names[0]) != DIHC_FeatureGroup:
            print("Invalid features...")
            exit(0)
            # return
        else:
            print("Extracting some features.")
            feature_names_copy = list(feature_names)
            feature_names = []
            for itm in feature_names_copy:
                feature_names.extend(itm.value)

            #remove duplicate and sort
            feature_names_copy = list(set(feature_names))
            all_feature_names = DIHC_FeatureGroup.all.value
            feature_names = [it for it in all_feature_names if it in feature_names_copy] 

        # Generate corresponding features
        for feat in feature_names:
            print(f"---> {feat}") 
            method = None
            final_feat = None
            try:
                final_feat = feat
                final_data = seg_values

                #Reuse appropriate method call
                if feat.startswith('fd_'):
                    if (feat in (self.fd_linear_statistical)) or (feat in (self.fd_linear_statistical_binwise)):
                        #FFT data for frequency domain features
                        # print('HHHHHHHHHH', feat, final_feat, final_data[:5], max(final_data))
                        data_dict = self.fd_data_dict
                        if (self.fd_data_dict is None):
                            data_dict = self.fd_spectralAmplitude(seg_values)
                            self.fd_data_dict = data_dict
                            # print('JJJJJJJJJJ', feat, final_feat, final_data[:5], max(final_data), max(self.fd_data_dict.values()))

                        final_feat_list = (feat.split('_'))
                        fnl = len(final_feat_list)
                        if fnl>1:
                            final_feat = final_feat_list[1]
                            final_data = list(data_dict.values())

                            tmp = data_dict.keys()

                            if fnl > 2:
                                tmp = [i for i in tmp if i in range(self.band_frequency_list[final_feat_list[2]][0], self.band_frequency_list[final_feat_list[2]][1])]
                                final_data = [data_dict[x] for x in tmp]

                        # print('KKKKKKKKK', feat, final_feat, final_data[:5], max(final_data))
                    elif (feat in (self.fd_spectral_band_power)):
                        final_feat_list = (feat.split('_'))
                        fnl = len(final_feat_list)

                        if fnl > 1:
                            final_feat = f'{final_feat_list[0]}_{final_feat_list[1]}'
                            self.band = (0, self.signal_frequency)

                            if fnl > 2:
                                self.band = (self.band_frequency_list[final_feat_list[2]][0], self.band_frequency_list[final_feat_list[2]][1])

                elif feat.startswith('entropyProfiled_'):
                    enProf = self.entropy_profile
                    if self.entropy_profile is None:
                        entProf_obj = DIHC_EntropyProfile()
                        enProf = entProf_obj.get_sample_entropy_profile(final_data)
                        dat = np.asarray(enProf)
                        if isinstance(dat[0], (list, np.ndarray)):
                            dat2 = [np.float64(item) for sublist in dat for item in sublist]
                        else:
                            dat2 = [np.float64(item) for item in dat]
                        enProf = np.array(dat2)
                        # enProf = self._get_sample_entropy_profile(final_data)
                        # dat = np.asarray(enProf)
                        # dat2 = [0.0]
                        # if len(enProf)>1:
                        #     dat2 = [np.float64(item) for sublist in dat for item in sublist]
                        # enProf = np.array(dat2)
                        self.entropy_profile = enProf

                    final_feat = (feat.split('_'))
                    # fnl = len(final_feat)
                    final_feat = final_feat[1]

                    final_data = enProf

                # print(feat, final_feat, seg_values, final_data)
                # print(f'Calling... {final_feat} for feature {feat}')
                method = getattr(self, final_feat)

            except AttributeError:
                print(f'Method for feature: {final_feat} is not implemented.')
                raise NotImplementedError("Class `{}` does not implement `{}`".format(self.__class__.__name__, final_feat))
                # return

            # feat_val = method(final_data)
            # print(feat, final_data, type(final_data), feat_val, type(feat_val))
            # print(feat, type(final_data), len(final_data), final_data)

            feat_val = 0
            result = np.all(final_data == final_data[0]) if len(final_data)>1 else True
            if not result:
                feat_val = method(final_data)
                # # Handling nan data
                # if feat_val == np.nan:
                #     feat_val = 0

            # print(f'{feat} -- {feat_val} -- {type(feat_val)}')
            feat_val = round(feat_val, 2)
            # feat_val = round(feat_val, 16)
            # print(f'{feat} -- {feat_val}')
            feature_values.append([feat_val])

        # print(f'{feature_names} -- {feature_values}')
        numpy_array = np.array(feature_values)
        numpy_array = numpy_array.T

        # print(f'{len(feature_names)} {numpy_array.shape}')
        # print(f'{numpy_array}')
        # print(f'data--- {numpy_array} {feature_names}')

        all_features = pd.DataFrame()
        if len(feature_names)>0 and len(numpy_array)>0:
            all_features = pd.DataFrame(numpy_array, columns=feature_names)
        # print(f'{all_features}')

        # ##########################################################
        # all_features = all_features[all_features != np.inf]
        # print(f'Data and type: ', type(all_features), all_features['spectralEntropy'])
        # if all_features.isnull().values.any():
        #     print(f'Infinity found: ', all_features)

        # Exceptional data management
        if self.manage_exceptional_data == 0:
            all_features = all_features[all_features != np.inf]
        elif self.manage_exceptional_data == 1:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.fillna(0)
        elif self.manage_exceptional_data == 2:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.dropna()
        elif self.manage_exceptional_data == 3:
            all_features = all_features[all_features != np.inf]
            all_features = all_features.fillna(all_features.mean()) 

        return all_features 

###########################################################################
### Time Domain Features

    ### Total value of the segment
    def total(self, data):
        tot = np.sum(data)
        return tot

    ### Summation value of the segment
    def summation(self, data):
        avg = np.sum(data)
        return avg

    ### Average value of the segment
    def average(self, data):
        avg = np.mean(data)
        return avg

    ### Minimum value of the segment
    def minimum(self, data):
        min = np.min(data)
        return min

    ### Maximum value of the segment
    def maximum(self, data):
        max = np.max(data)
        return max

    ### Mean value of the segment
    def mean(self, data):
        mean = np.mean(data)
        return mean

    ### Median value of the segment
    def median(self, data):
        med = np.median(data)
        return med

    ### Standard Deviation value of the segment
    def standardDeviation(self, data):
        std = np.std(data)
        return std

    ### Variance value of the segment
    def variance(self, data):
        var = np.var(data)
        return var

    ### kurtosis value of the segment
    def kurtosis(self, data):
        # kurtosis(y1, fisher=False)
        kur = sp.stats.kurtosis(data)
        return kur

    ### skewness value of the segment
    def skewness(self, data):
        skw = sp.stats.skew(data)
        return skw

    ### peak_or_Max value of the segment
    def peakOrMax(self, data):
        peak = self.maximum(data)
        return peak

    ### numberOfPeaks value of the segment
    def numberOfPeaks(self, data):
        # peaks, _ = find_peaks(x, distance=20)
        # peaks2, _ = find_peaks(x, prominence=1)  # BEST!
        # peaks3, _ = find_peaks(x, width=20)
        # peaks4, _ = find_peaks(x, threshold=0.4)
        numPeak = len(sig.find_peaks(data))
        return numPeak

    ### numberOfZeroCrossing value of the segment
    def numberOfZeroCrossing(self, data):
        # numZC1 = np.where(np.diff(np.sign(data)))[0]
        numZC = num_zerocross(data) #Antropy package
        return numZC #len(numZC1)

    ### positiveToNegativeSampleRatio value of the segment
    def positiveToNegativeSampleRatio(self, data):
        pnSampRatio = (np.sum(np.array(data) >= 0, axis=0)) / (np.sum(np.array(data) < 0, axis=0))
        return pnSampRatio

    ### positiveToNegativeSampleRatio value of the segment
    def positiveToNegativePeakRatio(self, data):
        pnPeakRatio = (len(sig.find_peaks(data))) / (len(sig.find_peaks(-data)))
        return pnPeakRatio

    ### meanAbsoluteValue value of the segment
    def meanAbsoluteValue(self, data):
        meanAbsVal = self.mean(abs(data))
        return meanAbsVal



############ Entropy
    # ### Collected from Antropy and pyeeg package

    def approximateEntropy(self, data):
        ae = app_entropy(data)
        return ae

    def sampleEntropy(self, data):
        se = sample_entropy(data)
        return se

    def permutationEntropy(self, data):
        pe = perm_entropy(data)
        return pe

    def spectralEntropy(self, data):
        sf = self.signal_frequency
        # se = spectral_entropy(data, sf)
        se = spectral_entropy(data, sf, method='welch')
        return se

    def singularValueDecompositionEntropy(self, data):
        svd_e = svd_entropy(data)
        return svd_e

    ############ Fracta dimension
    # Collected from Antropy and pyeeg packages

    def hjorthMobility(self, data):
        hjm, _ = hjorth_params(data)
        return hjm

    def hjorthComplexity(self, data):
        _, hjc = hjorth_params(data)
        return hjc

    def hurstExponent(self, data):
        hre = pyeeg.hurst(data)
        return hre

    def fisherInfo(self, data, tau=1, m=2):
        fsi = pyeeg.fisher_info(data, tau, m)
        return fsi

    def lempelZivComplexity(self, data):
        lzc = lziv_complexity(data)
        return lzc

    def petrosianFd(self, data):
        pfd = petrosian_fd(data)
        return pfd

    def katzFd(self, data):
        kfd = katz_fd(data)
        return kfd

    def higuchiFd(self, data):
        hfd = higuchi_fd(data)
        return hfd

    def detrendedFluctuation(self, data):
        dfl = detrended_fluctuation(data)
        return dfl 
    

    ## ######## Fuzzy entropy (Python implementation) 
    def fuzzyEntropy(self, data,  m=2, tau=1, r=0.2):
        """
        Fuzzy Entropy calculation in Python.
        
        Parameters:
            data : array-like
                Input signal (1D array).
            m : int
                Embedding dimension.
            tau : int
                Time delay.
            r : float
                Tolerance (as a fraction of std of s).
        
        Returns:
            fuzz_ent : float
                Fuzzy entropy value.
        """
        data = np.asarray(data).flatten()
        r = r * np.std(data)
        N = len(data)

        # Indices for embedding
        ind_m = np.array([np.arange(i, i + m * tau, tau) for i in range(N - m * tau)])
        ind_a = np.array([np.arange(i, i + (m + 1) * tau, tau) for i in range(N - m * tau)])

        ym = data[ind_m]
        ya = data[ind_a]

        # Compute Chebyshev distance
        cheb_ym = pdist(ym, metric='chebyshev')
        cheb_ya = pdist(ya, metric='chebyshev')

        cm = np.sum(np.exp(-np.log(2) * (cheb_ym / r) ** 2)) * 2 / (ym.shape[0] * (ym.shape[0] - 1))
        ca = np.sum(np.exp(-np.log(2) * (cheb_ya / r) ** 2)) * 2 / (ya.shape[0] * (ya.shape[0] - 1))

        if cm == 0 or ca == 0:
            return np.nan  # Avoid log(0)
        
        fuzz_ent = -np.log(ca / cm)
        return fuzz_ent 
    

    # ## ######## Distribution entropy (Python implementation) 
    # def distributionEntropy(self, data, m=2, M=500):
    #     """
    #     Compute Distribution Entropy (DistEn) using Peng Li's method.
        
    #     Parameters:
    #         data : array-like
    #             The input signal.
    #         m : int
    #             Embedding dimension (e.g., 2).
    #         M : int
    #             Number of bins for histogram (e.g., 500).
        
    #     Returns:
    #         dist_ent : float
    #             Distribution entropy value.
    #     """
    #     data = np.asarray(data).flatten()
    #     N = len(data)

    #     # Step 1: Form template matrix for embedding dimension m
    #     template_matrix = np.array([data[i:N - m + i + 1] for i in range(m)]).T
    #     mat_len = template_matrix.shape[0]

    #     # Step 2: Calculate Chebyshev distances between all template vectors
    #     all_distances = []
    #     for i in range(mat_len):
    #         template_vec = template_matrix[i]
    #         match_mat = np.delete(template_matrix, i, axis=0)
    #         # Chebyshev (max absolute) distance
    #         d = np.max(np.abs(match_mat - template_vec), axis=1)
    #         all_distances.extend(d)
        
    #     all_distances = np.array(all_distances)

    #     # Step 3: Histogram binning
    #     freq_count, _ = np.histogram(all_distances, bins=M)
    #     prob_freq = freq_count / len(all_distances)
    #     prob_nonzero = prob_freq[prob_freq > 0]

    #     # Step 4: Compute distribution entropy
    #     entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero))
    #     dist_ent = entropy / np.log2(M)
        
    #     return dist_ent
    


    def distributionEntropy(self, data, m=2, M=500): 
        """
        Python implementation of Distribution Entropy as per Peng Li's method.
        Parameters:
            data : 1D numpy array
            m : embedding dimension (typically 2)
            M : number of bins (typically 500)
        Returns:
            dist_en : Distribution Entropy value (should match MATLAB values closely)
        """
        data = np.asarray(data).flatten()
        N = len(data)
        if N <= m:
            raise ValueError("Input data length must be greater than embedding dimension m.")

        # Construct embedding matrix for dimension m
        tmplt_mat = np.array([data[i:N - m + i] for i in range(m)]).T
        mat_len = tmplt_mat.shape[0]

        # Compute Chebyshev distances
        all_dist_m = []
        for i in range(mat_len):
            tmpl_vec = tmplt_mat[i]
            match_mat = np.delete(tmplt_mat, i, axis=0)
            tmpl_repeated = np.tile(tmpl_vec, (mat_len - 1, 1))
            dist = np.max(np.abs(tmpl_repeated - match_mat), axis=1)
            all_dist_m.extend(dist)

        all_dist_m = np.array(all_dist_m)

        # Fixed bin histogram with M bins
        freq_count, _ = np.histogram(all_dist_m, bins=M, range=(np.min(all_dist_m), np.max(all_dist_m)))
        prob = freq_count / len(all_dist_m)

        # Compute Distribution Entropy
        nonzero_prob = prob[prob > 0]
        y = nonzero_prob * np.log2(nonzero_prob)
        dist_en = -np.sum(y) / np.log2(M)

        return dist_en





    def distributionEntropy4(self, data, m=4):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy6(self, data, m=6):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy8(self, data, m=8):
        return self.distributionEntropy(data, m=m)
    def distributionEntropy10(self, data, m=10):
        return self.distributionEntropy(data, m=m)
    



    def shannonEntropy(self, data, m=2):
        bases = collections.Counter([tmp_base for tmp_base in data])
        # define distribution
        dist = [i / sum(bases.values()) for i in bases.values()]

        # use scipy to calculate entropy
        entropy_value = scipyEntropy(dist, base=m)

        return entropy_value


    def _x_log2_x(self, data):
        """ Return x * log2(x) and 0 if x is 0."""
        res = data * np.log2(data)
        if np.size(data) == 1:
            if np.isclose(data, 0.0):
                res = 0.0
        else:
            res[np.isclose(data, 0.0)] = 0.0
        return res


    def renyiEntropy(self, data, alpha=2):
        assert alpha >= 0, "Error: renyi_entropy only accepts values of alpha >= 0, but alpha = {}.".format(alpha)  # DEBUG
        if np.isinf(alpha):
            # XXX Min entropy!
            return - np.log2(np.max(data))
        elif np.isclose(alpha, 0):
            # XXX Max entropy!
            return np.log2(len(data))
        elif np.isclose(alpha, 1):
            # XXX Shannon entropy!
            return - np.sum(self._x_log2_x(data))
        else:
            return (1.0 / (1.0 - alpha)) * np.log2(np.sum(data ** alpha)) 


### Frequency Domain Features

    ##Bandpass filtering
    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        tpl_res = butter(order, [low, high], btype='band')
        b, a = tpl_res[0], tpl_res[1]
        return b, a


    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y


    ##Fast Faurier Transformation
    def _fast_faurier_transformation(self, data, fs, fft_type=2):
        feat_data = None

        if fft_type==0:
            #Normal FFT using scipy
            fft_data = fft.fft(data)
            freqs = fft.fftfreq(len(data)) * fs
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}
        elif fft_type==1:
            #FFT for Amplitude calculation using fftpack
            fft_data = fftpack.fft(data)
            freqs = fftpack.fftfreq(len(data)) * fs
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}
        elif fft_type==2:
            #FFT for Amplitude calculation using fftpack
            fft_data = np.abs(np.fft.fft(data))
            freqs = np.fft.fftfreq(len(data), d=1.0 / fs)
            feat_data = {freqs[i]: fft_data[i] for i in range(len(freqs))}

        return feat_data


    ### Original Frequency domain features
    def fd_spectralAmplitude(self, data):
        filtered_data = data
        sample_per_second = self.sample_per_second

        if self.filtering_enabled:
            lowcut = self.lowcut
            highcut = self.highcut
            filtered_data = self._butter_bandpass_filter(data, lowcut, highcut, sample_per_second, order=6)

        feat_data = self._fast_faurier_transformation(data, sample_per_second)

        return feat_data


    ### Power of a signal or signal band
    ### @Author: raphaelvallat (Author of Antropy package) Source:- https://raphaelvallat.com/bandpower.html
    def fd_bandPower(self, data, method='multitaper', window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.
        Requires MNE-Python >= 0.14.
        Parameters
        ----------
        data : 1d-array :- Input signal in the time-domain.
        sf : float :- Sampling frequency of the data.
        band : list :- Lower and upper frequencies of the band of interest.
        method : string :- Periodogram method: 'welch' or 'multitaper'
        window_sec : float :- Length of each window in seconds. Useful only if method == 'welch'. If None, window_sec = (1 / min(band)) * 2.
        relative : boolean :-If True, return the relative power (= divided by the total power of the signal). If False (default), return the absolute power.
        ------
        Return
        ------
        bp : float :- Absolute or relative band power.
        ------
        Use
        ------
        # Multitaper delta power
        bp = fd_bandpower(data, sf, [0.5, 4], 'multitaper')
        311.559, and 0.790 (for relative band power)
        """
        sf = self.signal_frequency
        band = self.band

        band = np.asarray(band)
        low, high = band
        # Compute the modified periodogram (Welch)
        freqs, psd = 0, 0
        if method == 'welch':
            if window_sec is not None:
                nperseg = window_sec * sf
            else:
                nperseg = (2 / low) * sf
            freqs, psd = welch(data, sf, nperseg=nperseg)
        elif method == 'multitaper':
            psd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0)
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)
        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp





    # ### Minimum value of the segment
    # def fd_minimum(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     min = np.min(data)
    #     return min
    #
    # ### Maximum value of the segment
    # def fd_maximum(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     max = np.max(data)
    #     return max
    #
    # ### Mean value of the segment
    # def fd_mean(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     mean = np.mean(data)
    #     return mean
    #
    # ### Median value of the segment
    # def fd_median(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     med = np.median(data)
    #     return med
    #
    # ### Summation value of the segment
    # def fd_summation(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     avg = np.sum(data)
    #     return avg
    #
    # ### Average value of the segment
    # def fd_average(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     avg = self.mean(data)
    #     return avg
    #
    # ### Standard Deviation value of the segment
    # def fd_standardDeviation(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     std = np.std(data)
    #     return std
    #
    # ### Variance value of the segment
    # def fd_variance(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     var = np.var(data)
    #     return var
    #
    # ### kurtosis value of the segment
    # def fd_kurtosis(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     # kurtosis(y1, fisher=False)
    #     kur = sp.stats.kurtosis(data)
    #     return kur
    #
    # ### skewness value of the segment
    # def fd_skewness(self, data):
    #     data = self.fd_spectralAmplitude(data)
    #     data = data.values()
    #     skw = sp.stats.skew(data)
    #     return skw

### Time-Frequency Domain Features

### Wevlate Domain Features




