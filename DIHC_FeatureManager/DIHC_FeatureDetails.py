"""
File Name: DIHC_FeatureDetails.py
Author: WWM Emran (Emran Ali)
Involvement: HumachLab & Deakin- Innovation in Healthcare (DIHC)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Date: 3/09/2021 7:38 pm
"""

from enum import Enum


class DIHC_FeatureDetails:
    td_linear_statistical = ['maximum', 'minimum', 'mean', 'median', 'standardDeviation', 'variance', 'kurtosis',
                             'skewness', 'numberOfZeroCrossing', 'positiveToNegativeSampleRatio',
                             'positiveToNegativePeakRatio', 'meanAbsoluteValue']
    td_linear_statistical_short = ['max', 'min', 'mean', 'med', 'std', 'var', 'kur', 'skew', 'zCross', 'pnRatio',
                                   'pnPkRatio', 'meanAbs']
    td_linear_statistical_short2 = ['Max', 'Min', 'Mean', 'Med', 'Std', 'Var', 'Kur', 'Skew', 'zCross', 'PN-Ratio',
                                    'PN-PkRatio', 'MeanAbs']

    td_nonlinear_entropy = ['approximateEntropy', 'sampleEntropy', 'permutationEntropy',
                            'singularValueDecompositionEntropy',
                            'fuzzyEntropy', 'distributionEntropy', 'shannonEntropy', 'renyiEntropy']
    td_nonlinear_entropy_short = ['appEn', 'sampEn', 'permEn', 'svdEn', 'fuzzEn', 'distEn', 'shanEn', 'renEn']
    td_nonlinear_entropy_short2 = ['ApEn', 'SampEn', 'PermEn', 'SVDEn', 'FuzzEn', 'DistEn', 'ShanEn', 'RenEn']

    td_nonlinear_complexity_and_fractal_dimensions = ['lempelZivComplexity', 'hjorthMobility', 'hjorthComplexity',
                                                      'fisherInfo', 'petrosianFd', 'katzFd', 'higuchiFd',
                                                      'detrendedFluctuation']
    # 'hurstExponent',
    td_nonlinear_complexity_and_fractal_dimensions_short = ['LZComp', 'hjMob', 'hjComp', 'fishInfo', 'petFd', 'katzFd',
                                                            'higFd', 'detFluc']
    td_nonlinear_complexity_and_fractal_dimensions_short2 = ['LZComp', 'HjMob', 'HjComp', 'FishInfo', 'PetFd', 'KatzFd',
                                                             'HigFd', 'DetFluc']
    # 'hurstExp',

    # dist_en_variations = ['distributionEntropy4', 'distributionEntropy6', 'distributionEntropy8', 'distributionEntropy10']

    # td_nonlinear_features3_entropy_profiling = ['entropyProfiledTotalSampleEntropy', 'entropyProfiledAverageSampleEntropy', 'entropyProfiledMaximumSampleEntropy', 'entropyProfiledMinimumSampleEntropy',
    #                                             'entropyProfiledMedianSampleEntropy', 'entropyProfiledStandardDeviationSampleEntropy', 'entropyProfiledVarianceSampleEntropy',
    #                                             'entropyProfiledKurtosisSampleEntropy', 'entropyProfiledSkewnessSampleEntropy']

    td_nonlinear_samp_entropy_profiling = ['entropyProfiled_total_sampleEntropy',
                                           'entropyProfiled_average_sampleEntropy',
                                           'entropyProfiled_maximum_sampleEntropy',
                                           'entropyProfiled_minimum_sampleEntropy',
                                           'entropyProfiled_median_sampleEntropy',
                                           'entropyProfiled_standardDeviation_sampleEntropy',
                                           'entropyProfiled_variance_sampleEntropy',
                                           'entropyProfiled_kurtosis_sampleEntropy',
                                           'entropyProfiled_skewness_sampleEntropy']
    td_nonlinear_samp_entropy_profiling_short = ['ep_tot_sampEn', 'ep_avge_sampEn', 'ep_max_sampEn', 'ep_min_sampEn',
                                                 'ep_med_sampEn', 'ep_std_sampEn', 'ep_var_sampEn', 'ep_kur_sampEn',
                                                 'ep_skew_sampEn']
    td_nonlinear_samp_entropy_profiling_short2 = ['TotalSampEn', 'AvgSampEn', 'MaxSampEn', 'MinSampEn',
                                                  'MedSampEn', 'StdSampEn', 'VarSampEn', 'KurSampEn',
                                                  'SkewSampEn']
    # other means gamma frequency
    fd_linear_statistical = ['fd_maximum', 'fd_minimum', 'fd_mean', 'fd_median', 'fd_standardDeviation', 'fd_variance',
                             'fd_kurtosis', 'fd_skewness']
    fd_linear_statistical_short = ['fd_max', 'fd_min', 'fd_mean', 'fd_med', 'fd_std', 'fd_var', 'fd_kur', 'fd_skew']
    fd_linear_statistical_short2 = ['FD_Max', 'FD_Min', 'FD_Mean', 'FD_Med', 'FD_Std', 'FD_Var', 'FD_Kur', 'FD_Skew']

    fd_linear_statistical_binwise = ['fd_maximum_alpha', 'fd_minimum_alpha', 'fd_mean_alpha', 'fd_median_alpha',
                                     'fd_standardDeviation_alpha', 'fd_variance_alpha', 'fd_kurtosis_alpha',
                                     'fd_skewness_alpha',
                                     'fd_maximum_beta', 'fd_minimum_beta', 'fd_mean_beta', 'fd_median_beta',
                                     'fd_standardDeviation_beta', 'fd_variance_beta', 'fd_kurtosis_beta',
                                     'fd_skewness_beta',
                                     'fd_maximum_delta', 'fd_minimum_delta', 'fd_mean_delta', 'fd_median_delta',
                                     'fd_standardDeviation_delta', 'fd_variance_delta', 'fd_kurtosis_delta',
                                     'fd_skewness_delta',
                                     'fd_maximum_theta', 'fd_minimum_theta', 'fd_mean_theta', 'fd_median_theta',
                                     'fd_standardDeviation_theta', 'fd_variance_theta', 'fd_kurtosis_theta',
                                     'fd_skewness_theta',
                                     'fd_maximum_other', 'fd_minimum_other', 'fd_mean_other', 'fd_median_other',
                                     'fd_standardDeviation_other', 'fd_variance_other', 'fd_kurtosis_other',
                                     'fd_skewness_other']
    fd_linear_statistical_binwise_short = ['fd_max_alpha', 'fd_min_alpha', 'fd_mean_alpha', 'fd_med_alpha',
                                           'fd_std_alpha', 'fd_var_alpha', 'fd_kur_alpha', 'fd_skew_alpha',
                                           'fd_max_beta', 'fd_min_beta', 'fd_mean_beta', 'fd_med_beta', 'fd_std_beta',
                                           'fd_var_beta', 'fd_kur_beta', 'fd_skew_beta',
                                           'fd_max_delta', 'fd_min_delta', 'fd_mean_delta', 'fd_med_delta',
                                           'fd_std_delta', 'fd_var_delta', 'fd_kur_delta', 'fd_skew_delta',
                                           'fd_max_theta', 'fd_min_theta', 'fd_mean_theta', 'fd_med_theta',
                                           'fd_std_theta', 'fd_var_theta', 'fd_kur_theta', 'fd_skew_theta',
                                           'fd_max_other', 'fd_min_other', 'fd_mean_other', 'fd_med_other',
                                           'fd_std_other', 'fd_var_other', 'fd_kur_other', 'fd_skew_other']
    fd_linear_statistical_binwise_short2 = ['FD_MaxAlp', 'FD_MinAlp', 'FD_MeanAlp', 'FD_MedAlp', 'FD_StdAlp',
                                            'FD_VarAlp', 'FD_KurAlp', 'FD_SkewAlp',
                                            'FD_MaxBet', 'FD_MinBet', 'FD_MeanBet', 'FD_MedBet', 'FD_StdBet',
                                            'FD_VarBet', 'FD_KurBet', 'FD_SkewBet',
                                            'FD_MaxDel', 'FD_MinDel', 'FD_MeanDel', 'FD_MedDel', 'FD_StdDel',
                                            'FD_VarDel', 'FD_KurDel', 'FD_SkewDel',
                                            'FD_MaxThe', 'FD_MinThe', 'FD_MeanThe', 'FD_MedThe', 'FD_StdThe',
                                            'FD_VarThe', 'FD_KurThe', 'FD_SkewThe',
                                            'FD_MaxOth', 'FD_MinOth', 'FD_MeanOth', 'FD_MedOth', 'FD_StdOth',
                                            'FD_VarOth', 'FD_KurOth', 'FD_SkewOth']

    fd_nonlinear_entropy = ['spectralEntropy']
    fd_nonlinear_entropy_short = ['specEn']
    fd_nonlinear_entropy_short2 = ['SpecEn']

    fd_spectral_power = ['fd_bandPower']
    fd_spectral_power_short = ['fd_bandPw']
    fd_spectral_power_short2 = ['fd_bandPw']

    fd_spectral_band_power = ['fd_bandPower_alpha', 'fd_bandPower_beta', 'fd_bandPower_delta',
                              'fd_bandPower_theta', 'fd_bandPower_other']
    fd_spectral_band_power_short = ['fd_bandPw_alpha', 'fd_bandPw_beta', 'fd_bandPw_delta',
                                    'fd_bandPw_theta',
                                    'fd_bandPw_other']
    fd_spectral_band_power_short2 = ['fd_bandPwAlp', 'fd_bandPwBet', 'fd_bandPwDel', 'fd_bandPwThe',
                                     'fd_bandPwOth']

    band_frequency_list = {'alpha': (8, 14), 'beta': (14, 31), 'delta': (0, 5), 'theta': (5, 8), 'gamma': (31, 100),
                           'other': (31, 51)}

    # band_frequency_list = {'alpha': (8, 12), 'beta': (12, 30), 'delta': (0.5, 4), 'theta': (4, 8), 'gamma': (30, 100), 'other': (30, 51)}

    # ## Initialization
    def __init__(self):
        self.all_features = self.td_linear_statistical + self.td_nonlinear_entropy + self.td_nonlinear_complexity_and_fractal_dimensions + \
                            self.td_nonlinear_samp_entropy_profiling + self.fd_linear_statistical + self.fd_linear_statistical_binwise + \
                            self.fd_nonlinear_entropy + self.fd_spectral_power + self.fd_spectral_band_power
        self.all_features_short = self.td_linear_statistical_short + self.td_nonlinear_entropy_short + \
                                  self.td_nonlinear_complexity_and_fractal_dimensions_short + \
                                  self.td_nonlinear_samp_entropy_profiling_short + self.fd_linear_statistical_short + \
                                  self.fd_linear_statistical_binwise_short + self.fd_nonlinear_entropy_short + self.fd_spectral_power_short+ self.fd_spectral_band_power_short
        self.all_features_short2 = self.td_linear_statistical_short2 + self.td_nonlinear_entropy_short2 + \
                                   self.td_nonlinear_complexity_and_fractal_dimensions_short2 + \
                                   self.td_nonlinear_samp_entropy_profiling_short2 + self.fd_linear_statistical_short2 + \
                                   self.fd_linear_statistical_binwise_short2 + self.fd_nonlinear_entropy_short2 + self.fd_spectral_power_short2 + self.fd_spectral_band_power_short2
        return

    # ## Mapping features with short names
    def map_feature_names(self, feature_list):
        feat_short_names = []

        for feat in feature_list:
            if feat in self.all_features:
                i = self.all_features.index(feat)
                # feat_short_names.append(self.all_features_short[i])
                feat_short_names.append(self.all_features_short2[i])
        return feat_short_names



class DIHC_FeatureGroup(Enum):
    #TD-Lin
    tdLinStt = DIHC_FeatureDetails.td_linear_statistical

    tdLin = list(tdLinStt)

    #TD-NL
    tdNlEn = DIHC_FeatureDetails.td_nonlinear_entropy
    tdNlComFD = DIHC_FeatureDetails.td_nonlinear_complexity_and_fractal_dimensions
    tdNlEnSamProf = DIHC_FeatureDetails.td_nonlinear_samp_entropy_profiling

    tdNl = list(tdNlEn+tdNlComFD+tdNlEnSamProf)

    #TD
    td = list(tdLin+tdNl)

    #FD-Lin
    fdLinStt = DIHC_FeatureDetails.fd_linear_statistical
    fdLinSttBnd = DIHC_FeatureDetails.fd_linear_statistical_binwise

    fdLin = list(fdLinStt+fdLinSttBnd)

    #TD-NL
    fdNlEn = DIHC_FeatureDetails.fd_nonlinear_entropy
    fdNlPw = DIHC_FeatureDetails.fd_spectral_power
    fdNlPwBnd = DIHC_FeatureDetails.fd_spectral_band_power
    fdNlAllPw = DIHC_FeatureDetails.fd_spectral_power+DIHC_FeatureDetails.fd_spectral_band_power

    fdNl = list(fdNlEn+fdNlPw+fdNlPwBnd)

    #FD
    fd = list(fdLin+fdNl)

    #ALL
    all = list(td+fd) 

