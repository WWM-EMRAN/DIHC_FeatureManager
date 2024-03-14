# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import numpy as np
# from DIHC_FeatureManager import DIHC_FeatureManager
# from DIHC_FeatureManager import *
# import DIHC_FeatureManager
from DIHC_FeatureManager.DIHC_FeatureManager import *



if __name__ == '__main__':

    print(f'Data reading started...')
    samp_df = pd.read_csv('./signal_data.csv')
    print(f'Data reading completed...')
    print(samp_df.shape, samp_df.columns)

    print(f'Data minimization started...')
    # sig_freq = 1 #256
    # samp_data = np.array([52, 54, 6, 45, 14, 40, 42, 48, 52, 20, 28, 8, 63, 47, 23])

    sig_freq = 256
    # samp_data = samp_df['signal'].values.tolist()
    samp_data = samp_df.loc[:20*sig_freq-1, 'signal'].values#.tolist()
    # samp_data = samp_df.loc[:5100, 'signal'].values#.tolist()
    # samp_data = samp_df.iloc[:20*256-1, 0:1].values#.tolist()
    # print(len(samp_data))
    print(samp_data.shape, samp_data)
    print(f'Data minimization completed...')


    print(f'Data segmentation started...')
    feat_manager = DIHC_FeatureManager()
    seg_df = feat_manager.get_segments_for_data(samp_data, segment_length=5, signal_frequency=sig_freq)
    print(f'Data segmentation completed...')

    print(f'Showing segments...')
    print(seg_df.shape)
    print(seg_df)


    ########## This portion is for Feature extraction ##############
    # print(f'Feature extraction started...')
    # feat_manager = DIHC_FeatureManager()
    # # feat_df = feat_manager.extract_features_from_data(samp_data, segment_length=5, signal_frequency=sig_freq)
    # feat_df = feat_manager.extract_features_from_data(samp_data, segment_length=5, signal_frequency=sig_freq, has_matlab_engine=True)
    # # feat_df = feat_manager.extract_features_from_data(samp_data, segment_length=5, signal_frequency=sig_freq, has_matlab_engine=False)
    # # feat_df = feat_manager.extract_features_from_data(samp_data, feature_names=[DIHC_FeatureGroup.fdNlPw, DIHC_FeatureGroup.fdNlPwBnd], segment_length=5, signal_frequency=sig_freq, has_matlab_engine=False)
    # # feat_df = feat_manager.extract_features_from_data(samp_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.td], segment_length=5, signal_frequency=sig_freq, has_matlab_engine=True)
    # # feat_df = feat_manager.extract_features_from_data(samp_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.tdNl], segment_length=5, signal_frequency=sig_freq, has_matlab_engine=True)
    # print(f'Feature extraction completed...')

    # print(f'Showing features...')
    # # print(len(feat_df.columns.values.tolist()), feat_df.columns.values.tolist())
    # print(feat_df.shape)
    # print(feat_df)


    ########## This portion is for Sample Entropy Profile extraction ##############
    print(f'Entropy profile extraction started...')
    feat_manager = DIHC_FeatureManager()
    entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, signal_frequency=sig_freq)
    # entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, signal_frequency=sig_freq, has_matlab_engine=True)
    # entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, signal_frequency=sig_freq, has_matlab_engine=False)
    # entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, segment_overlap=0, signal_frequency=sig_freq, has_matlab_engine=False)
    print(f'Entropy profile extraction completed...')

    print(f'Showing entropy profile...')
    # print(len(entProf_df.columns.values.tolist()), entProf_df.columns.values.tolist())
    print(entProf_df.shape)
    print(entProf_df)






    # # matlab test
    # import matlab.engine
    # eng = matlab.engine.start_matlab()
    # # tf = eng.sumanddiff(4,2,nargout=2)
    # tf = eng.fuzzyEn2(samp_data.tolist(),2,1,0.2,nargout=1)
    # print(tf)
    # eng.quit()

