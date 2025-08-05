# %% [markdown]
# ### Testing the DIHC Feature Manager Package 

# %% [markdown]
# ##### Load the package "DIHC_FeatureManager" which is in the same directory as this notebook (or your main python script/notebook) 

# %%
# Importing necessary modules
import pandas as pd
import numpy as np
from DIHC_FeatureManager.DIHC_FeatureManager import *

# %%


# %% [markdown]
# ##### Reading sample data from the file "signal_data.csv" which is in the same directory as this notebook (or your main python script/notebook)

# %%
print(f'Data reading started...')
samp_df = pd.read_csv('./signal_data.csv')
samp_df

# %%


# %% [markdown]
# ##### Observing the shape and columns of the data

# %%
print(f'Data reading completed...')
samp_df.shape, samp_df.columns

# %%


# %% [markdown]
# ##### This data file contains 3 columns: "time", "signal", and "label". We will use only the "signal" column for feature extraction
# ##### For simplicity only the first 20 seconds of the signal is used for feature extraction

# %%
print(f'Data minimization started...')
# sig_freq = 1 #256
# samp_data = np.array([52, 54, 6, 45, 14, 40, 42, 48, 52, 20, 28, 8, 63, 47, 23])

sig_freq = 256
# samp_data = samp_df['signal'].values.tolist()
samp_data = samp_df.loc[:20*sig_freq-1, 'signal'].values#.tolist()
# samp_data = samp_df.loc[:5100, 'signal'].values#.tolist()
# samp_data = samp_df.iloc[:20*256-1, 0:1].values#.tolist()
# print(len(samp_data))
# print(samp_data.shape, samp_data)
print(f'Data minimization completed...')
print(samp_data.shape)
samp_data

# %%


# %% [markdown]
# 
# ##### Create the object of the class "DIHC_FeatureManager" and call the method "get_segments_for_data" to extract features from the data
# ##### Use different parameters of the method "get_segments_for_data" to extract different number of segments

# %%
print(f'Data segmentation started...')
feat_manager = DIHC_FeatureManager()
seg_arr = feat_manager.get_segments_for_data(samp_data, segment_length=5, signal_frequency=sig_freq)
print(f'Data segmentation completed...')

# %%


# %% [markdown]
# ##### Display the segmented data

# %%
print(seg_arr.shape)
seg_arr


# %%


# %% [markdown]
# ##### Create the object of the class "DIHC_FeatureManager" and call the method "extract_features_from_data" to extract features from the data
# ##### Use different parameters of the method "extract_features_from_data" to extract different features

# %%
print(f'Feature extraction started...')
# samp_data = samp_df.loc[:, 'signal'].values
feat_manager = DIHC_FeatureManager()
feat_df = feat_manager.extract_features_from_data(samp_data, segment_length=5, signal_frequency=sig_freq)
# feat_df = feat_manager.extract_features_from_data(samp_data, segment_length=5, segment_overlap=4, signal_frequency=sig_freq)
# feat_df = feat_manager.extract_features_from_data(samp_data, feature_names=[DIHC_FeatureGroup.fdNlPw, DIHC_FeatureGroup.fdNlPwBnd], segment_length=5, signal_frequency=sig_freq)
# feat_df = feat_manager.extract_features_from_data(samp_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.td], segment_length=5, signal_frequency=sig_freq)
# feat_df = feat_manager.extract_features_from_data(samp_data, feature_names=[DIHC_FeatureGroup.tdNlEn, DIHC_FeatureGroup.tdNl], segment_length=5, signal_frequency=sig_freq)
print(f'Feature extraction completed...')

# %%


# %% [markdown]
# ##### Display and save the extracted features

# %%
print(feat_df.shape)
feat_df


# %%
# feat_df.to_csv('./feat_matlab.csv', index=False) 
feat_df.to_csv('./feat_python.csv', index=False) 

# %%


# %% [markdown]
# #### Extract Sample Entropy (SampEn) Profile 
# 
# ##### Create the object of the class "DIHC_FeatureManager" and call the method "extract_sampEn_profile_from_data" to extract Sample entropy (SampEn) profile from the data
# ##### Use different parameters of the method "extract_sampEn_profile_from_data" to extract entropy profile for Sample entropy (SampEn) 

# %%
print(f'Entropy profile extraction started...')
feat_manager = DIHC_FeatureManager()
entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, signal_frequency=sig_freq)
# entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, signal_frequency=sig_freq)
# entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, signal_frequency=sig_freq)
# entProf_df = feat_manager.extract_sampEn_profile_from_data(samp_data, segment_length=5, segment_overlap=0, signal_frequency=sig_freq)
print(f'Entropy profile extraction completed...')

# %%


# %% [markdown]
# #### Display and save extracted Sample entropy (SampEn) profile 

# %%
print(entProf_df.shape)
entProf_df

# %%
# entProf_df.to_csv('./entProf_df_matlab.csv', index=False) 
entProf_df.to_csv('./entProf_df_python.csv', index=False) 

# %%



