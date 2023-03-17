"""
File Name: HumachLab_FeatureDetails.py 
Author: WWM Emran (Emran Ali)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au 
Date: 3/09/2021 7:38 pm
"""

import pandas as pd

### SRART: My modules ###
from DIHC_FeatureManager import *
from DIHC_FeatureManager.DIHC_FeatureExtractor import *
### END: My modules ###


class DIHC_FeatureManager:

    # enumerate

    # ## Initialization
    def __init__(self):
        self.feat_extractor = None
        self.feat_selector = None
        return

    # ## Feature extractor
    def get_features_from_data(self, data, feature_names=[], segment_length=None, segment_overlap=0, signal_frequency=256,
                               filtering_enabled=False, lowcut=1, highcut=48, manage_exceptional_data=0):
        if len(data)==0:
            print(f'Data is empty...')
            return

        sampPS = len(data) if segment_length is None else (segment_length*signal_frequency)

        self.feat_extractor = DIHC_FeatureExtractor(manage_exceptional_data=manage_exceptional_data, signal_frequency=signal_frequency,
                                                    sample_per_second=sampPS, filtering_enabled=filtering_enabled, lowcut=lowcut, highcut=highcut)
        all_feat_df = pd.DataFrame()

        if segment_length is None:
            all_feat_df = self.feat_extractor.get_all_features(data, feature_names)
        else:
            if (segment_length*signal_frequency) > len(data):
                print(f'Data can\'t be segmented...')
                all_feat_df = self.feat_extractor.get_all_features(data, feature_names)
            else:
                seg_st = 0
                seg_len = segment_length*signal_frequency
                seg_mov = seg_len-(segment_overlap*signal_frequency)
                while (seg_st<len(data)):
                    # print('Feature list...', feature_names)
                    seg_end = seg_st+seg_len
                    seg_data = data[seg_st:seg_end]
                    feat_df = self.feat_extractor.get_all_features(seg_data, feature_names)
                    all_feat_df = pd.concat([all_feat_df, feat_df])
                    all_feat_df = all_feat_df.reset_index(drop=True)
                    seg_st += seg_mov

        return all_feat_df


