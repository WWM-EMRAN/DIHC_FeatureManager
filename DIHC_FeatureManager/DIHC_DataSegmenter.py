"""
File Name: HumachLab_FeatureDetails.py
Author: WWM Emran (Emran Ali)
Involvement: HumachLab & Deakin- Innovation in Healthcare (DIHC)
Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au
Date: 3/09/2021 7:38 pm
"""

import pandas as pd

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

### SRART: My modules ###
from DIHC_FeatureManager import *
from DIHC_FeatureManager.DIHC_FeatureExtractor import *
### END: My modules ###


class DIHC_DataSegmenter:

    # ## Initialization
    def __init__(self, data, segment_length=None, segment_overlap=0, signal_frequency=256, prog_bar=None):
        self.data = data
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.signal_frequency = signal_frequency
        self.prog_bar = prog_bar
        return

    # ## Data Segmentor
    def generate_segments(self):
        if len(self.data)==0:
            print(f'Data is empty...')
            exit(0)
            # return

        if self.segment_length is None:
            print(f'Dealing with entire signal...')
        else:
            if (self.segment_length*self.signal_frequency) > len(self.data):
                print(f'Data can\'t be segmented...')
                return self.data
            else:
                seg_srl = 1
                seg_st = 0
                seg_len = int(self.segment_length*self.signal_frequency)
                seg_mov = int(seg_len-(self.segment_overlap*self.signal_frequency))
                num_segs = max(0, int((len(self.data)-seg_len)/seg_mov +1) )
                # print(f'---->> {seg_srl}, {seg_st}, {seg_len}, {seg_mov}')
                # print(f'Segmentation started...')
                if self.prog_bar is None:
                    self.prog_bar = tqdm(total=num_segs, desc=f'Segmentation started...')
                else:
                    self.prog_bar.set_description(f'Segmentation started...')
                while (seg_st<len(self.data)):
                    # print(f'Generating segment# {seg_srl}')
                    self.prog_bar.set_description(f'Generating segment# {seg_srl}')
                    seg_end = seg_st+seg_len
                    if seg_end>len(self.data):
                        seg_end = len(self.data) 
                        # break 
                    # print(f'====> {type(seg_st)} = {type(seg_end)}')
                    seg_data = self.data[int(seg_st):int(seg_end)]
                    yield seg_data
                    seg_st += seg_mov
                    seg_srl += 1
                # print(f'Segmentation finished...')
                self.prog_bar.set_description(f'Segmentation finished...')
        return


