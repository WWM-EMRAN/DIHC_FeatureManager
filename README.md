# DIHC_Downloader


## Project Name: DIHC_Downloader 
Author: Emran Ali

Involvement: HumachLab (HML) & Deakin- Innovation in Healthcare (DIHC)

Email: wwm.emran@gmail.com, emran.ali@research.deakin.edu.au


## What does it do?
Nested file downloader from nested directories of a web directory

This script contains a class that allows some functions to download files from the web directory when a specific name 
of the web directory is provided. It automatically finds the links to the multilevel nested directories, sorts them 
accordingly and saves them in the corresponding nested directories.

If the download is interrupted then next time with the same parameters it will start downloading from the file it was 
been interrupted, if the download directory contains the already downloaded files and folder structure. 


## Description 
#### Key Task:
Downloader class that manages all the functionalities for downloading files <br> 

#### Insight: <br> 
This class contains all the properties requires to customize the properties of download processes. 
It also contains the functions used for directory traversal and byte movement (downloading). 

###### Properties
-----------
- url_to_download : str

    Base URL of the web directory to download files from
    
- download_directory : str

    Download directory where the files will be stored
###### Optional
---------
- username: str

    If the web directory needs authentication to access the contents
    
- password: str

    If the web directory needs authentication to access the contents
    
- file_types_to_download: list(str)

    List of the specific types of files to download
    
- file_types_not_to_download: list(str)

    List of the specific types of files to to be excluded to download
    
- folder_indicator: list(str)

    Special folder names containing special characters that are usually excluded but expected to be downloaded
    
- url_not_to_consider: list(str)

    Web directory may contain unnecessary links that can be considered as folders but are not and meant to be
    excluded during downloading
    
- is_need_html: bool

    If the web directory contains html file and instead of traversing thru the link to find folders, that html files
    are needed to be downloaded
    
###### Methods
--------

- __ init() __

    Takes- Minimum two of the parameter values from above parameter list | Returns- Object of this class | Func-
    Creates an object with the corresponding parameter values assigned to it
    
- download()

    Takes- none | Returns- none | Func- Traverse thru the web directory to find the nested directories and their
    contents. Downloads them and sort accordingly in the local download directory.


## Application (Code Examples) 
    """ Importing necessary modules
    """
    from DIHC_Downloader import DIHC_Downloader


    """ Test parameter setup for normal web directory
    """
    ### Example-1
    url = 'https://www.physionet.org/files/chbmit/1.0.0/'
    directory = './'
    unusual_folders = ['1.0.0']
    downloader = DIHC_Downloader(url, download_directory=directory, folder_indicator=unusual_folders)


    """ Test parameter setup with web directory that requires authentication credentials
    """
    ### Example-2
    url = 'https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.1/'
    directory = './'
    unusual_folders = ['1.5.1', '../']
    unusual_url = ['/?']
    username = 'nedc_tuh_eeg'
    password = 'nedc_tuh_eeg'
    downloader = DIHC_Downloader(url, download_directory=directory, username=username, password=password, folder_indicator=unusual_folders, url_not_to_consider=unusual_url)

    """ Strat downloading process
    """
    # #%%
    downloader.download()


## About
Version: 0.9.0

Stage: Initial beta










# DIHC_FeatureManager
This Feature Manager project has been developed as part of feature engineering for Machine Learning models. 
It includes the Feature extraction, Feature selection and other feature engineering related functionalities.
[NB: This is not a complete library and we are continuously adding contents in it.] 

## Feature Extraction

### Features:
It contains the following features:
![Problem finding the image...](feature_list.png "List of features.")

### Example Code:
`from DIHC_FeatureManager.DIHC_FeatureManager import *`
`feat_df = feat_manager.get_features_from_data(samp_data, feature_names=DIHC_FeatureGroup.tdNlEn.value, segment_length=5, signal_frequency=sig_freq)`


### Function description
get_features_from_data(self, data, feature_names=[], segment_length=None, segment_overlap=0, signal_frequency=256, filtering_enabled=False, lowcut=1, highcut=48, manage_exceptional_data=0) 




## Credits
Some of the features are derived from the following sources. Appreciate all the authors of these libraries and papers.
1. https://raphaelvallat.com/entropy/build/html/index.html
2. https://doi.org/10.1109/TBME.2018.2808271
3. https://doi.org/10.1371/journal.pone.0193691 


