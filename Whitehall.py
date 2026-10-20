# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:52:04 2025

@author: coleb


General guidelines for navigating the files.

[Data]
There are multiple files for a single recording block with the same prefix.
1. Files collected during the experiments: ns5, ns2, nev, pkl, and hdf.
2. Files generated after the experiments: mat. 

[Filename prefix]: SUBJYYYYMMDD_NN_teXXXX
- SUBJ: first 4 characters of the subject names: airp for Airport, braz for Brazos
- YYYYMMDD: date for the recording.
- NN: the number of recording on that day that starts with 01 for a new session.
- XXXX: incremental unique ID for each recording.

[What do these files mean]
1. ns5: Analog signal collected at 30 kHz from Ripple. Used to sync hdf and Ripple signal.
2. ns2: LFP signal collected at 1 kHz.
3. nev: Spiking information such as waveforms and spike times.
4. hdf: Behavioral data generated from BMI3D.
5. mat: Synced time stamps between Ripple and behaviors.
6. pkl: Decoder files used in the actual BMI.

[What do we need to start processing data]
- We need the time stamps for spikes and behaviors, as well as waveforms.
- The sole purpose of the ns5 files (which are usually large) is to generate the mat files.
  If mat files already exists, discard the ns5 files; if mat files do not exist, run Sync(session.ns5)
- After extracting spike times and waveforms, the nev can be removed.
- Therefore, we need nev, mat, and hdf files for spike processing; include ns2 for LFP processing.
        

"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#TODO  
# te1888 and te1927 have units with 300 Hz FRs?? There's gotta be an error somewhere
# lfp ch numbers and/or elec ids?? are they consistent across recs? Which set of chs for which area?+
# turn lfp snips out into a df for better labelling ?
# can only do volatile blocks first atm for some things
# make q learning learning rate be fit separately for stable and volatile blocks
# plot values for with half of trials per axis
# rerun lasso grid search with new value
# go thru and turn internal methods to start with _
# change the bandpass filtering to be done on the whole array of snips using axis arg in sosfiltfilt



import os
import sys
import gif
# import copy
import time
# import math
# import pickle
import sklearn
# import neurodsp
# import matplotlib
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib_venn
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from numpy.polynomial.polynomial import polyfit
# from statsmodels.stats.outliers_influence import variance_inflation_factor


#### Libraries and Paths

# PROJ_FOLDER = r"D:\Whitehall"
PROJ_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis"
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
# PROJ_FOLDER = r"F:\cole"
# BMI_FOLDER = r"C:\Users\crb4972\Desktop\bmi_python"
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns') #Neuroshare python
NSX_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'blackrock')
DATA_FOLDER = os.path.join(PROJ_FOLDER, 'data')

# Add paths for necessary packages to search path
sys.path.insert(1,BMI_FOLDER) 
sys.path.insert(2, NS_FOLDER) 
sys.path.insert(3,os.path.join(PROJ_FOLDER,'utils'))

# import riglib.ripple.pyns.pyns.nsparser
from riglib.blackrock.brpylib import NsxFile
# from riglib.ripple.pyns.pyns.nsparser import ParserFactory
# from riglib.ripple.pyns.pyns.nsexceptions import NeuroshareError, NSReturnTypes
# from riglib.ripple.pyns.pyns.nsentity import AnalogEntity, SegmentEntity, EntityType, EventEntity, NeuralEntity

# os.chdir(NS_FOLDER)
from nsfile import NSFile
    
os.chdir(PROJ_FOLDER)

import Whitehall_utils as utils
from RishiValueModeling import ValueModelingClass
from file_info import get_block_info, get_area_info
import DecisionMakingBehavior_Whitehall as BehaviorAnalysis
from TwoStageQLearning_DualBaselineSide import Calc_DistrQlearning_2Targs_DualBaselineSide
from TwoStageQLearning_BaselineSide import Calc_DistrQlearning_2Targs_BaselineSide


#%% Main code block

#### Define global things

# Constants
TEST = False
SUBSESSIONS = ['all trials','stable block','volatile block']

# SESSIONS = ['braz20240927_01_te5384',
#             'braz20241001_03_te5390',
#             'braz20241002_04_te5394',
#             'braz20241004_02_te5396',
#             #'braz20250221_03_te1873', #I can't remember why this is commented out
#             'braz20250225_04_te1880',
#             #'braz20250228_03_te1888', #has units with 300 Hz FR ?
#             #'braz20250327_04_te1927', #same ^
#             #'braz20250326_04_te1923' #commented out for code to run quicker
#             ]

# SESSIONS = [ # 11 sessions with spike and lfp data for prelim testing
#             "airp20250919_02_te2177",  
#             "airp20251015_04_te2206",
#             "airp20251016_03_te2209",
#             "airp20251020_05_te2214",
#             "airp20251021_02_te2216",
#             "airp20251023_03_te2219",
#             "airp20251028_03_te2226",
#             "airp20251029_05_te2231",
#             "airp20251030_02_te2233",
#             "airp20251104_02_te2242", 
#             "airp20251111_02_te2250"
#             ]

# SESSIONS = [ # 35 sessions. Full dataset as of 5/1/26
#             "airp20250602_03_te2003",
#             "airp20250604_08_te2011",
#             "airp20250617_03_te2031",
#             "airp20250618_03_te2034",
#             "airp20250626_03_te2049",
#             "airp20250701_06_te2058",
#             "airp20250826_05_te2135",
#             "airp20250904_03_te2147",
#             "airp20250910_02_te2156",
#             "airp20250912_02_te2160",
#             "airp20250919_02_te2177",
#             "airp20251007_02_te2198",
#             "airp20251008_02_te2200",
#             "airp20251015_04_te2206",
#             "airp20251016_03_te2209",
#             "airp20251020_05_te2214",
#             "airp20251021_02_te2216",
#             # "airp20251023_03_te2219", no good units
#             "airp20251028_03_te2226",
#             "airp20251029_05_te2231",
#             "airp20251030_02_te2233",
#             "airp20251104_02_te2242",
#             "airp20251111_02_te2250",
#             # "airp20260224_03_te2287", no good units
#             "airp20260302_03_te2302",
#             "airp20260303_02_te2304",
#             "airp20260305_12_te2321",
#             # "airp20260310_03_te2324", no good units and Cd probe was lifted out. do not use for LFP either.
#             # "airp20260311_03_te2327", no good units
#             # "airp20260403_03_te2337", no good units
#             "airp20260406_03_te2340",
#             "airp20260408_03_te2343",
#             # "airp20260409_03_te2346", no good units
#             "airp20260414_03_te2352",
#             "airp20260415_03_te2355"
#             ]

# SESSIONS = [ #since reaiming and new s-probe
#             "airp20260708_05_te2496",
#             "airp20260630_04_te2468",
#             "airp20260625_05_te2453",
#             "airp20260624_06_te2448"
#             ]

# SESSIONS = [ #all airport
#             "airp20250602_03_te2003",
#             # "airp20250604_08_te2011", too few trials
#             "airp20250617_03_te2031",
#             "airp20250618_03_te2034",
#             "airp20250626_03_te2049",
#             "airp20250701_06_te2058",
#             "airp20250826_05_te2135",
#             "airp20250904_03_te2147",
#             "airp20250910_02_te2156",
#             "airp20250912_02_te2160",
#             "airp20250919_02_te2177",
#             "airp20251007_02_te2198",
#             "airp20251008_02_te2200",
#             # "airp20251015_04_te2206", too few trials
#             "airp20251016_03_te2209",
#             "airp20251020_05_te2214",
#             "airp20251021_02_te2216",
#             "airp20251023_03_te2219", #no good units
#             "airp20251028_03_te2226",
#             "airp20251029_05_te2231",
#             "airp20251030_02_te2233",
#             "airp20251104_02_te2242",
#             "airp20251111_02_te2250",
#             "airp20260224_03_te2287", #no good units
#             "airp20260302_03_te2302",
#             # "airp20260303_02_te2304", too few trials
#             "airp20260305_12_te2321",
#             "airp20260310_03_te2324", #no good units and Cd probe was lifted out. do not use for LFP either.
#             "airp20260311_03_te2327", #no good units
#             "airp20260403_03_te2337", #no good units
#             "airp20260406_03_te2340",
#             "airp20260408_03_te2343",
#             #"airp20260409_03_te2346", #no good units, did not finish task
#             "airp20260414_03_te2352",
#             "airp20260415_03_te2355",
#             "airp20260708_05_te2496",
#             "airp20260630_04_te2468",
#             "airp20260625_05_te2453",
#             "airp20260624_06_te2448",
#             ]

SESSIONS = [
            "braz20260728_03_te2536",
            "braz20260729_03_te2539",
            "braz20260730_03_te2542",
            "braz20260731_03_te2545",
            "braz20260803_03_te2548",
            "braz20260804_03_te2551",
            "braz20260805_03_te2554",
            "braz20260806_03_te2557",
            "braz20260810_03_te2560",
            # "braz20260811_03_te2563", #too few trials (1044), excluded
            # "braz20260812_03_te2566", #too few trials (939), excluded
            "braz20260813_03_te2569", 
            "braz20260814_03_te2572", 
            "braz20260817_03_te2575", 
            "braz20260818_03_te2578", 
            "braz20260821_03_te2581", 
            "braz20260824_03_te2584", 
            "braz20260827_03_te2587",
            # "braz20260828_03_te2590", #too few trials (1039), excluded
            "braz20260901_03_te2593",
            "braz20260902_03_te2596",
            "braz20260908_03_te2599",
            "braz20260909_03_te2602",
            "braz20260910_03_te2605",
            "braz20260911_03_te2608",
            "braz20260915_05_te2613",
            "braz20260916_03_te2616",
            # "braz20260918_03_te2619",
            # "braz20260924_03_te2622",
            # "braz20260925_03_te2625",
            # "braz20261001_03_te2628",
            # # "braz20261002_05_te2633", #too few trials (747), excluded
            # "braz20261007_03_te2636",
            # # "braz20261008_03_te2639", #too few trials (845), excluded
            # # "braz20261009_03_te2642", #too few trials (1041), excluded
            ]

REGRESSORS = [
              'Q1',
              'Q2',
              'Choice1',
              # 'Qhigh',
              'Qchosen',
              # 'Qdiff_',
              # 'absQdiff',
              # 'Choice_high',
              'Side',
              'Time',
              ]

FREQ_BANDS = {'Theta':(3.5,8.5), # Define freq bands (Hz)
              'Alpha':(8.5,12.5),
              'Beta':(12.5,30.5),
              # 'Gamma':(30.5,200.5),
              'Low Gamma':(30.5,60.5),
              'High Gamma':(60.5,200.5)}

AREAS = ['Cd','OFC']

REW_PROBS = [0., 0.2, 0.4, 0.6, 0.8, 1.]

VALUE_MODEL = 'DualBaseline'
# VALUE_MODEL = 'DualBaselineSide'

ALIGNMENT = 'Targets On' 
# Options are 'Targets On', 'Choice Made', and 'Reward' 
# These correspond to targets appearing, choice being made, and reward info disbursement, respectively

HOLD_DURATION = 0.6 # how long the actual center_hold time is [s]
ENCODING_WINDOW = (0.0,0.6) #window over which neuron encoding is determined [s]

T_BEFORE = 0.1 #how far to look before alignment point [s]
T_AFTER = 0.7 #how far to look after alignment point [s]
T_RES = 0.05 #time resolution ; size of time bins [s]
T_VECTOR = np.round(np.arange(-T_BEFORE+T_RES, T_AFTER, T_RES),decimals=2)
N_TIMEBINS = len(T_VECTOR)

# LEARNING_RATES = np.linspace(start=0.01,stop=0.99,num=50) #used for single learning rate
LEARNING_RATES = np.linspace(start=0.05,stop=0.95,num=19) #used for dual learning rates

# REGRESSORS = ['Q1','Q2','Choice1',
#               'Qlow','Qhigh','Qdiff','absQdiff',
#               'Choice_low','Side','Time']

#Since Q1+Q2=1 and Qlow+Qhigh=1, only one regressor of each pair is needed.
# REGRESSORS = ['Q1','Choice1',
#               'Qhigh','Qdiff_','absQdiff',
#               'Choice_low','Side','Time']

ALPHA_THRESHOLD = 0.05

USE_SLIDE = False
FR_INCLUSION_NUM_TRIALS_SLIDE_WINDOW = 20 #num trials for sliding average for spike min FR inclusion criteria
MIN_FR_THRESH = 1 #Hz
MIN_NUM_TRIALS = 1080 #trials

COLORS = ['red','blue','gold','green','orange','purple']
LINESTYLES = ['-','--',':','-.',(0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10))]
MARKERSTYLES = ['o','^','s','*','d','P','X','p','H','<','>']
HATCHSTYLES = ['','/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
# STABLE_COLOR, VOLATILE_COLOR = 'blue','orange'


#### Classes

class ProcessSpikes:
    '''
    From .mat, .hdf, and .nev files, load spike times and calculate firing rate 
    aligned to task events.
    
    Output: DataFrame with firing rate for each unit for each trial
    '''
    
    def __init__(self, session: str, verbose=True):

        print('\n')
        print('*'*25)
        print(session)
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session, self.session)
        self.verbose=verbose
        
        ## Data
        self.has_hdf = False # For behavior
        self.has_ns5 = False # For syncing 
        self.has_ns2 = False # For LFP
        self.has_nev = False # For waveform 
        self.has_decoder = False # For decoder
        self.has_mat = False # For syncing
        self.has_pkl = False # For spike times
        self.check_files()
        
        ## Get Spike Times
        self.spike_times = None 
        self.unit_labels = None
        # if self.has_pkl:
        #     self.load_spike_times()
        # else:
        #     print('Spikes not sorted!!!!!!!!')
        #     self.get_spike_times()
        self.load_spike_times()
        
        ## Get Times Align
        self.times_align = None
        self.get_times_align()
        
        ## Get Firing Rates and PSTHs
        self.fr_df = None #zscored
        self.raw_fr_df = None #not zscored
        self.psth_dict = None #zscored
        self.psth_df = None #zscored
        self.excluded_units = []
        self.psth_t_vector = T_VECTOR
        self.alignment = ALIGNMENT
        self.get_psth()
        
        ## Save Out Processed Data
        # self.dict_out = {'Session':session, 'fr_df':self.fr_df, 'psth_dict':self.psth_dict, 'psth_t_vector':T_VECTOR}
        self.dict_out = {'Session':session, 'psth_dict':self.psth_dict, 'psth_t_vector':T_VECTOR, 'alignment':ALIGNMENT}
        # utils.save_pkl(self, 'Spikes', self.session,DATA_FOLDER)
        print('*'*25)
        

    def check_files(self):
        """
        For a proper analysis to be done,
        the hdf, nev, mat, and pkl files are essential.
        The ns5 and ns2 are optional.
        """
        
        if os.path.exists(self.file_prefix + '.hdf'):
            self.has_hdf = True
        if os.path.exists(self.file_prefix + '.ns5'):
            self.has_ns5 = True
        if os.path.exists(self.file_prefix + '.ns2'):
            self.has_ns2 = True
        if os.path.exists(self.file_prefix + '.nev'):
            self.has_nev = True
        if os.path.exists(self.file_prefix + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix + '_KFDecoder.pkl'):
            self.has_decoder = True
        if os.path.exists(self.file_prefix + '_spike_times_dict.pkl'):
            self.has_pkl = True
        
          
        
    def get_spike_times(self):
        
        assert self.has_nev, FileNotFoundError(f'.nev file not found! Session: {self.session}')

        ## Get Spike Times
        print('Loading spike times..')
        self.nevfile = NSFile(self.file_prefix + '.nev')
        spike_entities = [e for e in self.nevfile.get_entities() if e.entity_type==3]
        headers = np.array([s.get_extended_headers() for s in spike_entities]) #get info for each ch
        # [print(i,h[b'NEUEVLBL'].label[:7]) for i,h in enumerate(headers) if b'NEUEVLBL' in h.keys()]
        unit_idxs = np.nonzero([h[b'NEUEVWAV'].number_sorted_units for h in headers])[0] #get ch idxs where there is a sorted 
        unit_idxs = [unit_idx for unit_idx in unit_idxs if b'NEUEVLBL' in headers[unit_idx].keys()] #exclude entities without NEUEVLBL field
        if TEST: unit_idxs=unit_idxs[:3] #to make runtime shorter for testing
        self.num_units = len(unit_idxs)
        self.unit_labels = ["Unit " + h[b'NEUEVLBL'].label[:7].decode() + self.session for h in headers[unit_idxs] ] #get labels of all sorted units
        # num_units_per_idx = [h[b'NEUEVWAV'].number_sorted_units for h in headers[unit_idxs]]
        [print(f'More than one unit found for {h[b"NEUEVLBL"].label[:7]}!') for h in headers[unit_idxs] if h[b'NEUEVWAV'].number_sorted_units > 1]
        recording_duration = self.nevfile.get_file_info().time_span # [sec]


        self.spike_times = [] #each element is a list spike times for a sorted unit
        spike_waveforms = [] #each element is a list of waveforms for a sorted unit
        for i,unit_idx in enumerate(unit_idxs): #loop thru sorted unit
            unit = spike_entities[unit_idx]
            self.spike_times.append([]) #initiate list of spike times for this unit
            spike_waveforms.append([]) #initiate list of spike waveforms for this unit
            
            for spike_idx in range(unit.item_count):
                self.spike_times[i].append(unit.get_segment_data(spike_idx)[0])
                spike_waveforms[i].append(unit.get_segment_data(spike_idx)[1])
            
            if self.verbose:
                print(f'{self.unit_labels[i]} - Avg FR: {unit.item_count/recording_duration:.2f} Hz. ({i+1}/{len(unit_idxs)})')
        print('All spike times loaded!')
        self.nevfile = None
        
        # Save out spike times so we don't need to load them from nev again
        spike_times_dict = {'unit_labels':self.unit_labels, 'spike_times':self.spike_times, 'spike_waveforms': spike_waveforms}
        utils.save_pkl(spike_times_dict,'spike_times_dict',self.session,DATA_FOLDER)
        
        return


    def load_spike_times(self):
        
        # Load dict of previously saved spike times
        #spike_times_dict = utils.load_pkl('spike_times_dict',self.session,DATA_FOLDER) #OLD WAY USING ABOVE GET SPIKE TIMES METHOD
        try:
            spike_times_dict = utils.load_pkl_2('goodunits_spike_times_dict',self.session,DATA_FOLDER)
        except:
            print('Spikes not manually sorted!!!!!!!!!!!!!!')
            self.get_spike_times()
            spike_times_dict = utils.load_pkl('spike_times_dict',self.session,DATA_FOLDER)
            
        
        self.unit_labels = spike_times_dict['unit_labels']
        self.spike_times = spike_times_dict['spike_times']
        self.num_units = len(self.unit_labels)
            
        #print out FRs for funsies
        if self.verbose:
            for i,unit_spikes in enumerate(self.spike_times): #loop thru sorted unit
                print(f'{self.unit_labels[i]} - Avg FR: {len(unit_spikes)/(unit_spikes[-1]-unit_spikes[0]):.2f} Hz. ({i+1}/{len(self.unit_labels)})')
        
        return
    

    def apply_inclusion_criteria(self, psth):
                
        win = (T_VECTOR>=ENCODING_WINDOW[0]) * (T_VECTOR<=ENCODING_WINDOW[1]) #to get num_spikes only over defined window
        fr = np.mean(psth[:,win],axis=1) #len = num_trials
        sliding_avg = utils.trial_sliding_avg(fr, FR_INCLUSION_NUM_TRIALS_SLIDE_WINDOW)
        
        if self.verbose:
            print(f'Hold avg FR: {np.mean(fr):.2f} Hz')
            print(f'Num holds with low fr: {sum(fr<0.1)}')
        
        #sliding avg FR
        if USE_SLIDE:
            if np.all(sliding_avg >= MIN_FR_THRESH):
                return True
            else:
                if self.verbose: print('EXCLUDED')
                return False
        
        # overall avg FR of holds
        if not USE_SLIDE:
            if np.mean(fr)>=MIN_FR_THRESH:
                return True
            else:
                if self.verbose: print('EXCLUDED')
                return False
        
        
        return
    
    def get_times_align(self):
        '''
        Gets the array of indices (sample numbers) corresponding to the hold_center, 
        target, and check_reward time points of the given session.
        This facilitates time-aligned analyses.
        
        Parameters
        ----------
        hdf_files : list of hdf files for a single session
        syncHDF_files : list of syncHDF_files files which are used to make the alignment between behavior data and spike data
            
        Outputs
        -------
        target_hold_TDT_ind : 1D array containing the TDT indices for the target hold onset times of the given session
        '''
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        assert self.has_mat, FileNotFoundError(f'.mat file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'
        self.syncHDF_file = self.file_prefix + '_syncHDF.mat'
        
        fs_hdf = 60 #hdf fs is always 60
        
        # load behavior data
        cb = BehaviorAnalysis.ChoiceBehavior_Whitehall([self.hdf_file]) #method needs hdf filenames in a list
        self.num_trials = cb.num_successful_trials
        
        # Find times for successful trials, depending on times_align reference point chosen
        match ALIGNMENT:
            case 'Targets On': #targets appearing, 'hold_center'
                ind = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
                
            case 'Choice Made': #choice made, 'target'
                ind = cb.ind_check_reward_states - 2 # times corresponding to target hold onset
            
            case 'Reward': #reward/no reward info dispensed to subject, 'check_reward'
                ind = cb.ind_check_reward_states #times corresponding to reward period onset

        
        # align spike tdt times with hold center hdf indices using syncHDF files
        time_align_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind,cb.state_time,self.syncHDF_file)

        # Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
        assert len(time_align_TDT_ind) == len(ind), f'Repeat hold times! Session: {self.session}'
        assert len(time_align_TDT_ind) == self.num_trials

        self.times_align = time_align_TDT_ind / DIO_freq

        print('Alignment loaded!')

        return 
        
        
    def get_firing_rate_old(self):
        '''
        Count how many spikes occured in the time window of interest and then divide by window length.
        Time window defined as [time_align - t_before : time_align + t_after]

        Returns
        -------
        None.

        '''

        ## Get Spike Counts
        print('Binning and counting spikes..')
        firing_rates = np.zeros((self.num_trials,self.num_units))
        for trial in range(self.num_trials):
            
            win_begin = self.times_align[trial] - T_BEFORE
            win_end = self.times_align[trial] + T_AFTER

            for i in range(self.num_units):
                
                unit_spikes = np.array(self.spike_times[i])
                num_spikes = sum( (unit_spikes>win_begin) & (unit_spikes<win_end) )
                
                firing_rates[trial,i] = num_spikes / (T_BEFORE + T_AFTER)
                
        print('Done counting spikes!')

        self.fr_df = pd.DataFrame(sp.stats.zscore(firing_rates,axis=0),columns=self.unit_labels)
                
        return 
    
    def get_firing_rate(self):
        '''
        Count how many spikes occured in the time window of interest and then divide by window length.
        Time window defined as [time_align - t_before : time_align + t_after]

        Returns
        -------
        None.

        '''

        ## Get Spike Counts
        print('Binning and counting spikes..')
        firing_rates = np.zeros((self.num_trials,self.num_units))
        for trial in range(self.num_trials):
            
            win_begin = self.times_align[trial] - T_BEFORE
            win_end = self.times_align[trial] + T_AFTER

            for i in range(self.num_units):
                
                unit_spikes = np.array(self.spike_times[i])
                num_spikes = sum( (unit_spikes>win_begin) & (unit_spikes<win_end) )
                
                firing_rates[trial,i] = num_spikes / (T_BEFORE + T_AFTER)
                
        print('Done counting spikes!')

        self.fr_df = pd.DataFrame(sp.stats.zscore(firing_rates,axis=0),columns=self.unit_labels)
                
        return
    
    
    def get_psth(self):
        '''
        asdfsa

        Returns
        -------
        None.

        '''

        ## Loop thru units and use compute_psth
        print('Getting PSTHs..')
        # psth_length = int(np.rint((T_BEFORE + T_AFTER)/T_RES)) #num_timepoints
        # psths = np.zeros((self.num_units, self.num_trials, psth_length))
        self.psth_dict = dict()
        psth_df_dict = dict()
        fr_df_dict = dict()
        raw_fr_df_dict = dict()
        
        assert ALIGNMENT == 'Targets On', 'FR only set up for hold_center atm'
        # win = (T_VECTOR>=0) * (T_VECTOR<=HOLD_DURATION) #to get num_spikes only over actual hold period
        win = (T_VECTOR>=ENCODING_WINDOW[0]) * (T_VECTOR<=ENCODING_WINDOW[1]) #to get num_spikes only over defined window
        
        for unit,unit_label in enumerate(self.unit_labels):
            
            unit_spikes = np.array(self.spike_times[unit])
            psth = self.compute_psth(unit_spikes) #psth = (num_trials x num_timepoints)
            
            #see if unit passes inclusion criteria
            if self.verbose:
                print(unit_label)
                print(f'Overall avg FR: {len(unit_spikes)/(unit_spikes[-1]-unit_spikes[0]):.2f} Hz')
            include_unit_flag = self.apply_inclusion_criteria(psth)
            
            if self.verbose:
                fig,axs=plt.subplots(2,1,layout="constrained")
                ax=axs[0]#fr vs trials
                ax.plot(np.mean(psth[:,win],axis=1))
                ax.set_xlabel('Trial')
                ax.set_ylabel('Hold center FR')
                ax.set_title('FR vs time')
                ax=axs[1] #hist
                ax.hist(np.mean(psth[:,win],axis=1),bins=np.arange(0,10,0.1))
                ax.set_xlabel('Hold center FR')
                ax.set_ylabel('Num trials')
                ax.set_title('FR Hist')
                fig.suptitle(f'{unit_label}\nIncluded: {include_unit_flag}')
                
            #only include current unit if it passed inclusion criteria
            if include_unit_flag:
                self.psth_dict[unit_label] = sp.stats.zscore(psth,axis=0)
                psth_df_dict[unit_label] = np.ravel(sp.stats.zscore(psth,axis=0))
                fr_df_dict[unit_label] = sp.stats.zscore(np.mean(psth[:,win],axis=1)) # shape = 1 x n_trials
                raw_fr_df_dict[unit_label] = np.mean(psth[:,win],axis=1) # shape = 1 x n_trials
                assert fr_df_dict[unit_label].shape[0] == self.num_trials
            else:
                self.excluded_units.append(unit_label)
            
        
            
        if self.verbose:
            print('Units excluded:')
            print(self.excluded_units)
            print('-'*20)
            
        print(f'{len(self.excluded_units)}/{self.num_units} units excluded.')
        print('Done making PSTHs!')
        
        # self.psth_dict: keys=unit_labels, values=psth arrays (trials x timepoints)
        
        self.psth_df = pd.DataFrame.from_dict(psth_df_dict,orient='columns') #df of size (n_trials * n_timepoints) x n_units
        self.fr_df = pd.DataFrame.from_dict(fr_df_dict,orient='columns') #df of size n_trials x n_units
        self.raw_fr_df = pd.DataFrame.from_dict(raw_fr_df_dict,orient='columns') #df of size n_trials x n_units
                
        return   
    
    
    def compute_psth(self,unit_spikes):
        '''
        Method that returns an array of psths for spiking activity aligned to the sample numbers indicated in samples_align
        with firing rates quantized to bins of size samp_resolution.

        Input:
        - unit_spikes: arr of all spike times

        Output: 
        - psth: T x N array containing the average firing rate over a window of total length N samples for T different
                time points (trials)
        '''        
        
        # Add 0.5 sec both before and after in order to incorporate spiking info on either side of the actual window 
        # for smoothing purposes. PSTH is trimmed back down to proper length after smoothing
        extra_t_before = 0.5
        extra_t_after = 0.5
        psth_length = int(np.rint((T_BEFORE + T_AFTER + extra_t_before + extra_t_after)/T_RES))
        num_timepoints = len(self.times_align)
        psth = np.zeros((num_timepoints, psth_length-1))
        # psth = np.zeros((num_timepoints, psth_length))
        smooth_psth = np.zeros(psth.shape)

        #set smoothing parameters
        # boxcar_length = 5
        # boxcar_window = boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
        b = sp.signal.windows.gaussian(39, 1)

        for i, tp in enumerate(self.times_align):
            data = unit_spikes
            t_window = np.arange(tp - T_BEFORE - extra_t_before, tp + T_AFTER + extra_t_after, T_RES)
            hist, bins = np.histogram(data, bins = t_window)
            hist_fr = hist/T_RES
            psth[i,:] = hist_fr[:psth_length-1]
            # psth[i,:] = hist_fr[:psth_length]
            #smooth_psth[i,:] = np.convolve(hist_fr[:psth_length-1], boxcar_window,mode='same')/boxcar_length
            smooth_psth[i,:] = sp.ndimage.convolve1d(hist_fr[:psth_length-1], b/b.sum())
            # smooth_psth[i,:] = sp.ndimage.convolve1d(hist_fr[:psth_length], b/b.sum())
       
        extra_time_bins_before = int(np.rint(extra_t_before / T_RES))
        extra_time_bins_after = int(np.rint(extra_t_after / T_RES))
        
        smooth_psth_out = smooth_psth[:, extra_time_bins_before:-extra_time_bins_after]
        assert len(T_VECTOR) == np.shape(smooth_psth_out)[1], f'{len(T_VECTOR)} vs {np.shape(smooth_psth_out)[1]}'
        
        return smooth_psth_out



class ProcessLFP:
    '''
    From .mat, .hdf, and .ns2 files, load LFP data aligned to task events.
    
    Output: DataFrame with LFP trace for each ch for each trial
    '''
    
    def __init__(self, session: str, verbose=False):

        
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session, self.session)

        ## Data
        self.has_hdf = False # For behavior
        self.has_ns5 = False # For syncing 
        self.has_ns2 = False # For LFP
        self.has_nev = False # For waveform 
        self.has_decoder = False # For decoder
        self.has_mat = False # For syncing
        self.has_pkl = False # For spike times
        self.check_files()
        
        ## Load LFP data
        assert self.has_ns2, FileNotFoundError(f'.ns2 file not found! Session: {self.session}')
        
        ## Get Times Align
        self.times_align = None #(secs)
        self.get_times_align_ns2()
        
        ## Get LFP snippets for Desired Epoch
        
        self.t_vector = T_VECTOR
        self.alignment = ALIGNMENT
        self.snips = self.get_lfp_snippets(self.times_align,T_BEFORE,T_AFTER) # (trials x chs x samples)
        # self.snips_dict = {'data':self.snips,'t_before':T_BEFORE,'t_after':T_AFTER,'alignment':ALIGNMENT}
        
        ## Define chs for each Area
        self.session_areas = get_area_info(self.session)
        self._get_chs()
        # assert self.n_chs == 96, f'{self.n_chs} chs!'
        

        ## Save Out Processed Data
        # self.dict_out = {'Session':session, 'snips_dict':self.snips_dict,
        #                  'fs':self.fs, 'n_chs':self.n_chs, 'n_trials':self.n_trials,
        #                  'session_areas':self.session_areas}
        
        self.all_bandpowers = self.get_all_bandpowers()
        del self.snips # to free up ~0.5gb memory. Snips aren't used elsewhere in code so far
        # utils.save_pkl(self, 'LFP', self.session,DATA_FOLDER)
        
        
    def _get_chs(self):
        
        has_vmPFC = 'vmPFC' in self.session_areas
        has_Cd = 'Cd' in self.session_areas
        has_OFC = 'OFC' in self.session_areas
        
        #vmPFC
        if has_vmPFC:
            self.vmPFC_chs = np.arange(0,32)
        
        #Cd
        if has_vmPFC and has_Cd:
            self.Cd_chs = np.arange(32,64)
            
        elif not has_vmPFC and has_Cd:
            self.Cd_chs = np.arange(0,32)
        
        #OFC
        if has_vmPFC and has_Cd and has_OFC:
            self.OFC_chs = np.arange(64,96)
        elif (has_vmPFC ^ has_Cd) and has_OFC: #  ^ = XOR
            self.OFC_chs = np.arange(32,64)
        elif not has_vmPFC and not has_Cd and has_OFC:
            self.OFC_chs = np.arange(0,32)
            
        return
    
    
    def check_files(self):
        """
        For a proper analysis to be done,
        the hdf, nev, mat, and pkl files are essential.
        The ns5 and ns2 are optional.
        """
        
        if os.path.exists(self.file_prefix + '.hdf'):
            self.has_hdf = True
        if os.path.exists(self.file_prefix + '.ns5'):
            self.has_ns5 = True
        if os.path.exists(self.file_prefix + '.ns2'):
            self.has_ns2 = True
        if os.path.exists(self.file_prefix + '.nev'):
            self.has_nev = True
        if os.path.exists(self.file_prefix + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix + '_KFDecoder.pkl'):
            self.has_decoder = True
        if os.path.exists(self.file_prefix + '_spike_times_dict.pkl'):
            self.has_pkl = True

        
    def get_times_align_ns2(self):
        '''
        Gets the array of indices (sample numbers) corresponding to the hold_center, 
        target, and check_reward time points of the given session.
        This facilitates time-aligned analyses.
        
        Parameters
        ----------
        hdf_files : list of hdf files for a single session
        syncHDF_files : list of syncHDF_files files which are used to make the alignment between behavior data and spike data
            
        Outputs
        -------
        target_hold_TDT_ind : 1D array containing the TDT indices for the target hold onset times of the given session
        '''
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        assert self.has_mat, FileNotFoundError(f'.mat file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'
        self.syncHDF_file = self.file_prefix + '_syncHDF.mat'
        
        fs_hdf = 60 #hdf fs is always 60
        
        # load behavior data
        cb = BehaviorAnalysis.ChoiceBehavior_Whitehall([self.hdf_file]) #method needs hdf filenames in a list
        self.n_trials = cb.num_successful_trials
        
        # Find times for successful trials, depending on times_align reference point chosen
        match ALIGNMENT:
            case 'Targets On': #targets appearing, 'hold_center'
                ind = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
                
            case 'Choice Made': #choice made, 'target'
                ind = cb.ind_check_reward_states - 2 # times corresponding to target hold onset
            
            case 'Reward': #reward/no reward info dispensed to subject, 'check_reward'
                ind = cb.ind_check_reward_states #times corresponding to reward period onset
        
        
        # align spike tdt times with hold center hdf indices using syncHDF files
        time_align_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind,cb.state_time,self.syncHDF_file)

        # Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
        assert len(time_align_TDT_ind) == len(ind), f'Repeat hold times! Session: {self.session}'
        assert len(time_align_TDT_ind) == self.n_trials

        self.times_align = time_align_TDT_ind / DIO_freq #convert from samples (30k sampling rate btw) to seconds        

        print('Alignment loaded!')

        return 
        
        
    def get_lfp_snippets(self,times_align,t_before=0.5,t_after=0.5):
        '''
        Count how many spikes occured in the time window of interest and then divide by window length.
        Time window defined as [time_align - t_before : time_align + t_after] (units are seconds)
        

        Returns
        -------
        None.

        '''

        print('Getting LFP Snippets..')
        self.ns2file = NsxFile(self.file_prefix + '.ns2')
        self.fs = 1000.
        assert self.ns2file.basic_header['Label'] == '1 ksamp/sec'
        
        #convert from secs to samples
        samps_before = np.round(self.fs * t_before).astype(int)
        samps_after = np.round(self.fs * t_after).astype(int)
        samps_align = np.round(self.fs * times_align).astype(int)
        
        data = self.ns2file.getdata()['data'] # (chs x samples) [uV]
        self.n_chs = np.shape(data)[0]
        snippets = np.zeros((len(samps_align), self.n_chs, samps_before+samps_after)) # (trials x chs x samples)

        #Loop thru trials
        for trial, samp_align in enumerate(samps_align):
        
            #Get window of samples for current trial (all units are now samples)
            samp_begin = samp_align - samps_before
            samp_end = samp_align + samps_after
            
            #Get time-aligned snippet for current trial from LFP data array. Filter out line noise too
            snippets[trial,:,:] = self.filter_line_noise(data[:,samp_begin:samp_end])
        
        self.ns2file.close()
        self.ns2file = None
        print('LFP Snippets compiled!')       
        return snippets # (trials x chs x samples)


    def filter_line_noise(self,sig):
        
        #set up to notch filter out 60,120,180 Hz line noise
        Q=30 #quality factor
        b60,a60 = sp.signal.iirnotch(60.,Q,self.fs) #this is done after downsampling would be done
        b120,a120 = sp.signal.iirnotch(120.,Q,self.fs)
        b180,a180 = sp.signal.iirnotch(180.,Q,self.fs)
        
        # Filter line noise    
        sig = sp.signal.filtfilt(b60,a60,sig) #filter out 60hz noise
        sig = sp.signal.filtfilt(b60,a60,sig) #filter out 60hz noise, again
        sig = sp.signal.filtfilt(b120,a120,sig) #filter out 120hz noise
        sig = sp.signal.filtfilt(b180,a180,sig) #filter out 180hz noise
        # snippets = sp.signal.decimate(snippets,q) #downsample down to 500Hz. This function applies anti-aliasing filters.
        
        return sig

    def _get_bandpower_area(self, freq_band: str, area: str):
        '''
        Power calculated by mean of sqaure of signal 
        (integrating square of signal over time window and dividing by number of time steps)

        Parameters
        ----------
        freq_band : str
            DESCRIPTION.
        area : str
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        print(f'Getting {freq_band} power for {area}..')
        def bandpass_filter(sig,f_lo,f_hi,fs):
            order = 4
            sos = sp.signal.butter(order,[f_lo,f_hi],'bandpass',output='sos',fs=fs)
            sig_out = sp.signal.sosfiltfilt(sos,sig)
            return sig_out
        
        match area:
            case 'vmPFC':
                chs = self.vmPFC_chs
            case 'Cd':
                chs = self.Cd_chs
            case 'OFC':
                chs = self.OFC_chs
            case _:
                raise ValueError('Invalid brain area')
        n_chs = len(chs)
        
        f_lo = FREQ_BANDS[freq_band][0]
        f_hi = FREQ_BANDS[freq_band][1]
        
        band_power_chs = np.zeros((n_chs,self.n_trials,N_TIMEBINS))
        for i,ch in enumerate(chs):
            for trial in range(self.n_trials):
                snip = self.snips[trial,ch,:]
                snip_filt = bandpass_filter(snip,f_lo,f_hi,self.fs)
                snip_sq = snip_filt*snip_filt
                
                for k in range(N_TIMEBINS):
                    win_start, win_end = k, k+1 # [timebin #]
                    band_power_chs[i,trial,k] = np.mean(snip_sq[win_start:win_end])
                    
        norm_band_power_chs = sp.stats.zscore(band_power_chs,axis=1,nan_policy='omit') #zscore over trials before avging over chs. This ensuresbig magnitude chs don't swamp small mag chs
        band_power_area = np.nanmean(norm_band_power_chs,axis=0) #avg over all chs of that area
        return band_power_area # shape = (trials x timepoints)
    
    
    def get_all_bandpowers(self,overwrite_flag=False):

        n_bands = len(FREQ_BANDS)
        n_areas = len(self.session_areas)
        all_bandpowers = dict()
        bandpowers = np.zeros((self.n_trials,n_bands,N_TIMEBINS))
        for i,area in enumerate(self.session_areas):
            for j,freq_band in enumerate(FREQ_BANDS):
                bandpowers[:,j,:] = self._get_bandpower_area(freq_band,area)
            
            all_bandpowers[area] = bandpowers
                
        return all_bandpowers # dict of arrays for each area of shape = (trials x freq_bands x timepoints)
    




class ProcessBehavior:
    '''
    From .hdf file, load task and behavior details and calculate value with
    either Q-Learning or empirical-based methods.
    
    Output: DataFrame with behavioral details for regression.
    '''
    
    def __init__(self, session: str, subsession: str):
        

        
        self.session = session
        self.subsession = subsession
        self.file_prefix = os.path.join(DATA_FOLDER, self.session, self.session)
        
        ## Data
        self.has_hdf = False # For behavior
        self.has_ns5 = False # For syncing 
        self.has_ns2 = False # For LFP
        self.has_nev = False # For waveform 
        self.has_decoder = False # For decoder
        self.has_mat = False # For syncing
        self.check_files()
        
        ## Get Behavior
        self.behavior_df = None
        self.get_behavior()
        # self.check_behavior()    dont call plots in init. call later if you wnat them
        # self.plot_choices()
        
        ## Save Out
        self.dict_out = {'Session':session, 'Subsession':subsession, 'df':self.behavior_df}
        
        

    def check_files(self):
        """
        For a proper analysis to be done,
        the hdf, nev, mat, and pkl files are essential.
        The ns5 and ns2 are optional.
        """
        
        if os.path.exists(self.file_prefix + '.hdf'):
            self.has_hdf = True
        if os.path.exists(self.file_prefix + '.ns5'):
            self.has_ns5 = True
        if os.path.exists(self.file_prefix + '.ns2'):
            self.has_ns2 = True
        if os.path.exists(self.file_prefix + '.nev'):
            self.has_nev = True
        if os.path.exists(self.file_prefix + '_syncHDF.mat'):
            self.has_mat = True
        if os.path.exists(self.file_prefix + '_nev_output.pkl'):
            self.has_nev_output = True
        if os.path.exists(self.file_prefix + '_KFDecoder.pkl'):
            self.has_decoder = True

     
    def get_behavior(self,Q_learning=True,learning_rates=None):
        '''
        Method which takes the hdf files for a session and computes key behavioral metrics for each 
        trial, such as value of each target and rpe. These along with information about each
        trial, such as whether it was free/forced and which target was chosen, are saved
        into a matrix which is ready to be plugged into a regression model to find what
        neurons are encoding throughout the course of a session.
        
        learning_rates = [pos_alpha,neg_alpha]

        Returns
        -------
        behavior_regressors: nested dict with entires being:
                'hc' and 'rp', each which is a dictionary containing:
                    'matrix' : T x num_behavior_vars array
                    'labels' : list of strs of length num_behavior_vars
            - The matrix array is the behavior for each time point T which the FR of each unit will be regressed by
              to see what units encode what behavior.
            - The labels list explains what each behavior is in the matrix.
              Labels for each behavior var must be in same order as they are put in behavior_regressor_matrix.
              Label for the nuisance constant should NOT be included. It will be added on in this method.

        '''
        
        ## NOTE: what was named "LV target" in older scripts is now named "Q1" since the values 
        ## change and switch over time. LV will now denoted the lower of the two targets
        ## at any given time.
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'        
        
        #load behavior over all hdf files at once for behavioral data section
        behavior = BehaviorAnalysis.ChoiceBehavior_Whitehall([self.hdf_file]) #method needs hdf filenames in a list
        
        # get choices, rewards, and trial type
        choices,rewards,instructed_or_freechoice = behavior.GetChoicesAndRewards()
        
        # get block info (stable or volatile)
        if behavior.is_stable_block is not None:
            is_stable_block = behavior.is_stable_block
            is_volatile_block = behavior.is_volatile_block
        else:
            is_stable_block, is_volatile_block = get_block_info(self.session)
            is_stable_block, is_volatile_block = is_stable_block[:len(choices)], is_volatile_block[:len(choices)] #trim to actual num of trials

        # Actual reward probability of each target throughout task
        rew_probs_2, rew_probs_1 = behavior.GetRewardProbability(is_stable_block)
        
        # Get which side (left or right) each choice was
        choice_side_raw = behavior.GetTargetSideSelection()
        choice_side = (choice_side_raw-0.5)*2 #make to be -1 or 1
        
        
        # Chop all variables to specified subsession. All later variables are calculated using these.
        
        if self.subsession == 'stable block':
            choices = choices[np.nonzero(is_stable_block)]
            rewards = rewards[np.nonzero(is_stable_block)]
            instructed_or_freechoice = instructed_or_freechoice[np.nonzero(is_stable_block)]
            rew_probs_1 = rew_probs_1[np.nonzero(is_stable_block)]
            rew_probs_2 = rew_probs_2[np.nonzero(is_stable_block)]
            choice_side = choice_side[np.nonzero(is_stable_block)]
            is_volatile_block = is_volatile_block[np.nonzero(is_stable_block)]
            is_stable_block = is_stable_block[np.nonzero(is_stable_block)]
            assert np.all(is_stable_block==1) and np.all(is_volatile_block==0)
        elif self.subsession == 'volatile block':
            choices = choices[np.nonzero(is_volatile_block)]
            rewards = rewards[np.nonzero(is_volatile_block)]
            instructed_or_freechoice = instructed_or_freechoice[np.nonzero(is_volatile_block)]
            rew_probs_1 = rew_probs_1[np.nonzero(is_volatile_block)]
            rew_probs_2 = rew_probs_2[np.nonzero(is_volatile_block)]
            choice_side = choice_side[np.nonzero(is_volatile_block)]
            is_stable_block = is_stable_block[np.nonzero(is_volatile_block)]
            is_volatile_block =  is_volatile_block[np.nonzero(is_volatile_block)]
            assert np.all(is_stable_block==0) and np.all(is_volatile_block==1)
            
        
        # Calculate value of targets throughout trials
        if Q_learning and learning_rates: #to do a specific learning rate (uses modified version of Rishi's method)
            beta=1 #beta doesn't matter since we are only interested in value here
            kappa=1 #same for kappa ^^
            Q0 = 0.5*np.ones(2)
            # instructed_or_freechoice = np.full_like(choices,2) #all whitehall trials are free choice
            if len(learning_rates)==2: #dual
                parameters = [learning_rates[0],learning_rates[1],beta,kappa]
                # Q1,Q2,_,_,_,_ = BehaviorAnalysis.Calc_DistrQlearning_2Targs_DualBaseline(learning_rates[0],learning_rates[1],beta,Q0,choices,rewards,instructed_or_freechoice)
                Q1,Q2,_,_,_,_ = Calc_DistrQlearning_2Targs_DualBaselineSide(parameters, Q0, choices, rewards, instructed_or_freechoice, choice_side_raw)
            elif len(learning_rates)==1: #single
                parameters = [learning_rates,beta,kappa]
                # Q1,Q2,_,_,_,_ = BehaviorAnalysis.Calculate_Qlearning(beta, learning_rates[0], Q0, choices,rewards,instructed_or_freechoice)
                Q1,Q2,_,_,_,_ = Calc_DistrQlearning_2Targs_BaselineSide(parameters, Q0, choices, rewards, instructed_or_freechoice, choice_side_raw)
        elif Q_learning and not learning_rates: #do Rishi's Q-learning method w/ specified model
            vmc = ValueModelingClass()
            value_dict = vmc.get_values(choices,rewards,instructed_or_freechoice,choice_side_raw, num_trials_A=0, num_trials_B=0, method=VALUE_MODEL)
            [Q1,Q2] = [value_dict['Q_low'],value_dict['Q_high']]
            self.Q_learning_params = value_dict['params']
        else: # do empirical value 
            [Q1,Q2], _ = behavior.CalcValue_2Targs(choices,rewards,win_sz=10,smooth=True)
        
        # split choices for each targ into own variable    
        Q1_choices = np.zeros_like(choices)
        Q1_choices[choices==1] = 1 #1 if he chose Q1, 0 otherwise
        Q2_choices = np.zeros_like(choices)
        Q2_choices[choices==2] = 1 #1 if he chose Q2, 0 otherwise
        
        # make time regressor 
        time = np.arange(len(choices))
        time = np.log(time+1)
        time -= np.mean(time) # to make centered at 0
        
        # Calculate value difference between targets throughout trials
        Qdiff = Q1 - Q2
        absQdiff = np.abs(Qdiff)
        
        # Calculate RPE: rpe(t) = r(t) - Q(t). Note: Q just means value. Doesn't have to use q-learning specifically.
        rpe=[]
        for trial,choice in enumerate(choices):
            if choice==1: #Q1 choice
                rpe.append(rewards[trial] - Q1[trial])
            elif choice==2: #Q2 choice
                rpe.append(rewards[trial] - Q2[trial])
        rpe = np.transpose(np.array(rpe))
        
        # #Split rpe into positive and negative rpe signals
        # pos_rpe = copy.deepcopy(rpe) #use deep copy so I can change one without changing the other
        # neg_rpe = copy.deepcopy(rpe)
        # pos_rpe[pos_rpe<0] = 0 #for positive rpe, set all negative rpes to zero
        # neg_rpe[neg_rpe>0] = 0 #for negative rpe, set all positive rpes to zero        
        
        # Split Q1 and Q2 into Qlow and Qhigh. Also do for choices
        Qlow = np.zeros_like(Q1)
        Qhigh = np.zeros_like(Q2)
        Qlow_choices = np.zeros_like(Q1)
        Qhigh_choices = np.zeros_like(Q2)
        for trial in range(len(Q1)):
            
            #Q1 = lower value -> Qlow=Q1 and Qhigh=Q2
            if Q1[trial] < Q2[trial]: 
                Qlow[trial] = Q1[trial]
                Qhigh[trial] = Q2[trial]
                
                #if Q1 chosen, then Qlow chosen
                if Q1_choices[trial] == 1: 
                    Qlow_choices[trial] = 1
                
                #if Q2 chosen, then Qhigh chosen
                else:                       
                    Qhigh_choices[trial] = 1
                    
            #Q1 = higher value -> Qlow=Q2 and Qhigh=Q1         
            else:
                Qlow[trial] = Q2[trial]
                Qhigh[trial] = Q1[trial]
                
                #if Q1 chosen, then Qhigh chosen
                if Q1_choices[trial] == 1: 
                    Qhigh_choices[trial] = 1
                
                #if Q2 chosen, then Qlow chosen
                else:                       
                    Qlow_choices[trial] = 1
            
        
        # Split Q1 and Q2 into Qchosen and Qunchosen    
        Q_chosen = np.zeros_like(Q1)
        Q_unchosen = np.zeros_like(Q2)
        for trial in range(len(Q1)):
            
            # If Q1 chosen
            if Q1_choices[trial]==1:
                Q_chosen[trial] = Q1[trial]
                Q_unchosen[trial] = Q2[trial]
                
            # if Q2 chosen
            else:
                Q_chosen[trial] = Q2[trial]
                Q_unchosen[trial] = Q1[trial]


        assert len(choices) == len(choice_side) == len(time) == len(Q1) == len(Q2) == len(rpe) == len(is_stable_block) == len(rew_probs_1)

        
        self.behavior_df = pd.DataFrame.from_dict(
            {'Q1':Q1,       'Q2':Q2,        'Choice1':Q1_choices,     'Choice2':Q2_choices, 
             'Qlow':Qlow,   'Qhigh':Qhigh,  'Qdiff_':Qdiff,  'absQdiff':absQdiff,
             'Qchosen':Q_chosen,            'Qunchosen':Q_unchosen,
             'Choice_low':Qlow_choices,  'Choice_high':Qhigh_choices,
             'Reward':rewards, 'RPE':rpe, 'Side':choice_side, 'Time':time,
             'Rew_prob_1':rew_probs_1, 'Rew_prob_2':rew_probs_2,
             'Stable':is_stable_block, 'Volatile':is_volatile_block}) #, 'Stim':stim_holds})
        
        self.num_trials = len(choices)
        
        if not Q_learning:
            print('Behavior loaded!')
            
        if self.behavior_df.isna().values.any():
            print('!'*20,f'\nNans found in behavior df for {self.session}',self.behavior_df)
            

        return
        
    
    def check_all_behavior(self):
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'        
        
        #load behavior over all hdf files at once for behavioral data section
        behavior = BehaviorAnalysis.ChoiceBehavior_Whitehall([self.hdf_file]) #method needs hdf filenames in a list
        
        # get rxn times
        behavior.GetRxnTime_ToggleSwitch()
        
        self.plot_values()
        
        self.plot_choices()
        
        return
     
    def plot_values(self,save_flag=False):
        
        fig,ax=plt.subplots(layout='constrained')
        
        #Q1 vs Q2
        ax.plot(self.behavior_df['Q1'],color='red',label='Q1')
        ax.plot(self.behavior_df['Q2'],color='green',label='Q2')
        ax.legend()
        ax.set_ylabel('Value')
        ax.set_xlabel('Trials')
        fig.suptitle(self.session)
        
        
        # plot half of the trials on each axis
        half = len(self.behavior_df['Q1']) //2
        
        fig,ax = plt.subplots(layout='constrained')
        # ax=axs[0]
        ax.plot(self.behavior_df['Q1'][:half],color='red',label='Q1')
        ax.plot(self.behavior_df['Q2'][:half],color='green',label='Q2')
        ax.legend()
        ax.set_ylabel('Value')
        # ax.set_xlabel('Trials')
        
        fig,ax = plt.subplots(layout='constrained')
        x=np.arange(half,len(self.behavior_df['Q1']))
        ax.plot(x,self.behavior_df['Q1'][half:],color='red',label='Q1')
        ax.plot(x,self.behavior_df['Q2'][half:],color='green',label='Q2')
        # ax.legend()
        ax.set_ylabel('Value')
        ax.set_xlabel('Trials')
      
        fig.suptitle(self.session)
        
        return fig   
    
    def plot_values_vs_choices_vs_rewprob(self,num_targs=1,num_trials_to_plot=200,save_flag=False):
        
        fig,ax=plt.subplots(layout='constrained')
        
        #actual rew prob
        ax.plot(self.behavior_df['Rew_prob_1'].iloc[:num_trials_to_plot],color='k',linestyle='--',label='Reward Probability 1')
        if num_targs==2:
            ax.plot(self.behavior_df['Rew_prob_2'].iloc[:num_trials_to_plot],color='gray',linestyle='--',label='Reward Probability 2')
        
        #smoothed choice prob
        window_length=10
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)
        ax.plot(P_LV[:num_trials_to_plot],color='red',label='Choice 1')
        if num_targs==2:
            ax.plot(P_HV[:num_trials_to_plot],color='green',label='Choice 2')
            
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[:num_trials_to_plot],color='orange',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[:num_trials_to_plot],color='turquoise',label='Q2')
            
        #plot labels
        ax.set_title('Modeled Values vs Actual Choices vs Actual Rew Prob')
        ax.legend()
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsChoicesVsRewProb',PROJ_FOLDER)
        
        return fig
    
    
    def plot_values_vs_choices(self,num_targs=1,num_trials_to_plot=200,save_flag=False):
        
        fig,ax=plt.subplots(layout='constrained')
        
        #smoothed choice prob
        window_length=10
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)
        ax.plot(P_LV[:num_trials_to_plot],color='red',label='Choice 1')
        if num_targs==2:
            ax.plot(P_HV[:num_trials_to_plot],color='green',label='Choice 2')
            
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[:num_trials_to_plot],color='orange',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[:num_trials_to_plot],color='turquoise',label='Q2')
            
        #plot labels
        ax.set_title('Modeled Values vs Actual Choices')
        ax.legend()
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsChoices',PROJ_FOLDER)
        
        return fig
    
    
    def plot_values_vs_rewards(self,num_targs=2,num_trials_to_plot=200,save_flag=False):
        
        fig,ax=plt.subplots(layout='constrained')
            
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[:num_trials_to_plot],color='orange',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[:num_trials_to_plot],color='turquoise',label='Q2')
            
        #Rewards
        for trial in range(num_trials_to_plot):
            if self.behavior_df['Choice1'].iloc[trial] == 1:
                if self.behavior_df['Reward'].iloc[trial] == 1:
                    ax.plot(trial,self.behavior_df['Q1'].iloc[trial],color='orange',marker='^') #rewarded choice 1s
                else:
                    ax.plot(trial,self.behavior_df['Q1'].iloc[trial],color='orange',marker='s') #unrewarded choice 1s
            
            if num_targs==2:
                if self.behavior_df['Choice2'].iloc[trial] == 1:
                    if self.behavior_df['Reward'].iloc[trial] == 1:
                        ax.plot(trial,self.behavior_df['Q2'].iloc[trial],color='turquoise',marker='^') #rewarded choice 2s
                    else:
                        ax.plot(trial,self.behavior_df['Q2'].iloc[trial],color='turquoise',marker='s') #unrewarded choice 2s
        
            
        #plot labels
        ax.set_title('Modeled Values vs Rewards')
        triangles = mlines.Line2D([], [], color='k', marker='^',label='Rewarded')
        squares = mlines.Line2D([], [], color='k', marker='s',label='Unrewarded')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(triangles)
        handles.append(squares)
        ax.legend(handles=handles)
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsRews',PROJ_FOLDER)
        
        return fig
    
    
    def plot_values_vs_rewprob(self,num_targs=2,num_trials_to_plot=200,save_flag=False):

        fig,ax=plt.subplots(layout='constrained')
        
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[:num_trials_to_plot],color='red',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[:num_trials_to_plot],color='green',label='Q2')
        
        #actual rew prob
        ax.plot(self.behavior_df['Rew_prob_1'].iloc[:num_trials_to_plot],color='red',linestyle='--',label='Reward Probability 1')
        if num_targs==2:
            ax.plot(self.behavior_df['Rew_prob_2'].iloc[:num_trials_to_plot],color='green',linestyle='--',label='Reward Probability 2')
        
        #plot labels
        ax.set_title('Modeled Values vs Actual Reward Probability')
        ax.legend()
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')   
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsRewProb',PROJ_FOLDER)
        
        return fig
    
    def plot_value_dist(self,save_flag=False):
        
        def get_values_rewprob_volatility(df,stable_or_volatile,rew_prob):
            Q1 = df['Q1'].loc[(df[stable_or_volatile]==1) & (df['Rew_prob_1']==rew_prob)]
            Q2 = df['Q2'].loc[(df[stable_or_volatile]==1) & (df['Rew_prob_2']==rew_prob)]
            
            return np.concatenate([Q1,Q2])
        
        ## Histogram plots
        
        bins=np.linspace(0.0,1.0,num=20)
        
        # rewprob = 0/1 plot
        fig,ax=plt.subplots(layout='constrained')
        fig.suptitle(self.session)
        ax.set_title('Easy Choices')
        for i,stable_or_volatile in enumerate(['Stable','Volatile']):
            for ii,rew_prob in enumerate([0.,1.]):
                values = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
                ax.hist(values,density=True,bins=bins,histtype='step',
                        label=stable_or_volatile,linestyle=LINESTYLES[i],color=COLORS[ii])
                
        # rewprob = 0.2/0.8 plot
        fig,ax=plt.subplots(layout='constrained')
        fig.suptitle(self.session)
        ax.set_title('Medium Choices')
        for i,stable_or_volatile in enumerate(['Stable','Volatile']):
            for ii,rew_prob in enumerate([0.2,0.8]):
                values = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
                ax.hist(values,density=True,bins=bins,histtype='step',
                        label=stable_or_volatile,linestyle=LINESTYLES[i],color=COLORS[ii])
                
                
        # rewprob = 0.4/0.6 plot
        fig,ax=plt.subplots(layout='constrained')
        fig.suptitle(self.session)
        ax.set_title('Hard Choices')
        for i,stable_or_volatile in enumerate(['Stable','Volatile']):
            for ii,rew_prob in enumerate([0.4,0.6]):
                values = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
                ax.hist(values,density=True,bins=bins,histtype='step',
                        label=stable_or_volatile,linestyle=LINESTYLES[i],color=COLORS[ii])
        
            
        # ## Variance Bar plots
        
        # # rewprob = 0/1 plot
        # fig,ax=plt.subplots(layout='constrained')
        # for i,stable_or_volatile in enumerate(['Stable','Volatile']):
        #     rew_prob = 0.
        #     values1 = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
        #     rew_prob = 1.
        #     values2 = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
        #     values = np.concatenate([values1,values2])
        #     ax.bar(i,np.var(values),label=stable_or_volatile,hatch=HATCHSTYLES[i],color='gray')
                
        # # rewprob = 0.2/0.8 plot
        # fig,ax=plt.subplots(layout='constrained')
        # for i,stable_or_volatile in enumerate(['Stable','Volatile']):
        #     rew_prob = 0.2
        #     values1 = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
        #     rew_prob = 0.8
        #     values2 = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
        #     values = np.concatenate([values1,values2])
        #     ax.bar(i,np.var(values),label=stable_or_volatile,hatch=HATCHSTYLES[i],color='gray')
                
                
        # # rewprob = 0.4/0.6 plot
        # fig,ax=plt.subplots(layout='constrained')
        # for i,stable_or_volatile in enumerate(['Stable','Volatile']):
        #     rew_prob = 0.4
        #     values1 = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
        #     rew_prob = 0.6
        #     values2 = get_values_rewprob_volatility(self.behavior_df,stable_or_volatile,rew_prob)
        #     values = np.concatenate([values1,values2])
        #     ax.bar(i,np.var(values),label=stable_or_volatile,hatch=HATCHSTYLES[i],color='gray')

        
        
        

        
        
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueDistribution',PROJ_FOLDER)
        
        return #kurt_vol,kurt_stab
    
    
    def plot_choices_and_rewards(self,save_flag=False):
            
        color1 = 'gold'
        color2 = 'blue'
        
        ## choice behavior and rewards figure for Fig 1
        #smooth choice data 
        window_length=50
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)

        fig,ax = plt.subplots(layout='constrained')

        ax.plot(P_LV,color=color1,label='Target 1')
        ax.plot(P_HV,color=color2,label='Target 2')

        #Get reward data for each choice
        LV_rew = np.nonzero((self.behavior_df['Choice1']==1) & (self.behavior_df['Reward']==1))
        LV_unrew = np.nonzero((self.behavior_df['Choice1']==1) & (self.behavior_df['Reward']==0))
        HV_rew = np.nonzero((self.behavior_df['Choice1']!=1) & (self.behavior_df['Reward']==1))
        HV_unrew = np.nonzero((self.behavior_df['Choice1']!=1) & (self.behavior_df['Reward']==0))

        ymin_hi, ymax_hi = 1.1, 1.2 #for rewarded trials
        ymin_lo, ymax_lo = -0.2, -0.1 #for unrewarded trials

        ax.vlines(LV_rew,ymin_hi,ymax_hi,color=color1)
        ax.vlines(LV_unrew,ymin_lo,ymax_lo,color=color1)
        ax.vlines(HV_rew,ymin_hi,ymax_hi,color=color2)
        ax.vlines(HV_unrew,ymin_lo,ymax_lo,color=color2)
        
        tick_marks = [np.mean([ymin_lo,ymax_lo]),0,0.25,0.5,0.75,1,np.mean([ymin_hi,ymax_hi])]
        tick_labels = ['Unrewarded',0,0.25,0.5,0.75,1,'Rewarded']
        ax.set_yticks(tick_marks,tick_labels)
        
        ax.set_ylabel('Choice Probability')
        ax.set_xlabel('Trials')
        
        # ax.set_xlim([0,1000])

        ax.legend()
        ax.set_title(self.session)
        
        
        if save_flag:
            utils.save_svg(fig,self.session,'ChoiceAndRewardBehavior',PROJ_FOLDER)
            
        return fig    
    
    def plot_choices(self,save_flag=False):
        
        color1 = 'gold'
        color2 = 'blue'
        
        ## choice behavior and rewards figure for Fig 1
        #smooth choice data 
        window_length=50
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)

        fig,ax = plt.subplots(layout='constrained')

        ax.plot(P_LV,color=color1,label='Target 1')
        ax.plot(P_HV,color=color2,label='Target 2')

        tick_marks = [0,0.25,0.5,0.75,1]
        tick_labels = [0,0.25,0.5,0.75,1]
        ax.set_yticks(tick_marks,tick_labels)
        
        ax.set_ylabel('Choice Probability')
        ax.set_xlabel('Trials')
        
        
        def blocks_volatile_first():
            vol_block_edges = np.arange(0,24*20,20)
            stab_block_edges = np.arange(vol_block_edges[-1],vol_block_edges[-1]+6*100,100)
            edges = np.concat((vol_block_edges,stab_block_edges))
            return edges
        
        edges = blocks_volatile_first()
        # ax.vlines(edges,0,1)
        every_other_edge = edges[::2]
        y=[0,1]
        for edge_num in range(len(edges)//2-1):
            ax.fill_betweenx(y,edges[edge_num*2],edges[edge_num*2+1],color='k',alpha=0.1)
        
        # ax.set_xlim([0,1000])

        ax.legend()
        ax.set_title(self.session)
        
        
        if save_flag:
            utils.save_svg(fig,self.session,'ChoiceBehavior',PROJ_FOLDER)
            # fname=f'{self.file_prefix}_choice_behav.svg'
            # plt.savefig(fname,format='svg')
            # print(f'{fname} saved!')
            
        
        
        
        return fig
    
    
    def get_behavior_var_corr(self,list_of_vars):
        
        num_vars = len(list_of_vars)
        ind_vars = np.zeros((num_vars,self.num_trials))
        corr_matrix = np.zeros((num_vars,num_vars))
        
        #gather behavioral vars from behavior df
        for i,var in enumerate(list_of_vars): 
            ind_vars[i,:] = self.behavior_df[var].to_numpy()
            
        # calculate pairwise corr between all combos of vairables
        for i in range(num_vars):
            for j in range(num_vars):
                res = sp.stats.pearsonr(ind_vars[i,:],ind_vars[j,:])
                corr_matrix[i,j] = res.statistic
                
        corr_matrix = np.multiply(corr_matrix,corr_matrix) #make it r squared
        
        return corr_matrix
      

class LearningRate_Analysis:
    
    def __init__(self,overwrite_flag=False,do_stable_first_only=False,do_volatile_first_only=False,do_even_num_trials=False):
        
        self.fig_folder = 'LearningRates'
        
        self.do_even_num_trials=do_even_num_trials
       
        self.lr_dfs = []
        self.nt_dfs = []
        for s,session in enumerate(SESSIONS):
            
            start = time.time()
            
            #to test stab->vol vs vol->stab
            is_stable_block, is_volatile_block = get_block_info(session)
            if do_volatile_first_only:
                if is_stable_block[0]==1 and is_volatile_block[0]==0:
                    continue #skip stab first
            if do_stable_first_only:
                if is_stable_block[0]==0 and is_volatile_block[0]==1:
                    continue #skip vol first
            
            
            #Get encodings
            try:
                neuron_type_df = utils.load_csv('NeuronTypes',session,DATA_FOLDER)
            except:
                print('Need to run neuron typing first!')
                return
            
            # Get learning rate for each unit
            if not overwrite_flag and utils.does_csv_exist('LearningRates',session,DATA_FOLDER): #don't overwrite, load
                learning_rate_df = utils.load_csv('LearningRates',session,DATA_FOLDER)
            else: #don't load, compute
                Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
                learning_rate_df = self.get_learning_rates(Spikes.fr_df,Behav,neuron_type_df)
                utils.save_csv(learning_rate_df,'LearningRates',session,DATA_FOLDER)
            self.lr_dfs.append(learning_rate_df)
            self.nt_dfs.append(neuron_type_df)
            
            utils.get_time_left(start,s,len(SESSIONS))
            
            
             
        self.allsess_lr_df = utils.merge_sessions_df(self.lr_dfs)
        self.allsess_nt_df = utils.merge_sessions_df(self.nt_dfs)
        utils.save_csv(self.allsess_lr_df,'LearningRates','',DATA_FOLDER)
        utils.save_csv(self.allsess_nt_df,'NeuronTypes','',DATA_FOLDER)
        
        if do_stable_first_only:
            suptitle = 'Stable to Volatile Sessions'
        elif do_volatile_first_only:
            suptitle = 'Volatile to Stable Sessions'
        elif do_even_num_trials:
            suptitle = 'All Sessions, even num_trials for stable and volatile'
        else:
            suptitle = 'All Sessions'
            
        # Plot
        self.plot_learning_rate_dual_violins(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        self.plot_learning_rate_dual_hists(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        self.plot_winbias_violins(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        self.plot_learning_rate_scatter_1(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        self.plot_learning_rate_scatter_2(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        self.plot_learning_rate_scatter_all_combined(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        self.plot_winbias_hist(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        
        return 
        

        
    def get_learning_rates(self,fr_df,Behavior,neuron_type_df):
    
        df_rows_learning_rates = []
        print(f'{len(fr_df.columns)} neurons')
        
        for stable_or_volatile in SUBSESSIONS:        
        
            assert len(Behavior.behavior_df) == len(fr_df), 'Number of trials do not match!'
            # num_trials = len(fr_df)
            
            if self.do_even_num_trials:
                trials = utils.get_trials_even_num(Behavior.behavior_df,stable_or_volatile)
            else:
                trials = utils.get_trials(Behavior.behavior_df,stable_or_volatile)
        
            if not len(trials) > 1: #if no trials for current subsession
                print(f'No trials for {stable_or_volatile}. Skipping!')    
                continue
            else:
                print(stable_or_volatile)
        
            # Get FR for each neuron
            units = fr_df.columns
            # for i,unit in enumerate(units):
            #     neuron_type = utils.get_unit_data(neuron_type_df,f'{unit} - {stable_or_volatile}','Neuron Type')
            #     print(neuron_type)
            #     print('Q' in neuron_type)
            # xxx
            for i,unit in enumerate(units):
                
                start = time.time()
                
                neuron_type = str(utils.get_unit_data(neuron_type_df, f'{unit} - all trials', 'Neuron Type'))
                if "Q" not in neuron_type:
                    continue
                
                fr = fr_df[unit].iloc[trials] #vector of FRs for subsession trial
                
                ## Get regression weights for all learning rate combos
                #loop thru all possible learning rates
                df_rows_regression_weights = []
                for pos_alpha in LEARNING_RATES:
                    for neg_alpha in LEARNING_RATES:
                    
                        # get beahvior according to current learning rates
                        Behavior.get_behavior(Q_learning=True, learning_rates=[pos_alpha,neg_alpha])
                        regressor_matrix = Behavior.behavior_df[REGRESSORS].iloc[trials]
                    
                        # regress fr against this learning rate behavior
                        res = utils.linear_encoding_regression(fr, regressor_matrix)
                    
                        row_dict_regression_weights = {'pos_alpha':pos_alpha, 'neg_alpha':neg_alpha} #each row of the df will be the lrs for each reg
                        # Get weights for each regressor
                        regressor_names = res.params.keys() #includes const in idx 0
                        
                        #loop thru all regressors
                        for j in range(len(regressor_names)):
                            if res.pvalues[regressor_names[j]] < ALPHA_THRESHOLD and res.f_pvalue < ALPHA_THRESHOLD: #if signif
                                row_dict_regression_weights[regressor_names[j]] = abs(res.params.values[j]) #fill in the row of the df with param
                            else:
                                row_dict_regression_weights[regressor_names[j]] = np.nan #if not signif, put in nan so it will get skipped
                        
                        df_rows_regression_weights.append(row_dict_regression_weights)
                
                # Organize regression weights for each learning rate into a df  
                regression_weights_df = pd.DataFrame(df_rows_regression_weights)
                # print(regression_weights_df)
                # xxx
                # utils.save_csv(regression_weights_df,'RegressionWeights',"debug",DATA_FOLDER)
                
                # Find learning rate combination which maximizes each regression weight
                unit_subsession_labeled = f'{unit} - {stable_or_volatile}'
                row_dict_learning_rates = {'Unit':unit_subsession_labeled}
                for j in range(len(REGRESSORS)):
                    
                    #only do if this neuron was found to encode that regressor
                    if REGRESSORS[j] in neuron_type:
                    
                        if not np.all(np.isnan(regression_weights_df[REGRESSORS[j]])): #to catch if all regression weights are nan
                            # idx of max regression weight for current regressor
                            idx_max_regressor = regression_weights_df[REGRESSORS[j]].idxmax()
                            
                            # Find and store learning rate which corresponds to max regression weight
                            row_dict_learning_rates['Pos Learning Rate for ' + REGRESSORS[j]] = regression_weights_df['pos_alpha'].iloc[idx_max_regressor]
                            row_dict_learning_rates['Neg Learning Rate for ' + REGRESSORS[j]] = regression_weights_df['neg_alpha'].iloc[idx_max_regressor]
                    
                    else: #if all weights are nan
                        # Put in np.nan since its more consistent when saving and loading dfs
                        row_dict_learning_rates['Pos Learning Rate for ' + REGRESSORS[j]] = np.nan
                        row_dict_learning_rates['Neg Learning Rate for ' + REGRESSORS[j]] = np.nan
                    
                df_rows_learning_rates.append(row_dict_learning_rates)
                
                # print(f'Unit {i+1}/{len(units)} for {stable_or_volatile}')
                utils.get_time_left(start,i,len(units))
                
                
        learning_rate_df = pd.DataFrame(df_rows_learning_rates) #rows: units x subsessions, cols: lrs for each reg
        # utils.save_csv(learning_rate_df,'LR',"debug",DATA_FOLDER)
        # xxx
        
        #filter so that learning rates are only displayed for regressors that are encoded by that unit
        # learning_rate_df = filter_learning_rates(learning_rate_df,neuron_type_df) not needed. filtering done by seeing if best regression passes alpha thresh
    
        return learning_rate_df

    def _strip_sign_prefix(self,col):
        for prefix in ('Pos ', 'Neg '):
            if col.startswith(prefix):
                return col[len(prefix):]
        return col

    def _collect_pairs(self,lr_col,unique_units,area_learning_rate_df):
        """Return (vol_lrs, stab_lrs) for units encoding lr_col across volatile/stable subsessions."""
        vols, stabs = [], []
        for unit in unique_units:
            # if col_noprefix in utils.get_unit_data( #GETTING RID OF NEURON TYPE FILTERING 5-13
            #     area_neuron_type_df, f'{unit} - {SUBSESSIONS[0]}', 'Neuron Type'
            # ):
            volatile = utils.get_unit_data(
                area_learning_rate_df, f'{unit} - {SUBSESSIONS[2]}', lr_col
            )
            stable = utils.get_unit_data(
                area_learning_rate_df, f'{unit} - {SUBSESSIONS[1]}', lr_col
            )
            if not np.isnan(volatile) and not np.isnan(stable):  # ensures both exist
                vols.append(volatile)
                stabs.append(stable)
        assert len(vols) == len(stabs)
        return vols, stabs
    
    def plot_learning_rate_dual_violins(self,learning_rate_df, neuron_type_df, suptitle, fig_folder):

        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']

        # Pair up positive and negative learning rate columns that share the same neuron type suffix.
        # Expected column naming: 'Positive Learning rate for Q...' / 'Negative Learning rate for Q...'
        pos_cols = {col: col for col in cols if 'Q' in col and col.startswith('Pos')}
        neg_cols = {col: col for col in cols if 'Q' in col and col.startswith('Neg')}

        # Match pairs by stripping the sign prefix so we can iterate together.
        # e.g. 'Positive Learning rate for Q_value' -> 'Learning rate for Q_value'
        

        # paired_suffixes = sorted(
        #     set(strip_sign_prefix(c) for c in pos_cols) &
        #     set(strip_sign_prefix(c) for c in neg_cols)
        # )
        # Also handle any unpaired columns (only pos or only neg) gracefully.
        all_suffixes = sorted(
            set(self._strip_sign_prefix(c) for c in list(pos_cols) + list(neg_cols))
        )

        for i, area in enumerate(areas):

            area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)

            if area_learning_rate_df.empty:  # if no units for this area, skip
                continue

            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)

            for suffix in all_suffixes:
                pos_col = f'Pos {suffix}'
                neg_col = f'Neg {suffix}'
                has_pos = pos_col in cols
                has_neg = neg_col in cols

                col_noprefix = suffix.removeprefix('Learning Rate for ')

                # Collect per-subsession distributions for positive and negative LRs.
                def collect_rates(lr_col):
                    """Return (rates_list, labels) across subsessions for lr_col."""
                    rates_list = []
                    labels = []
                    for subsession in SUBSESSIONS:
                        learning_rates = []
                        for unit in unique_units:
                            unit_subsession_labeled = f'{unit} - {subsession}'

                            # # Only include units that encode this neuron type during this subsession. GETTING RID OF NEUORN TYPE FILTERING 5-13
                            # if col_noprefix in utils.get_unit_data(
                            #     area_neuron_type_df, unit_subsession_labeled, 'Neuron Type'
                            # ):
                            lr = utils.get_unit_data(area_learning_rate_df, unit_subsession_labeled, lr_col)
                            if not np.isnan(lr):
                                learning_rates.append(lr)

                        if len(learning_rates) > 0:
                            rates_list.append(np.array(learning_rates).flatten())
                            labels.append(subsession + f'\nnum_units: {len(learning_rates)}')

                    return rates_list, labels

                pos_rates, pos_labels = collect_rates(pos_col) if has_pos else ([], [])
                neg_rates, neg_labels = collect_rates(neg_col) if has_neg else ([], [])

                # Skip entirely if there is nothing to plot.
                if not pos_rates and not neg_rates:
                    print(f'{area} {col_noprefix}: no data, skipped!')
                    continue

                # Determine subplot layout: two columns when both signs are present, one otherwise.
                n_plots = (1 if has_pos and pos_rates else 0) + (1 if has_neg and neg_rates else 0)
                fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), sharey=True)
                if n_plots == 1:
                    axes = [axes]

                ax_idx = 0

                def _draw_violin(ax, rates_list, labels, sign_label):
                    violin = ax.violinplot(rates_list)
                    for pc in violin['bodies']:
                        pc.set_facecolor(COLORS[i])
                        pc.set_edgecolor(COLORS[i])
                    ax.set_xticks(np.arange(1, len(rates_list) + 1), labels)
                    ax.set_ylabel(f'{sign_label} learning rate')
                    ax.set_ylim([-0.1, 1.1])
                    ax.set_title(f'{sign_label} LR\n{area} — {col_noprefix}')
                    # print(suffix, area)
                    # print(labels)
                    # print(rates_list)

                if has_pos and pos_rates:
                    _draw_violin(axes[ax_idx], pos_rates, pos_labels, 'Positive')
                    ax_idx += 1

                if has_neg and neg_rates:
                    _draw_violin(axes[ax_idx], neg_rates, neg_labels, 'Negative')

                fig.suptitle(suptitle)
                
                plt.show()

                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_pos_neg_violin',
                    fig_folder,
                    PROJ_FOLDER,
                )

        return


    def plot_winbias_violins(self,learning_rate_df, neuron_type_df, suptitle, fig_folder):

        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']

        # Pair up positive and negative learning rate columns that share the same neuron type suffix.
        # Expected column naming: 'Positive Learning rate for Q...' / 'Negative Learning rate for Q...'
        pos_cols = {col: col for col in cols if 'Q' in col and col.startswith('Pos')}
        neg_cols = {col: col for col in cols if 'Q' in col and col.startswith('Neg')}

        # Match pairs by stripping the sign prefix so we can iterate together.
        # e.g. 'Positive Learning rate for Q_value' -> 'Learning rate for Q_value'
        

        # paired_suffixes = sorted(
        #     set(strip_sign_prefix(c) for c in pos_cols) &
        #     set(strip_sign_prefix(c) for c in neg_cols)
        # )
        # Also handle any unpaired columns (only pos or only neg) gracefully.
        all_suffixes = sorted(
            set(self._strip_sign_prefix(c) for c in list(pos_cols) + list(neg_cols))
        )

        for i, area in enumerate(areas):

            area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)

            if area_learning_rate_df.empty:  # if no units for this area, skip
                continue

            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)

            for suffix in all_suffixes:
                pos_col = f'Pos {suffix}'
                neg_col = f'Neg {suffix}'
                has_pos = pos_col in cols
                has_neg = neg_col in cols

                col_noprefix = suffix.removeprefix('Learning Rate for ')

                pos_vols, pos_stabs = self._collect_pairs(pos_col,unique_units,area_learning_rate_df) if pos_col else ([], [])
                neg_vols, neg_stabs = self._collect_pairs(neg_col,unique_units,area_learning_rate_df) if neg_col else ([], [])

                # # Skip entirely if there is nothing to plot.
                # if not pos_rates and not neg_rates:
                #     print(f'{area} {col_noprefix}: no data, skipped!')
                #     continues
                
                # compute winbias
                winbias_vols = np.array(pos_vols) - np.array(neg_vols)
                label_vols = f'volatile block\nnum_units: {len(winbias_vols)}'
                winbias_stabs = np.array(pos_stabs) - np.array(neg_stabs)
                label_stabs = f'stable block\nnum_units: {len(winbias_stabs)}'
                winbias_delta = winbias_vols - winbias_stabs
                label_delta = f'volatile - stable\nnum_units: {len(winbias_delta)}'

                fig, axs = plt.subplots(1,2,layout='constrained')

                def _draw_violin(ax, rates_list, labels_list, colors_list, axis_label, ylims):
                    x = np.arange(1, len(rates_list) + 1)
                    violin = ax.violinplot(rates_list)
                    for i,pc in enumerate(violin['bodies']):
                        pc.set_facecolor(colors_list[i])
                        pc.set_edgecolor(colors_list[i])
                        ax.plot(x[i]-0.015,np.mean(rates_list[i]),color=COLORS[i],marker='>')
                    ax.hlines(0,min(x)-0.25,max(x)+0.25,color='k',linestyle='dashed')
                    ax.set_xticks(x, labels_list)
                    ax.set_ylabel(f'{axis_label}')
                    ax.set_ylim(ylims)
                    ax.set_title(f'{axis_label}\n{area} — {col_noprefix}')


                _draw_violin(axs[0], [winbias_stabs,winbias_vols], [label_stabs,label_vols], [COLORS[0],COLORS[1]],'Win-Bias',[-1.1, 1.1])
                _draw_violin(axs[1], [winbias_delta], [label_delta], [COLORS[2]], 'Delta Win-Bias',[-2.2, 2.2])

                fig.suptitle(suptitle)
                
                plt.show()

                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_winbias_violin',
                    fig_folder,
                    PROJ_FOLDER,
                )

        return


    def plot_learning_rate_changes_hist(self,learning_rate_df,neuron_type_df,suptitle,fig_folder):
        
        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']
        
        for i,area in enumerate(areas):
            
            area_neuron_type_df = utils.area_parser(neuron_type_df,area)
            area_learning_rate_df = utils.area_parser(learning_rate_df,area)
            
            if area_learning_rate_df.empty: # if no units for this area, skip
                continue
            
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"],SUBSESSIONS)
        
            for col in cols:
                if 'Q' in col: #only do for value related stuff
                    
                    col_noprefix = col.removeprefix('Learning rate for ')
                
                    # Get change in learning rate for all units that encode the current col
                    deltas = []
                    for unit in unique_units:
                        
                        # if this unit encodes the current column over all trials
                        if col_noprefix in utils.get_unit_data(area_neuron_type_df,f'{unit} - {SUBSESSIONS[0]}','Neuron Type'):
            
                            volatile = utils.get_unit_data(area_learning_rate_df,f'{unit} - {SUBSESSIONS[2]}',col)
                            stable = utils.get_unit_data(area_learning_rate_df,f'{unit} - {SUBSESSIONS[1]}',col)
                            
                            if not np.isnan(volatile) and not np.isnan(stable): #if they exist
                                deltas.append(stable - volatile)
                                
                    num_units = len(deltas)
                    
                    
                    if num_units > 0:
                        fig,ax=plt.subplots(layout='constrained')
                        ax.hist(deltas,bins=20,range=(-1,1),color=COLORS[i])
                        ax.set_xlabel(f'Delta {col}\n(Stable minus Volatile)')
                        ax.set_ylabel('Number of neurons')
                        ax.set_xlim([-1.1,1.1])
                        [ylo,yhi] = ax.get_ylim()
                        ax.vlines(0,ylo,yhi,color='k',linestyle='--',label='No change')
                        ax.legend()
                        ax.set_title(f'Total number of {col_noprefix} {area} units: {num_units}')
                        # ax.set_title(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
                        fig.suptitle(suptitle)
                        
                        
                        utils.save_svg(fig,f'{suptitle}_{area}_{col_noprefix}_changes_hist',fig_folder,PROJ_FOLDER)    

        return



    def plot_learning_rate_paired_changes(self,learning_rate_df,neuron_type_df,suptitle,fig_folder):
        
        cols = learning_rate_df.columns
        
        areas = AREAS + ['all areas']
        
        for i,area in enumerate(areas):
            
            area_neuron_type_df = utils.area_parser(neuron_type_df,area)
            area_learning_rate_df = utils.area_parser(learning_rate_df,area)
            
            print(area)
            if area_learning_rate_df.empty: # if no units for this area, skip
                print('skipped!')
                continue
            
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"],SUBSESSIONS)
            
            for col in cols:
                if 'Q' in col: #only do for value related stuff
                    
                    col_noprefix = col.removeprefix('Learning rate for ')
                
                    # Get change in learning rate for all units that encode the current col
                    vols = []
                    stabs = []
                    for unit in unique_units:
                        
                        # if this unit encodes the current column over all trials
                        if col_noprefix in utils.get_unit_data(area_neuron_type_df,f'{unit} - {SUBSESSIONS[0]}','Neuron Type'):
            
                            volatile = utils.get_unit_data(area_learning_rate_df,f'{unit} - {SUBSESSIONS[2]}',col)
                            stable = utils.get_unit_data(area_learning_rate_df,f'{unit} - {SUBSESSIONS[1]}',col)
                            
                            if not np.isnan(volatile) and not np.isnan(stable): #if they exist
                                vols.append(volatile)
                                stabs.append(stable)
                                
                    assert len(vols) == len(stabs)            
                    num_units = len(vols)
                    
                    if num_units > 0:
                        fig,ax=plt.subplots(layout='constrained')
                        
                        x_vol, x_stab = 1, 2
                        color_increase, color_decrease = 'gray', COLORS[i]
                        
                        for vol,stab in zip(vols,stabs):
                            color = color_increase if stab >= vol else color_decrease
                            
                            #individal points
                            ax.plot([x_vol, x_stab], [vol, stab], color=color) #connecting lines
                            ax.scatter([x_vol, x_stab], [vol, stab], color=color) #individual dots
                            
                        #means
                        ax.plot([x_vol, x_stab], [np.mean(vols), np.mean(stabs)], color='k',linewidth=2) #connecting lines
                        ax.scatter([x_vol, x_stab], [np.mean(vols), np.mean(stabs)], color='k',linewidth=2,marker="D") #individual dots 
                            
                            
                        ax.set_xlim(0.8, 2.2)
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_xticks([x_vol, x_stab])
                        ax.set_xticklabels(["Volatile", "Stable"], fontsize=12, fontweight="bold")
                        ax.set_ylabel("Learning Rate", fontsize=12)
                        
                        legend_handles = [
                            mpatches.Patch(color=color_increase, label="Increased"),
                            mpatches.Patch(color=color_decrease, label="Decreased"),
                            plt.Line2D([0], [0], color='k', marker="D", markersize=7,
                                       label="Mean", linewidth=2),
                        ]
                        ax.legend(handles=legend_handles, fontsize=8.5, frameon=False,
                                  loc="upper left", labelcolor="#444")
                        ax.set_title(f'Total number of {area} {col_noprefix} units: {num_units}')
                        # ax.set_title(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
                        fig.suptitle(suptitle)
                        
                        
                        utils.save_svg(fig,f'{suptitle}_{area}_{col_noprefix}_paired_changes',fig_folder,PROJ_FOLDER)    

        return

    
    def plot_learning_rate_paired_changes_dual(self, learning_rate_df, neuron_type_df, suptitle, fig_folder):
    
        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']
    
        # Pair positive and negative LR columns by their shared neuron-type suffix.
        # Expected naming: 'Positive Learning rate for Q...' / 'Negative Learning rate for Q...'    
        pos_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Pos')}
        neg_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Neg')}
        all_suffixes = sorted(set(pos_cols) | set(neg_cols))
    
        for i, area in enumerate(areas):
    
            # area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)
    
            for suffix in all_suffixes:
                pos_col = pos_cols.get(suffix)
                neg_col = neg_cols.get(suffix)
                col_noprefix = suffix.removeprefix('Learning Rate for ')
    
                pos_vols, pos_stabs = self._collect_pairs(pos_col,unique_units,area_learning_rate_df) if pos_col else ([], [])
                neg_vols, neg_stabs = self._collect_pairs(neg_col,unique_units,area_learning_rate_df) if neg_col else ([], [])
    
                # Skip if neither sign has data.
                panels = [(pos_vols, pos_stabs, 'Positive'), (neg_vols, neg_stabs, 'Negative')]
                panels = [(v, s, lbl) for v, s, lbl in panels if len(v) > 0]
                if not panels:
                    continue
    
                fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), sharey=True)
                if len(panels) == 1:
                    axes = [axes]
    
                x_vol, x_stab = 1, 2
                color_increase, color_decrease = 'gray', COLORS[i]
    
                for ax, (vols, stabs, sign_label) in zip(axes, panels):
                    num_units = len(vols)
    
                    for vol, stab in zip(vols, stabs):
                        color = color_increase if stab >= vol else color_decrease
                        ax.plot([x_vol, x_stab], [vol, stab], color=color)
                        ax.scatter([x_vol, x_stab], [vol, stab], color=color)
    
                    # Group means
                    ax.plot([x_vol, x_stab], [np.mean(vols), np.mean(stabs)], color='k', linewidth=2)
                    ax.scatter([x_vol, x_stab], [np.mean(vols), np.mean(stabs)],
                               color='k', linewidth=2, marker="D")
    
                    ax.set_xlim(0.8, 2.2)
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_xticks([x_vol, x_stab])
                    ax.set_xticklabels(["Volatile", "Stable"], fontsize=12, fontweight="bold")
                    ax.set_ylabel(f'{sign_label} Learning Rate', fontsize=12)
    
                    legend_handles = [
                        mpatches.Patch(color=color_increase, label="Increased"),
                        mpatches.Patch(color=color_decrease, label="Decreased"),
                        plt.Line2D([0], [0], color='k', marker="D", markersize=7,
                                   label="Mean", linewidth=2),
                    ]
                    ax.legend(handles=legend_handles, fontsize=8.5, frameon=False,
                              loc="upper left", labelcolor="#444")
                    ax.set_title(
                        f'{sign_label} LR\n'
                        f'Total number of {area} {col_noprefix} units: {num_units}'
                    )
    
                fig.suptitle(suptitle)
                
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_pos_neg_paired_changes',
                    fig_folder,
                    PROJ_FOLDER,
                )
    
        return

    
    
    def plot_learning_rate_scatter_1(self, learning_rate_df, neuron_type_df, suptitle, fig_folder):
    
        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']
    
        # Pair positive and negative LR columns by their shared neuron-type suffix.
        # Expected naming: 'Positive Learning rate for Q...' / 'Negative Learning rate for Q...'
        pos_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Pos')}
        neg_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Neg')}
        all_suffixes = sorted(set(pos_cols) | set(neg_cols))
    
        for i, area in enumerate(areas):
    
            area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)
    
            for suffix in all_suffixes:
                pos_col = pos_cols.get(suffix)
                neg_col = neg_cols.get(suffix)
                col_noprefix = suffix.removeprefix('Learning Rate for ')
    
                
    
                pos_vols, pos_stabs = self._collect_pairs(pos_col,unique_units,area_learning_rate_df) if pos_col else ([], [])
                neg_vols, neg_stabs = self._collect_pairs(neg_col,unique_units,area_learning_rate_df) if neg_col else ([], [])
    
                # Skip if neither sign has data.
                panels = [(pos_vols, pos_stabs, 'Positive'), (neg_vols, neg_stabs, 'Negative')]
                panels = [(v, s, lbl) for v, s, lbl in panels if len(v) > 0]
                if not panels:
                    continue
    
                fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), sharey=True)
                if len(panels) == 1:
                    axes = [axes]
    
                color_increase, color_decrease = 'gray', COLORS[i]
    
                for ax, (vols, stabs, sign_label) in zip(axes, panels):
                    num_units = len(vols)
    
                    for vol, stab in zip(vols, stabs):
                        color = color_increase if stab >= vol else color_decrease
                        ax.scatter(vol, stab, color=color)
    
                    # Mean
                    ax.scatter(np.mean(vols), np.mean(stabs),
                               color='k', linewidth=2, marker="D")
                    
                    # Diagonal
                    ax.plot([0,1],[0,1],'--',linewidth=2,color='k')
    
                    ax.set_xlim(-0.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_ylabel(f'Stable Blocks Learning Rate', fontsize=12)
                    ax.set_xlabel(f'Volatile Blocks Learning Rate', fontsize=12)
    
                    legend_handles = [
                        mpatches.Patch(color=color_increase, label="Increased"),
                        mpatches.Patch(color=color_decrease, label="Decreased"),
                        plt.Line2D([0], [0], color='k', marker="D", markersize=7,
                                   label="Mean", linewidth=2),
                        plt.Line2D([0], [0], color='k', linestyle="--",
                                   label="Diagonal", linewidth=2)
                    ]
                    ax.legend(handles=legend_handles, fontsize=8.5, frameon=False,
                              loc="upper left", labelcolor="#444")
                    ax.set_title(
                        f'{sign_label} Learning Rate\n'
                        f'Total number of {area} {col_noprefix} units: {num_units}'
                    )
    
                fig.suptitle(suptitle)
                
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_scatter',
                    fig_folder,
                    PROJ_FOLDER,
                )
    
        return
    
    
    def plot_learning_rate_scatter_2(self, learning_rate_df, neuron_type_df, suptitle, fig_folder):
    
        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']
    
        # Pair positive and negative LR columns by their shared neuron-type suffix.
        # Expected naming: 'Positive Learning rate for Q...' / 'Negative Learning rate for Q...'
        pos_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Pos')}
        neg_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Neg')}
        all_suffixes = sorted(set(pos_cols) | set(neg_cols))
    
        for i, area in enumerate(areas):
    
            area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)
    
            for suffix in all_suffixes:
                pos_col = pos_cols.get(suffix)
                neg_col = neg_cols.get(suffix)
                col_noprefix = suffix.removeprefix('Learning Rate for ')
    
                
    
                pos_vols, pos_stabs = self._collect_pairs(pos_col,unique_units,area_learning_rate_df) if pos_col else ([], [])
                neg_vols, neg_stabs = self._collect_pairs(neg_col,unique_units,area_learning_rate_df) if neg_col else ([], [])
    
                # Skip if neither sign has data.
                panels = [(pos_stabs, neg_stabs, 'Stable'), (pos_vols, neg_vols, 'Volatile')]
                panels = [(p, n, lbl) for p, n, lbl in panels if len(p) > 0]
                if not panels:
                    continue
    
                fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), sharey=True)
                if len(panels) == 1:
                    axes = [axes]
    
                color_increase, color_decrease = 'gray', COLORS[i]
    
                for ax, (poss, negs, block_label) in zip(axes, panels):
                    num_units = len(poss)
    
                    for pos, neg in zip(poss, negs):
                        color = color_increase if neg >= pos else color_decrease
                        ax.scatter(pos, neg, color=color)
    
                    # Mean
                    ax.scatter(np.mean(poss), np.mean(negs),
                               color='k', linewidth=2, marker="D")
                    
                    # Diagonal
                    ax.plot([0,1],[0,1],'--',linewidth=2,color='k')
    
                    ax.set_xlim(-0.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_ylabel(f'Negative Learning Rate', fontsize=12)
                    ax.set_xlabel(f'Positive Learning Rate', fontsize=12)
    
                    legend_handles = [
                        mpatches.Patch(color=color_increase, label="Increased"),
                        mpatches.Patch(color=color_decrease, label="Decreased"),
                        plt.Line2D([0], [0], color='k', marker="D", markersize=7,
                                   label="Mean", linewidth=2),
                        plt.Line2D([0], [0], color='k', linestyle="--",
                                   label="Diagonal", linewidth=2)
                    ]
                    ax.legend(handles=legend_handles, fontsize=8.5, frameon=False,
                              loc="upper left", labelcolor="#444")
                    ax.set_title(
                        f'{block_label} Blocks\n'
                        f'Total number of {area} {col_noprefix} units: {num_units}'
                    )
    
                fig.suptitle(suptitle)
                
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_scatter',
                    fig_folder,
                    PROJ_FOLDER,
                )
    
        return



    def plot_learning_rate_dual_hists(self, learning_rate_df, neuron_type_df, suptitle, fig_folder):
    
        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']
    
        value_cols = list(np.unique([self._strip_sign_prefix(c) for c in cols if 'Q' in c]))
        value_cols.append('Qoffer') #to combine Q1 and Q2
    
        for i, area in enumerate(areas):
    
            # area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)
    
            for col in value_cols:
                if col != 'Qoffer':
                    col_noprefix = col.removeprefix('Learning Rate for ')
        
                    pos_vols, pos_stabs = self._collect_pairs(f'Pos {col}',unique_units,area_learning_rate_df)
                    neg_vols, neg_stabs = self._collect_pairs(f'Neg {col}',unique_units,area_learning_rate_df)
                
                elif col == 'Qoffer':
                    col_noprefix = 'Qoffer'
                    
                    #Q1
                    col = 'Learning Rate for Q1'
                    pos_vols1, pos_stabs1 = self._collect_pairs(f'Pos {col}',unique_units,area_learning_rate_df)
                    neg_vols1, neg_stabs1 = self._collect_pairs(f'Neg {col}',unique_units,area_learning_rate_df)
                    
                    #Q2
                    col = 'Learning Rate for Q2'
                    pos_vols2, pos_stabs2 = self._collect_pairs(f'Pos {col}',unique_units,area_learning_rate_df)
                    neg_vols2, neg_stabs2 = self._collect_pairs(f'Neg {col}',unique_units,area_learning_rate_df)
                    
                    pos_vols = pos_vols1 + pos_vols2
                    neg_vols = neg_vols1 + neg_vols2
                    
                    col = 'Learning Rate for Qoffer'
    
                # Skip if neither sign has data.
                if len(pos_vols)==0 and len(neg_vols)==0:
                    continue

                ## Pos LR
                fig,ax = plt.subplots(layout='constrained')
                
                #hists
                ax.hist(pos_vols,label='volatile',bins=20,range=(0,1),alpha=0.5,color='tab:blue')
                ax.hist(pos_stabs,label='stable',bins=20,range=(0,1),alpha=0.5,color='tab:orange')
                #mean markers
                y_max1 = np.max(np.histogram(pos_vols,bins=20,range=(0,1))[0])
                y_max2 = np.max(np.histogram(pos_stabs,bins=20,range=(0,1))[0])
                y_max = np.max([y_max1,y_max2])
                ax.plot(np.mean(pos_vols),y_max,marker='v',color='tab:blue')
                ax.plot(np.mean(pos_stabs),y_max,marker='v',color='tab:orange')
                #plot details
                ax.set_title(f'Positive LR Histogram - {area}\n{col_noprefix}')
                ax.set_ylabel('Number of neurons')
                ax.set_xlabel('Positive Learning Rate')
                ax.legend()

                fig.suptitle(suptitle)
                
                
                #stats
                print('-'*25)
                print('Pos LR: volatile vs stable')
                print(col)
                print(area)
                res = sp.stats.ttest_ind(pos_stabs, pos_vols, equal_var=True)
                print(res)
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_posLRhist',
                    fig_folder,
                    PROJ_FOLDER,
                )

                ## Neg LR
                fig,ax = plt.subplots(layout='constrained')
                
                #hists
                ax.hist(neg_vols,label='volatile',bins=20,range=(0,1),alpha=0.5,color='tab:blue')
                ax.hist(neg_stabs,label='stable',bins=20,range=(0,1),alpha=0.5,color='tab:orange')
                #mean markers
                y_max1 = np.max(np.histogram(neg_vols,bins=20,range=(0,1))[0])
                y_max2 = np.max(np.histogram(neg_stabs,bins=20,range=(0,1))[0])
                y_max = np.max([y_max1,y_max2])
                ax.plot(np.mean(neg_vols),y_max,marker='v',color='tab:blue')
                ax.plot(np.mean(neg_stabs),y_max,marker='v',color='tab:orange')
                #plot details
                ax.set_title(f'Negative LR Histogram - {area}\n{col_noprefix}')
                ax.set_ylabel('Number of neurons')
                ax.set_xlabel('Negative Learning Rate')
                ax.legend()

                fig.suptitle(suptitle)
                
                
                #stats
                print('-'*25)
                print('Neg LR: volatile vs stable')
                print(col)
                print(area)
                res = sp.stats.ttest_ind(neg_stabs, neg_vols, equal_var=True)
                print(res)
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_negLRhist',
                    fig_folder,
                    PROJ_FOLDER,
                )
                
                
        return


    
    
    def plot_winbias_hist(self, learning_rate_df, neuron_type_df, suptitle, fig_folder):
    
        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']
    
        # Pair positive and negative LR columns by their shared neuron-type suffix.
        # Expected naming: 'Positive Learning rate for Q...' / 'Negative Learning rate for Q...'
        pos_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Pos')}
        neg_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Neg')}
        all_suffixes = sorted(set(pos_cols) | set(neg_cols))
    
        for i, area in enumerate(areas):
    
            area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)
    
            for suffix in all_suffixes:
                pos_col = pos_cols.get(suffix)
                neg_col = neg_cols.get(suffix)
                col_noprefix = suffix.removeprefix('Learning Rate for ')
    
                
    
                pos_vols, pos_stabs = self._collect_pairs(pos_col,unique_units,area_learning_rate_df) if pos_col else ([], [])
                neg_vols, neg_stabs = self._collect_pairs(neg_col,unique_units,area_learning_rate_df) if neg_col else ([], [])
    
                # Skip if neither sign has data.
                if len(pos_vols)==0 and len(neg_vols)==0:
                    continue
                
                
                # winbias histogram
                winbias_vols = np.array(pos_vols) - np.array(neg_vols)
                winbias_stabs = np.array(pos_stabs) - np.array(neg_stabs)
                
                fig,ax = plt.subplots(layout='constrained')
                
                #hists
                ax.hist(winbias_vols,label='volatile',bins=20,range=(-1,1),alpha=0.5,color='tab:blue')
                ax.hist(winbias_stabs,label='stable',bins=20,range=(-1,1),alpha=0.5,color='tab:orange')
                #mean markers
                y_max1 = np.max(np.histogram(winbias_vols,bins=20,range=(-1,1))[0])
                y_max2 = np.max(np.histogram(winbias_stabs,bins=20,range=(-1,1))[0])
                y_max = np.max([y_max1,y_max2])
                ax.plot(np.mean(winbias_vols),y_max,marker='v',color='tab:blue')
                ax.plot(np.mean(winbias_stabs),y_max,marker='v',color='tab:orange')
                #plot details
                ax.set_title(f'Win-Bias Histogram - {area}\n{col_noprefix}')
                ax.set_ylabel('Number of neurons')
                ax.set_xlabel('Win-Bias\nPos LR - Neg LR')
                ax.legend()

                fig.suptitle(suptitle)
                
                
                #stats
                print('-'*25)
                print('Win-Bias: volatile vs stable')
                print(suffix)
                print(area)
                res = sp.stats.ttest_ind(winbias_stabs, winbias_vols, equal_var=True)
                print(res)
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_winbiashist',
                    fig_folder,
                    PROJ_FOLDER,
                )
                
                
                # change in winbias histogram
                winbias_change = winbias_vols - winbias_stabs
                
                fig,ax = plt.subplots(layout='constrained')
                #histogram
                ax.hist(winbias_change,label='delta',bins=40,range=(-2,2),alpha=0.5,color='tab:green')
                #mean marker
                y_max = np.max(np.histogram(winbias_change,bins=40,range=(-2,2))[0])
                ax.plot(np.mean(winbias_change),y_max,marker='v',color='tab:green')
                #plot details
                ax.set_title(f'Change in Win-Bias Histogram - {area}\n{col_noprefix}')
                ax.set_ylabel('Number of neurons')
                ax.set_xlabel('Delta Win-Bias\nVolatile - Stable')
                ax.legend()

                fig.suptitle(suptitle)
                
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_delwinbiashist',
                    fig_folder,
                    PROJ_FOLDER,
                )
                
                
                # winbias scatter plot
                fig,ax = plt.subplots(layout='constrained')
                ax.scatter(winbias_vols,winbias_stabs)
                ax.set_title(f'Win-Bias in volatile vs stable blocks/n{area} - {col_noprefix}')
                ax.set_ylabel('Win-Bias in Stable Block')
                ax.set_xlabel('Win-Bias in Volatile Block')

                fig.suptitle(suptitle)
                
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_winbiasscatter',
                    fig_folder,
                    PROJ_FOLDER,
                )    
                
                
        return    
    
    
    
    def _collect_pairs_with_units(self, lr_col, unique_units, area_learning_rate_df):
        """Return {unit: (vol, stab)} for units with non-NaN data in both subsessions."""
        result = {}
        for unit in unique_units:
            volatile = utils.get_unit_data(
                area_learning_rate_df, f'{unit} - {SUBSESSIONS[2]}', lr_col
            )
            stable = utils.get_unit_data(
                area_learning_rate_df, f'{unit} - {SUBSESSIONS[1]}', lr_col
            )
            if not np.isnan(volatile) and not np.isnan(stable):
                result[unit] = (volatile, stable)
        return result


    def plot_learning_rate_scatter_all_combined(self, learning_rate_df, neuron_type_df, suptitle, fig_folder):
    
        cols = learning_rate_df.columns
        areas = AREAS + ['all areas']
    
        # Pair positive and negative LR columns by their shared neuron-type suffix.
        # Expected naming: 'Positive Learning rate for Q...' / 'Negative Learning rate for Q...'
        pos_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Pos')}
        neg_cols = {self._strip_sign_prefix(c): c for c in cols if 'Q' in c and c.startswith('Neg')}
        all_suffixes = sorted(set(pos_cols) | set(neg_cols))
    
        for i, area in enumerate(areas):
    
            area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            unique_units = utils.get_unique_units(area_learning_rate_df["Unit"], SUBSESSIONS)
    
            for suffix in all_suffixes:
                pos_col = pos_cols.get(suffix)
                neg_col = neg_cols.get(suffix)
                col_noprefix = suffix.removeprefix('Learning Rate for ')
    
                pos_data = (self._collect_pairs_with_units(pos_col, unique_units, area_learning_rate_df)
                            if pos_col else {})
                neg_data = (self._collect_pairs_with_units(neg_col, unique_units, area_learning_rate_df)
                            if neg_col else {})
    
                if not pos_data and not neg_data:
                    continue
    
                fig, ax = plt.subplots(figsize=(6, 5))
    
                color_increase, color_decrease = 'gray', COLORS[i]
    
                # Units present in both get a connecting line.
                shared_units = set(pos_data) & set(neg_data)
                for unit in shared_units:
                    pos_vol, pos_stab = pos_data[unit]
                    neg_vol, neg_stab = neg_data[unit]
                    ax.plot([pos_vol, neg_vol], [pos_stab, neg_stab],
                            color='lightgray', linewidth=0.8, zorder=1)
    
                # Positive points ('+')
                for unit, (vol, stab) in pos_data.items():
                    color = color_increase if stab >= vol else color_decrease
                    ax.scatter(vol, stab, color=color, marker='+', s=80,
                               linewidths=1.5, zorder=2)
    
                # Negative points ('-')
                for unit, (vol, stab) in neg_data.items():
                    color = color_increase if stab >= vol else color_decrease
                    ax.scatter(vol, stab, color=color, marker='_', s=80,
                               linewidths=1.5, zorder=2)
    
                # Means — same +/- markers as the data, but bigger and bolder so they stand out
                if pos_data:
                    pos_vols = [v for v, s in pos_data.values()]
                    pos_stabs = [s for v, s in pos_data.values()]
                    ax.scatter(np.mean(pos_vols), np.mean(pos_stabs),
                               color='k', marker='+', s=400, linewidths=4, zorder=3)
                if neg_data:
                    neg_vols = [v for v, s in neg_data.values()]
                    neg_stabs = [s for v, s in neg_data.values()]
                    ax.scatter(np.mean(neg_vols), np.mean(neg_stabs),
                               color='k', marker='_', s=400, linewidths=4, zorder=3)
    
                # Diagonal
                ax.plot([0, 1], [0, 1], '--', linewidth=2, color='k')
    
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                ax.set_ylabel('Stable Blocks Learning Rate', fontsize=12)
                ax.set_xlabel('Volatile Blocks Learning Rate', fontsize=12)
    
                legend_handles = [
                mpatches.Patch(color=color_increase, label="Increased"),
                mpatches.Patch(color=color_decrease, label="Decreased"),
                plt.Line2D([0], [0], color='gray', marker="+", markersize=10,
                           label="Positive LR", linewidth=0, markeredgewidth=1.5),
                plt.Line2D([0], [0], color='gray', marker="_", markersize=10,
                           label="Negative LR", linewidth=0, markeredgewidth=1.5),
                plt.Line2D([0], [0], color='k', linestyle="--",
                           label="Diagonal", linewidth=2)
            ]
                ax.legend(handles=legend_handles, fontsize=8.5, frameon=False,
                          loc="upper left", labelcolor="#444")
    
                n_pos, n_neg, n_shared = len(pos_data), len(neg_data), len(shared_units)
                ax.set_title(
                    f'{area} {col_noprefix} units\n'
                    f'num units: {n_shared}'
                )
    
                fig.suptitle(suptitle)
                
    
                utils.save_svg(
                    fig,
                    f'{suptitle}_{area}_{col_noprefix}_scatter',
                    fig_folder,
                    PROJ_FOLDER,
                )
    
        return
    
        
    def run_tuning_curves(self):
        
        for s,session in enumerate(SESSIONS):
            
            start = time.time()
            
            print(session)
            
            
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            neuron_type_df = utils.load_csv('NeuronTypes',session,DATA_FOLDER)
            learning_rate_df = utils.load_csv('LearningRates',session,DATA_FOLDER)
            
            print('data loaded')
            
            plot_tuning_curves(Spikes.fr_df,Behav.behavior_df,neuron_type_df,learning_rate_df)
            
            utils.get_time_left(start,s,len(SESSIONS))        
            
        return

class NeuronType_Analysis:
    
    def __init__(self,overwrite_flag=False):
        
        self.fig_folder = 'NeuronTypeVenns'
       
        self.dfs = []
        self.coefs = []
        for s,session in enumerate(SESSIONS):
            
            start = time.time()
            #Get encodings
            if not overwrite_flag: #don't overwrite, load
                neuron_type_df = utils.load_csv('NeuronTypes',session,DATA_FOLDER)
            else: #don't load, compute
                Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
                neuron_type_df,coef = self.get_neuron_types(Spikes.fr_df,Behav.behavior_df)
                self.coefs.append(coef)
                if neuron_type_df is not None:
                    utils.save_csv(neuron_type_df,'NeuronTypes',session,DATA_FOLDER)
                else:
                    print(f'{session} yielded no neuron types!')
                    
            if neuron_type_df is not None:        
                self.dfs.append(neuron_type_df)
                
            utils.get_time_left(start,s,len(SESSIONS))
                
        self.sessavg_df = utils.merge_sessions_df(self.dfs)
        
        
        for area in AREAS+['all areas']:
            area_df = utils.area_parser(self.sessavg_df,area)
            subsession_df = area_df[area_df['Unit'].str.contains(SUBSESSIONS[0])]
            suptitle = f'SessionAvg - {area}'
            self.plot_neuron_type_venns(subsession_df,suptitle,self.fig_folder)
            
        # self.plot_lasso_coefs(self.coefs)
        
        return
    
    
    def plot_lasso_coefs(self,coefs):
        
        # plot hist of lasso/ridge coefficients        
        fig,axs = plt.subplots(2,1,layout='constrained')
        range_ = (-1.5,2.5)
        ax=axs[0]
        ax.hist(np.array(utils.flatten(coefs)),range=range_)
        ax.set_xlabel('all coefs of data regularization regressions')
        ax.set_ylabel('Num')
        ax.legend(REGRESSORS)
        ax=axs[1]
        ax.hist(np.array(utils.flatten(coefs)).flatten(),range=range_,bins=50)
        ax.set_xlabel('all coefs of data regularization regressions')
        ax.set_ylabel('Num')
        
        return
    
    
    def get_neuron_types(self,fr_df,behavior_df):
        coefs=[]
        encodings = []
        unit_labels = []
        neuron_type_df = None
        for stable_or_volatile in SUBSESSIONS:
            
            trials = utils.get_trials(behavior_df,stable_or_volatile)
            
            if len(trials) < 1:
                continue #skip iteration if there aren't any trials for this subsession
            
            # n_regressors = len(REGRESSORS)
            # regressor_matrix = behavior_df[REGRESSORS].iloc[trials] #shape = (trials x regressors)
            
            area_fr_df = utils.area_parser(fr_df,'all areas')
            units = list(area_fr_df.keys())
            
            if len(units) < 1:
                return None #skip whole thing if there aren't any units for this session
            
            ## Run regression for each unit against regressors
            for i,unit in enumerate(units): #loop thru all units of session
                
                # linear reg
                signif_regressors = utils.linear_encoding_regression_feature_selection(
                    REGRESSORS,fr_df[unit].iloc[trials],behavior_df.iloc[trials],ALPHA_THRESHOLD)
                
                # #data regularization
                # signif_regressors,coef = utils.regularization_regression_feature_selection(
                #     REGRESSORS,fr_df[unit].iloc[trials],behavior_df.iloc[trials],'lasso',0.001,1e-4)
                # coefs.append(coef)    
        
                # res = utils.linear_encoding_regression(fr, regressor_matrix)
                
                # pvals = res.pvalues[1:] #exclude constant (position 0)
                
                # if res.f_pvalue < ALPHA_THRESHOLD:
                #     signif_regressors = str([str(feat) for feat in np.array(REGRESSORS)[pvals < ALPHA_THRESHOLD]])
                #     if len(signif_regressors) < 1:
                #         signif_regressors = 'none'
                # else:
                #     signif_regressors = 'none'
                    
                encodings.append(signif_regressors)
                unit_labels.append(f'{unit} - {stable_or_volatile}')
        
        neuron_type_df =  pd.DataFrame.from_dict({'Unit':unit_labels, 'Neuron Type':encodings})
         
        return neuron_type_df, coefs        
        
        
    def plot_neuron_type_venns(self, neuron_type_df, suptitle, fig_folder):
         # For variable names (e.g. VcO):
         # uppercase = contains
         # lowercase = does not contain
         
         # Boolean algebra:
         # * = AND
         # + = OR
         
         # In future, make area and stable vs volatile parameters ??  
         
         def count(encoding_str):
             return sum(neuron_type_df['Neuron Type'].str.contains(encoding_str))
         
         def contains(encoding_str):
             return neuron_type_df['Neuron Type'].str.contains(encoding_str)
         
         def no_contains(encoding_str):
             return ~neuron_type_df['Neuron Type'].str.contains(encoding_str)
         
         def plot_venn2(Ab, aB, AB, labelA, labelB):
             fig,ax = plt.subplots(layout='constrained')
             matplotlib_venn.venn2(subsets = (Ab, aB, AB),
                   set_labels = (labelA, labelB))
             return fig,ax
         
         def plot_venn3(Abc, aBc, ABc, abC, AbC, aBC, ABC, labelA, labelB, labelC):
             fig,ax = plt.subplots(layout='constrained')
             matplotlib_venn.venn3(subsets = (Abc, aBc, ABc, abC, AbC, aBC, ABC),
                   set_labels = (labelA, labelB, labelC))
             return fig,ax
         
         # value (V) vs choice (C) vs action/time (O)
         Vco = sum( contains('Q') *       no_contains('Choice') * (no_contains('Side') *   no_contains('Time')) )
         vCo = sum( no_contains('Q') *    contains('Choice') *    (no_contains('Side') *   no_contains('Time')) )
         VCo = sum( contains('Q') *       contains('Choice') *    (no_contains('Side') *   no_contains('Time')) )
         vcO = sum( no_contains('Q') *    no_contains('Choice') * (contains('Side') +      contains('Time')) )
         VcO = sum( contains('Q') *       no_contains('Choice') * (contains('Side') +      contains('Time')) )
         vCO = sum( no_contains('Q') *    contains('Choice') *    (contains('Side') +      contains('Time')) )
         VCO = sum( contains('Q') *       contains('Choice') *    (contains('Side') +      contains('Time')) )
         fig,ax = plot_venn3(Vco,vCo,VCo,vcO,VcO,vCO,VCO, 'Value','Choice','Other')
         fig.suptitle(suptitle)
         utils.save_svg(fig,f'Value-Choice-Other_{suptitle}',fig_folder,PROJ_FOLDER)

         # Q1 (H) vs Q2 (A) vs Qchosen (C)
         Hac = sum( contains('Q1') *       no_contains('Q2') * no_contains('Qchosen') )
         hAc = sum( no_contains('Q1') *    contains('Q2') *    no_contains('Qchosen') )
         HAc = sum( contains('Q1') *       contains('Q2') *    no_contains('Qchosen') )
         haC = sum( no_contains('Q1') *    no_contains('Q2') * contains('Qchosen') )
         HaC = sum( contains('Q1') *       no_contains('Q2') * contains('Qchosen') )
         hAC = sum( no_contains('Q1') *    contains('Q2') *    contains('Qchosen') )
         HAC = sum( contains('Q1') *       contains('Q2') *    contains('Qchosen') )
         fig,ax = plot_venn3(Hac,hAc,HAc,haC,HaC,hAC,HAC, 'Q1','Q2','Qchosen')
         fig.suptitle(suptitle)
         utils.save_svg(fig,f'Q1-Q2-Qchosen_{suptitle}',fig_folder,PROJ_FOLDER)
         
         
         # Bar chart of all regressors 
         counts=[]
         for reg in REGRESSORS:
             counts.append(count(reg))
         
         idx_sort = np.argsort(counts)   
         fig,ax = plt.subplots(layout='constrained')
         ax.bar(np.array(REGRESSORS)[idx_sort[::-1]],np.array(counts)[idx_sort[::-1]])
         ax.set_ylabel('Num Neurons')
         for tick in ax.xaxis.get_major_ticks()[1::2]:
             tick.set_pad(15)
         fig.suptitle(suptitle)
         utils.save_svg(fig, f'RegBarChart_{suptitle}', fig_folder,PROJ_FOLDER)
         
         return  




#### test

class DualVsSingleAlpha_Analysis:
    def __init__(self):
        return

    def get_rsqrs(self):
        DUAL_LRS = np.linspace(start=0.05,stop=0.95,num=19)
        SINGLE_LRS = np.linspace(start=0.01,stop=0.99,num=50)
        dual_rsqrs=[]
        single_rsqrs=[]
        for s,session in enumerate(SESSIONS):

            # if session == "braz20260911_03_te2608":
            if True:

                start = time.time()
                print(session)
                Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
                neuron_type_df = utils.load_csv('NeuronTypes',session,DATA_FOLDER)
                dual_rsqr = self.get_learning_rates_rsqr_dual(Spikes.fr_df,Behav,neuron_type_df,DUAL_LRS)
                single_rsqr = self.get_learning_rates_rsqr_single(Spikes.fr_df,Behav,neuron_type_df,SINGLE_LRS)
                dual_rsqrs = dual_rsqrs + dual_rsqr
                single_rsqrs = single_rsqrs + single_rsqr

                utils.get_time_left(start,s,len(SESSIONS))

        dict_out = {'dual_rsqrs':dual_rsqrs, 'single_rsqrs':single_rsqrs}
        utils.save_pkl(dict_out,'LR_Rsqrs','',DATA_FOLDER)
        return

    def plot_rsqrs(self):

        dict_in = utils.load_pkl('LR_Rsqrs','',DATA_FOLDER)
        single_rsqr = dict_in['single_rsqrs']
        dual_rsqr = dict_in['dual_rsqrs']
        # dict_in = utils.load_pkl('LR_Rsqrs1','',DATA_FOLDER)
        # single_rsqr = single_rsqr + dict_in['single_rsqrs']
        # dual_rsqr = dual_rsqr + dict_in['dual_rsqrs']

        dual_avg,dual_sem = utils.get_avg_sem(dual_rsqr,axis=0)
        single_avg,single_sem = utils.get_avg_sem(single_rsqr,axis=0)

        fig,ax=plt.subplots(layout="constrained")
        ax.bar([0,1],[single_avg,dual_avg],yerr=[single_sem,dual_sem],tick_label=['Single LR','Dual LR'])
        # ax.set_xticks([0,1],['Single LR','Dual LR'])
        ax.scatter(np.zeros(len(single_rsqr)),single_rsqr,alpha=0.2)
        ax.scatter(np.ones(len(dual_rsqr)),dual_rsqr,alpha=0.2)
        # ax.set_xlim([-0.5,1.5])
        ax.set_ylim([0,0.2])

        ax.set_ylabel('R-Squared')
        ax.set_title(f'Measuring Fit: Single vs Dual LRs\nn={len(dual_rsqr)}')
        plt.show()
        print(f'DUAL: Avg +/- sem FR: {dual_avg:.4f} +/- {dual_sem:.4f}')
        print(f'SINGLE: Avg +/- sem FR: {single_avg:.4f} +/- {single_sem:.4f}')
        print('Dual vs Single')
        res = sp.stats.ttest_ind(dual_rsqr, single_rsqr, equal_var=True)
        print(res)

        return

    def get_learning_rates_rsqr_single(fr_df, Behavior, neuron_type_df, learning_rates_to_test):

        print('SINGLE')
        df_rows_learning_rates = []
        rsqrs = []
        print(f'{len(fr_df.columns)} neurons')

        for stable_or_volatile in ['all trials']:

            assert len(Behavior.behavior_df) == len(fr_df), 'Number of trials do not match!'
            # num_trials = len(fr_df)

            trials = utils.get_trials(Behavior.behavior_df, stable_or_volatile)

            if not len(trials) > 1:  # if no trials for current subsession
                print(f'No trials for {stable_or_volatile}. Skipping!')
                continue
            else:
                print(stable_or_volatile)

            # Get FR for each neuron
            units = fr_df.columns
            # for i,unit in enumerate(units):
            #     neuron_type = utils.get_unit_data(neuron_type_df,f'{unit} - {stable_or_volatile}','Neuron Type')
            #     print(neuron_type)
            #     print('Q' in neuron_type)
            # xxx
            for i, unit in enumerate(units):

                start = time.time()
                neuron_type = str(utils.get_unit_data(neuron_type_df, f'{unit} - {stable_or_volatile}', 'Neuron Type'))
                if "Q" not in neuron_type:
                    continue

                fr = fr_df[unit].iloc[trials]  # vector of FRs for subsession trial

                ## Get regression weights for all learning rate combos
                # loop thru all possible learning rates
                df_rows_regression_weights = []
                for alpha in learning_rates_to_test:

                    # get beahvior according to current learning rates
                    Behavior.get_behavior(Q_learning=True, learning_rates=[alpha])
                    regressor_matrix = Behavior.behavior_df[REGRESSORS].iloc[trials]

                    # regress fr against this learning rate behavior
                    res = utils.linear_encoding_regression(fr, regressor_matrix)

                    row_dict_regression_weights = {'alpha': alpha,
                                                   'rsqr': res.rsquared}  # each row of the df will be the lrs for each reg
                    # Get weights for each regressor
                    regressor_names = res.params.keys()  # includes const in idx 0

                    # loop thru all regressors
                    for j in range(len(regressor_names)):
                        if res.pvalues[
                            regressor_names[j]] < ALPHA_THRESHOLD and res.f_pvalue < ALPHA_THRESHOLD:  # if signif
                            row_dict_regression_weights[regressor_names[j]] = abs(
                                res.params.values[j])  # fill in the row of the df with param
                        else:
                            row_dict_regression_weights[
                                regressor_names[j]] = np.nan  # if not signif, put in nan so it will get skipped

                    df_rows_regression_weights.append(row_dict_regression_weights)

                # Organize regression weights for each learning rate into a df
                regression_weights_df = pd.DataFrame(df_rows_regression_weights)
                # print(regression_weights_df)
                # xxx
                # utils.save_csv(regression_weights_df,'RegressionWeights',"debug",DATA_FOLDER)

                # Find learning rate combination which maximizes each regression weight
                unit_subsession_labeled = f'{unit} - {stable_or_volatile}'
                row_dict_learning_rates = {'Unit': unit_subsession_labeled}
                for j in range(len(REGRESSORS)):

                    # only do if this neuron was found to encode that regressor
                    if REGRESSORS[j] in neuron_type:

                        if not np.all(np.isnan(
                                regression_weights_df[REGRESSORS[j]])):  # to catch if all regression weights are nan
                            # idx of max regression weight for current regressor
                            idx_max_regressor = regression_weights_df[REGRESSORS[j]].idxmax()

                            # Find and store learning rate which corresponds to max regression weight
                            row_dict_learning_rates['Learning Rate for ' + REGRESSORS[j]] = \
                            regression_weights_df['alpha'].iloc[idx_max_regressor]
                            # row_dict_learning_rates['rsqr for ' + REGRESSORS[j]] = regression_weights_df['rsqr'].iloc[idx_max_regressor]
                            rsqrs.append(regression_weights_df['rsqr'].iloc[idx_max_regressor])
                            print('rsqr saved!')

                        else:  # if all weights are nan
                            # Put in np.nan since its more consistent when saving and loading dfs
                            row_dict_learning_rates['Learning Rate for ' + REGRESSORS[j]] = np.nan
                            # row_dict_learning_rates['rsqr for ' + REGRESSORS[j]] = np.
                            print()

                    else:  # if regressor not encoded by this unit
                        # Put in np.nan since its more consistent when saving and loading dfs
                        row_dict_learning_rates['Learning Rate for ' + REGRESSORS[j]] = np.nan
                        # row_dict_learning_rates['rsqr for ' + REGRESSORS[j]] = np.nan

                df_rows_learning_rates.append(row_dict_learning_rates)

                # print(f'Unit {i+1}/{len(units)} for {stable_or_volatile}')
                utils.get_time_left(start, i, len(units))

        return rsqrs

    def get_learning_rates_rsqr_dual(fr_df, Behavior, neuron_type_df, learning_rates_to_test):
        print('DUAL')
        df_rows_learning_rates = []
        rsqrs = []
        print(f'{len(fr_df.columns)} neurons')

        for stable_or_volatile in ['all trials']:

            assert len(Behavior.behavior_df) == len(fr_df), 'Number of trials do not match!'
            # num_trials = len(fr_df)

            trials = utils.get_trials(Behavior.behavior_df, stable_or_volatile)

            if not len(trials) > 1:  # if no trials for current subsession
                print(f'No trials for {stable_or_volatile}. Skipping!')
                continue
            else:
                print(stable_or_volatile)

            # Get FR for each neuron
            units = fr_df.columns
            # for i,unit in enumerate(units):
            #     neuron_type = utils.get_unit_data(neuron_type_df,f'{unit} - {stable_or_volatile}','Neuron Type')
            #     print(neuron_type)
            #     print('Q' in neuron_type)
            # xxx
            for i, unit in enumerate(units):

                start = time.time()

                neuron_type = str(utils.get_unit_data(neuron_type_df, f'{unit} - {stable_or_volatile}', 'Neuron Type'))
                if "Q" not in neuron_type:
                    continue

                fr = fr_df[unit].iloc[trials]  # vector of FRs for subsession trial

                ## Get regression weights for all learning rate combos
                # loop thru all possible learning rates
                df_rows_regression_weights = []
                for pos_alpha in learning_rates_to_test:
                    for neg_alpha in learning_rates_to_test:

                        # get beahvior according to current learning rates
                        Behavior.get_behavior(Q_learning=True, learning_rates=[pos_alpha, neg_alpha])
                        regressor_matrix = Behavior.behavior_df[REGRESSORS].iloc[trials]

                        # regress fr against this learning rate behavior
                        res = utils.linear_encoding_regression(fr, regressor_matrix)

                        row_dict_regression_weights = {'pos_alpha': pos_alpha, 'neg_alpha': neg_alpha,
                                                       'rsqr': res.rsquared}  # each row of the df will be the lrs for each reg
                        # Get weights for each regressor
                        regressor_names = res.params.keys()  # includes const in idx 0

                        # loop thru all regressors
                        for j in range(len(regressor_names)):
                            if res.pvalues[
                                regressor_names[j]] < ALPHA_THRESHOLD and res.f_pvalue < ALPHA_THRESHOLD:  # if signif
                                row_dict_regression_weights[regressor_names[j]] = abs(
                                    res.params.values[j])  # fill in the row of the df with param
                            else:
                                row_dict_regression_weights[
                                    regressor_names[j]] = np.nan  # if not signif, put in nan so it will get skipped

                        df_rows_regression_weights.append(row_dict_regression_weights)

                # Organize regression weights for each learning rate into a df
                regression_weights_df = pd.DataFrame(df_rows_regression_weights)
                # print(regression_weights_df)
                # xxx
                # utils.save_csv(regression_weights_df,'RegressionWeights',"debug",DATA_FOLDER)

                # Find learning rate combination which maximizes each regression weight
                unit_subsession_labeled = f'{unit} - {stable_or_volatile}'
                row_dict_learning_rates = {'Unit': unit_subsession_labeled}
                print(neuron_type)
                for j in range(len(REGRESSORS)):

                    # only do if this neuron was found to encode that regressor
                    if REGRESSORS[j] in neuron_type:

                        if not np.all(np.isnan(
                                regression_weights_df[REGRESSORS[j]])):  # to catch if any regression weights are nan
                            # idx of max regression weight for current regressor
                            idx_max_regressor = regression_weights_df[REGRESSORS[j]].idxmax()

                            # Find and store learning rate which corresponds to max regression weight
                            row_dict_learning_rates['Pos Learning Rate for ' + REGRESSORS[j]] = \
                            regression_weights_df['pos_alpha'].iloc[idx_max_regressor]
                            row_dict_learning_rates['Neg Learning Rate for ' + REGRESSORS[j]] = \
                            regression_weights_df['neg_alpha'].iloc[idx_max_regressor]
                            # row_dict_learning_rates['rsqr for ' + REGRESSORS[j]] = regression_weights_df['rsqr'].iloc[idx_max_regressor]
                            rsqrs.append(regression_weights_df['rsqr'].iloc[idx_max_regressor])
                            print('rsqr saved!')

                        else:  # if all weights are nan
                            # Put in np.nan since its more consistent when saving and loading dfs
                            row_dict_learning_rates['Pos Learning Rate for ' + REGRESSORS[j]] = np.nan
                            row_dict_learning_rates['Neg Learning Rate for ' + REGRESSORS[j]] = np.nan
                            # row_dict_learning_rates['rsqr for ' + REGRESSORS[j]] = np.nan

                    else:  # if regressor not encoded by this unit
                        # Put in np.nan since its more consistent when saving and loading dfs
                        row_dict_learning_rates['Pos Learning Rate for ' + REGRESSORS[j]] = np.nan
                        row_dict_learning_rates['Neg Learning Rate for ' + REGRESSORS[j]] = np.nan
                        # row_dict_learning_rates['rsqr for ' + REGRESSORS[j]] = np.nan

                df_rows_learning_rates.append(row_dict_learning_rates)

                # print(f'Unit {i+1}/{len(units)} for {stable_or_volatile}')
                utils.get_time_left(start, i, len(units))

        return rsqrs


class BehaviorLearningRates:

    def __init__(self):
        return

    @staticmethod
    def plot_behavior_learningrates_dual():

        assert VALUE_MODEL == 'DualBaseline'
        # subsessions = ['stable block','volatile block']

        pos_lrs = np.zeros((len(SESSIONS),len(SUBSESSIONS)))
        neg_lrs = np.zeros((len(SESSIONS),len(SUBSESSIONS)))
        betas = np.zeros((len(SESSIONS),len(SUBSESSIONS)))
        for ss,subsession in enumerate(SUBSESSIONS):

            print('*'*50)
            print(subsession)
            print('*'*50)

            for s,session in enumerate(SESSIONS):
                start=time.time()
                Behav = ProcessBehavior(session,subsession)
                param_dict = Behav.Q_learning_params

                pos_lrs[s,ss] = param_dict['pos_alpha']
                neg_lrs[s,ss] = param_dict['neg_alpha']
                betas[s,ss] = param_dict['beta']

                utils.get_time_left(start,s,len(SESSIONS))


        fig,axs=plt.subplots(1,3,layout='constrained')
        fig.suptitle(f'All Sessions (n={len(SESSIONS)})')

        #pos lr
        ax=axs[0]
        ax.set_title('Positive LR')
        ax.boxplot(pos_lrs,tick_labels=SUBSESSIONS)
        ax.set_ylim([-0.1,1.1])
        ax.set_ylabel('Learning Rate')
        ax.tick_params(axis='x', labelrotation=45)

        #neg lr
        ax=axs[1]
        ax.set_title('Negative LR')
        ax.boxplot(neg_lrs,tick_labels=SUBSESSIONS)
        ax.set_ylim([-0.1,1.1])
        # ax.set_ylabel('Learning Rate')
        ax.set_yticks([])
        ax.tick_params(axis='x', labelrotation=45)

        #beta (inverse temp)
        ax=axs[2]
        ax.set_title('Beta (Inverse Temperature)')
        ax.boxplot(betas,tick_labels=SUBSESSIONS)
        ax.set_ylim([-0.1,20])
        ax.set_ylabel('Beta')
        ax.tick_params(axis='x', labelrotation=45)

        

        return

    @staticmethod
    def plot_behavior_learningrates_single():

        assert VALUE_MODEL == 'Baseline'
        # subsessions = ['stable block','volatile block']

        lrs = np.zeros((len(SESSIONS),len(SUBSESSIONS)))
        betas = np.zeros((len(SESSIONS),len(SUBSESSIONS)))
        for ss,subsession in enumerate(SUBSESSIONS):

            print('*'*50)
            print(subsession)
            print('*'*50)

            for s,session in enumerate(SESSIONS):
                start=time.time()
                print(session)
                Behav = ProcessBehavior(session,subsession)
                param_dict = Behav.Q_learning_params
                lrs[s,ss] = param_dict['alpha']
                betas[s,ss] = param_dict['beta']

                utils.get_time_left(start,s,len(SESSIONS))


        fig,axs=plt.subplots(1,2,layout='constrained')
        fig.suptitle(f'All Sessions (n={len(SESSIONS)})')

        #pos lr
        ax=axs[0]
        ax.set_title('Learning Rate')
        ax.boxplot(lrs,tick_labels=SUBSESSIONS)
        ax.set_ylim([-0.1,1.1])
        ax.set_ylabel('Learning Rate')
        ax.tick_params(axis='x', labelrotation=45)

        #beta (inverse temp)
        ax=axs[1]
        ax.set_title('Beta (Inverse Temperature)')
        ax.boxplot(betas,tick_labels=SUBSESSIONS)
        ax.set_ylim([-0.1,20])
        ax.set_ylabel('Beta')
        ax.tick_params(axis='x', labelrotation=45)

        

        return








    



def plot_tuning_curves(fr_df,behavior_df,neuron_type_df,learning_rate_df):
    assert len(fr_df) == len(behavior_df)
    num_trials = len(fr_df)
    
    units = list(fr_df.keys())
    
    if len(units) < 1:
        return None #skip whole thing if there aren't any units for this session
    
    for i,unit in enumerate(units): #loop thru all units of session
    
        neuron_type = str(utils.get_unit_data(neuron_type_df,unit+' - all trials','Neuron Type'))
        print(unit)
        print(neuron_type)
        
        if 'Q' not in neuron_type: #only do for value neurons
            continue
        
        if 'Unit A' in unit: #exclude vmPFC
            continue
        
        #get trials relevant to neuron type
        if neuron_type == 'Qchosen':
            trials = list(range(num_trials))
        elif neuron_type == 'Q1':
            trials = list(behavior_df.loc[(behavior_df['Choice1']==1)].index)
        elif neuron_type == 'Q2':
            trials = list(behavior_df.loc[(behavior_df['Choice2']==1)].index)
         
        #catch edge case problems since we are dealing with t+1
        if trials[-1] >= num_trials-1:
            trials = trials[:-1]

        #get rpe of trials where he chose the relevant target and the change in fr on the very next trial
        delta_fr = np.diff(fr_df[unit])[trials] #delta_fr = fr(t+1) - fr(t). Note fr is during the hold center (valuation) period
        rpe = behavior_df['RPE'][trials] #rpe(t) = r(t) - Q(t)
        
        #color by optimism (aka winbias)
        pos_lr = utils.get_unit_data(learning_rate_df,unit+ ' - all trials',f'Pos Learning Rate for {neuron_type}')
        neg_lr = utils.get_unit_data(learning_rate_df,unit+ ' - all trials',f'Neg Learning Rate for {neuron_type}')
        optimism = pos_lr - neg_lr #ranges from -1 to +1
        if optimism > 0.33: #optimistic
            left_color = 'gray'
            right_color = 'blue'
        elif optimism < -0.33: #pessimistic
            left_color = 'blue'
            right_color = 'gray'
        else: #neutral
            left_color = right_color = 'blue'
        
        #plotit
        fig,ax = plt.subplots(layout="constrained")
        #left half, neg rpe
        color=left_color
        x = rpe[rpe<-0.25]
        y = delta_fr[rpe<-0.25]
        ax.scatter(x,y,color=color,alpha=0.1)
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'-',color=color)
        #right half, pos rpe
        color=right_color
        x = rpe[rpe>=0.25]
        y = delta_fr[rpe>=0.25]
        ax.scatter(x,y,color=color,alpha=0.1)
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'-',color=color)
        
        
        ax.set_title('Optimism tuning curve')
        ax.set_ylabel('Delta FR')
        ax.set_xlabel('RPE')
        
        fig.suptitle(f'{unit}\n{neuron_type} encoding unit')
        plt.show()
    
    
    return



def filter_learning_rates(learning_rate_df,neuron_type_df):
    
    units = neuron_type_df['Unit'].values
    cols = learning_rate_df.columns
    
    for i,unit in enumerate(units):
        
        neuron_type = utils.get_unit_data(neuron_type_df,unit,'Neuron Type')
        
        for col in cols: #loop thru "Learning rate for <regressor>" df columns
            if col.startswith('Learning rate for '):
                
                # if this neuron's type does not include the regressor specified by this column,
                # mark it as such
                if col.removeprefix('Learning rate for ') not in neuron_type: 
                    learning_rate_df.loc[i,col] = 'Does not encode' 
                
    return learning_rate_df    


def filter_dict_by_neurontype(dict_to_filter,neuron_type_df,neuron_type,subsession):
    
    units = neuron_type_df['Unit'].values
    idxs = []
    
    for i,unit in enumerate(units):
        current_neuron_type = utils.get_unit_data(neuron_type_df,unit,'Neuron Type')
        
        if (neuron_type in "".join(current_neuron_type)) and (subsession in unit):
            unit_nosubsess = unit.removesuffix(f' - {subsession}')
            idxs.append(dict_to_filter['unit_labels'].index(unit_nosubsess))
        
    filtered_dict = {} 
    for key,list_ in dict_to_filter.items():
        filtered_dict[key] = [list_[i] for i in idxs]
                
    return filtered_dict


#### Plot Methods #######################################################
 

def barchart_metric_by_rewprob(metric_arr,behavior_df,ax_title,sup_title):
 
    fig,ax=plt.subplots(layout='constrained')

    intergroup_spacing = 3
    intragroup_spacing = 0.8
    colors = ['black','darkgray','dimgray']
    handles =[]        
    for i,stable_or_volatile in enumerate(SUBSESSIONS):
        # print(stable_or_volatile)
        
        for j,rew_prob in enumerate(REW_PROBS):
            # print(rew_prob)
            
            trials = utils.get_trials_rewprob_volatility(behavior_df,stable_or_volatile,rew_prob)
            # print(trials)
            assert trials is not None
            avg,sem = utils.get_avg_sem(metric_arr[trials],axis=0)
            # print(avg)
            
            x = intergroup_spacing * j + intragroup_spacing * i
            ax.bar(x,avg,color=colors[i])
            ax.errorbar(x,avg,yerr=sem,ecolor='k')
            
        patch = mpatches.Patch(color=colors[i], label=stable_or_volatile)
        handles.append(patch)    
        
    ax.legend(handles=handles,loc='lower left')      
    start = intragroup_spacing
    step = intergroup_spacing
    x_ticks = np.arange(start,start + step*len(REW_PROBS),step=step)
    ax.set_xticks(x_ticks,REW_PROBS)
    ax.set_xlabel('Reward Probabilities')
    
    
    ax.set_title(ax_title)
    fig.suptitle(sup_title)
    
    
    
    return


def sliding_win_vs_trials(metric_arr,behavior_df,ax_title,sup_title,num_trials_slide):
        
    block_borders = utils.get_block_borders(behavior_df)
    slide_avg = utils.trial_sliding_avg(metric_arr, num_trials_slide)
    
    fig,ax=plt.subplots(layout='constrained')
    ax.plot(slide_avg)
    ymin,ymax = ax.get_ylim()
    ax.vlines(block_borders,ymin,ymax,color='k',linestyle='dashed')
    ax.set_title(ax_title)
    fig.suptitle(sup_title)
    
    
    return



#### Run Methods #######################################################






def regression_rsqr_table():
    
    df_dict = dict()
    
    all_regs = [
                  'Q1',
                  'Choice1',
                  'Qhigh',
                  'Qchosen',
                  'Qdiff_',
                  'absQdiff',
                  'Choice_high',
                  'Side',
                  'Time'
                  ]
    
    for s,session in enumerate(SESSIONS):
        
        Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
        units = list(Spikes.fr_df.columns)
        behavior_df = Behav.behavior_df
        start = time.time()
        
        for j,unit in enumerate(units):
            fr = Spikes.fr_df[unit]
                     
            # lasso (regularization feature selection)
            score,coefs = utils.regularization_regression(fr,behavior_df[all_regs],'lasso',0.001,1e-4,verbose=False)
            lasso_res = score
            
            # ridge (regularization feature selection)
            score,coefs = utils.regularization_regression(fr,behavior_df[all_regs],'ridge',0.01,1e-4,verbose=False)
            ridge_res = score
            
            # full regressions
            res = utils.linear_encoding_regression(fr,behavior_df[REGRESSORS])
            Q1Qch_res = res.rsquared
            
            res = utils.linear_encoding_regression(fr,behavior_df[['Q1','absQdiff','Choice1','Side','Time']])
            Q1absQdiff_res = res.rsquared
            
            res = utils.linear_encoding_regression(fr,behavior_df[['Q1','Qhigh','Choice1','Side','Time']])
            Q1Qhigh_res = res.rsquared
            
            res = utils.linear_encoding_regression(fr,behavior_df[['Qdiff_','Qchosen','Choice1','Side','Time']])
            QdiffQch_res = res.rsquared
            
            # simple regressions
            simple_reg_res = utils.simple_regression_best_rsqr(all_regs,fr,behavior_df)
                
                
            # Compile results into a dict which will become a row in the final df
            df_dict[unit] = [lasso_res, ridge_res, Q1Qch_res, Q1absQdiff_res, Q1Qhigh_res, QdiffQch_res, simple_reg_res]
        utils.get_time_left(start,s,len(SESSIONS))
    col_names = ['Lasso', 'Ridge', 'Q1 Qchosen', 'Q1 |Qdiff|', 'Q1 Qhigh', 'Qdiff Qchosen', 'Simple']
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=col_names)
    utils.save_csv(df,'RegressionRSqrTable','',DATA_FOLDER)
    
    fig,ax = plt.subplots(layout='constrained')
    ax.bar(col_names,df.mean(axis=0))
    ax.errorbar(col_names,df.mean(axis=0),df.sem(axis=0),capsize=3,ecolor='k',elinewidth=2,capthick=2,barsabove=True,fmt='.')
    data = [df[col_names[i]] for i in range(len(col_names))]
    ax.plot(range(len(col_names)),data,'o',alpha=0.1)
    ax.set_ylabel('Avg R-Squared')
    
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15) #
    utils.save_svg(fig,'Reg_bar','NeuronTypeVenns',PROJ_FOLDER)
    
    return df


def run_value_model_comparison():
    
    aics_list=[]
    for s,session in enumerate(SESSIONS):
        
        hdf_file = os.path.join(DATA_FOLDER, session, session + '.hdf')
        vmc = ValueModelingClass()
        fig, ax1,ax2,ax3,ax4, aic_values, ll_values, acc_values = vmc.plot_model_comparison(hdf_file, num_trials_A=0, num_trials_B=0)
        aics_list.append(aic_values)
    
    
    ## sum aic over sessions
    models = list(aics_list[0].keys())
    tot_aic_dict = {}
    
    for model in models:
        
        aics = []
        for aic_dict in aics_list: #umpack each sess
            
            aics.append(aic_dict[model])
        
        tot_aic_dict[model] = np.sum(aics)
     
        
    ## plot
    fig,ax = plt.subplots(layout='constrained')
    # Sort models by aic (lower is better)
    sorted_models = sorted(tot_aic_dict.items(), key=lambda x: x[1])
    model_names = [key for key,val in sorted_models]
    aic_vals = [aic for _, aic in sorted_models]
    
    # Create bar chart
    bars = ax.barh(range(len(sorted_models)), aic_vals, alpha=0.7)
    
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel('AIC (lower is better)', fontsize=14)
    ax.set_title('Model Comparison (AIC)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Highlight best model (lowest AIC)
    best_idx = 0
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)
    
    # Add AIC values as text
    for i, (model, aic) in enumerate(sorted_models):
        ax.text(aic + (max(aic_vals) - min(aic_vals)) * 0.01, i, 
                f'{aic:.1f}', 
                va='center', fontsize=9)

    return


def lasso_param_gridsearch(lasso_or_ridge='lasso'):
    
    # tols = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
    # reg_strengths = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    # tols = [1e-4]
    tols = [1e-2, 1e-3, 1e-4]
    # reg_strengths = [1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4]
    reg_strengths = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
    ntols = len(tols)
    nregs = len(reg_strengths)
    nsess = len(SESSIONS)
    
    master_regressor_list = ['Q1','Choice1','Qhigh','Qchosen','Qdiff_','absQdiff','Choice_high','Side','Time']
    
    result_matrix = np.zeros((ntols,nregs))

    for s,session in enumerate(SESSIONS):
        Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
        units = list(Spikes.fr_df.columns)
        behavior_df = Behav.behavior_df
        
        start = time.time()
        
        for i,tol in enumerate(tols):
            for j,reg_strength in enumerate(reg_strengths):
                
                scores=[]
                for unit in units:
                    fr = Spikes.fr_df[unit]
                    score,coefs = utils.regularization_regression(fr,behavior_df[master_regressor_list],lasso_or_ridge,reg_strength,tol,verbose=False)
                    scores.append(score)
                result_matrix[i,j] += np.mean(score)
                
        utils.get_time_left(start,s,nsess)               
        
    result_matrix = result_matrix / nsess 
    utils.save_pkl(result_matrix,f'lasso_gridsearch3_{lasso_or_ridge}','',DATA_FOLDER)
    
    fig,ax=plt.subplots(layout='constrained')
    im=ax.matshow(result_matrix)
    ax.set_xlabel('Regularization Strength')
    ax.set_ylabel('Tolerance')
    ax.set_xticks(range(nregs),reg_strengths)
    ax.set_yticks(range(ntols),tols)
    ax.set_title(f'{lasso_or_ridge} hyperparameter grid search')
    c=fig.colorbar(im)
    c.ax.set_ylabel('R^2')
    
    return


def run_value_distribution():
    
    df_list =[]
    for s,session in enumerate(SESSIONS):
            
        Behav = ProcessBehavior(session)
        # Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
        
        df_list.append(Behav.behavior_df)
  
    plot_value_dist(df_list,save_flag=False)
    print(f'len df_list: {len(df_list)}')
    
    return


def run_cumulative_units():
    
    def get_avg_fr(spike_times):
        return len(spike_times)/(spike_times[-1]-spike_times[0])
    
    fig1,axs1 = plt.subplots(1,len(AREAS),layout='constrained') #cumu units
    fig2,ax2 = plt.subplots(layout='constrained') #units per session
    fig3,axs3 = plt.subplots(1,len(AREAS),layout='constrained') #fr hist
    for a,area in enumerate(AREAS):
        print('\n'*3,area)
        unit_cumu_list = []
        unit_counter = 0
        unit_perday_list = []
        avg_fr_list = []
        avg_value_fr_list = []
        value_unit_cumu_list = []
        value_unit_counter = 0
        
        
        for s,session in enumerate(SESSIONS):
            
            print(f'num units: {unit_counter}')
            print(f'num value units: {value_unit_counter}')
            
            unit_dict= utils.load_pkl_2('goodunits_spike_times_dict',session,DATA_FOLDER)
            
            # cumulative num of units
            n_units = utils.count_units_area(unit_dict,area)
            unit_counter += n_units
            unit_cumu_list.append(unit_counter)
            unit_perday_list.append(n_units)
            
            
            # FRs of units
            if n_units > 0:
                area_dict = utils.area_parser(unit_dict,area)
                for unit in range(n_units):
                    avg_unit_fr = get_avg_fr(area_dict['spike_times'][unit])
                    avg_fr_list.append(avg_unit_fr)
            
                # num of value units
                neuron_type_df = utils.area_parser(utils.load_csv('NeuronTypes',session,DATA_FOLDER),area)
                filt_dict = filter_dict_by_neurontype(area_dict,neuron_type_df,'Q','all trials')
                n_value_units = utils.count_units_area(filt_dict,area)
                value_unit_counter += n_value_units
                
                # FRs of value units
                if n_value_units > 0:
                    for unit in range(n_value_units):
                        avg_unit_fr = get_avg_fr(filt_dict['spike_times'][unit])
                        avg_value_fr_list.append(avg_unit_fr)
                    
                    # print(area_dict['unit_labels'])
                    # print('-'*20)
                    # print(filt_dict['unit_labels'])
                    #compare to neuron type df excel file
                    # xxx
  
            value_unit_cumu_list.append(value_unit_counter)
            
            
        # Cumulative
        ax=axs1[a]
        ax.plot(unit_cumu_list,color=COLORS[a],linestyle='-',label='all units')
        ax.plot(value_unit_cumu_list,color='green',linestyle='-',label='value units')
        ax.set_title(f'{area}')
        if a==0:
            ax.set_ylabel('Cumulative number of units')
        else:
            ax.set_yticks([])
        ax.set_xlabel('Recording session')
        ax.legend(loc='upper left')
        
        # Per session
        ax=ax2
        ax.plot(unit_perday_list,color=COLORS[a],linestyle='-',label=f'{area}')
        ax.set_ylabel('Number of units per session')
        ax.set_xlabel('Recording session')
        ax.legend(loc='upper left')
        
        # FR hist
        ax=axs3[a]
        ax.hist(avg_fr_list,range=(0,5),bins=10,color=COLORS[a],label='all units')
        ax.hist(avg_value_fr_list,range=(0,5),bins=10,color='green',label='value units')
        ax.set_title(f'{area}')
        if a==0:
            ax.set_ylabel('Number of units')
        else:
            ax.set_yticks([])
        ax.set_xlabel('Avg FR')
        ax.legend()
        
        
        
    ymax = max([ax.get_ylim()[1] for ax in axs1])
    for a in range(len(AREAS)):
        axs1[a].set_ylim([0,ymax])
    fig1.suptitle('Cumulative Recorded Units')
    fig1.tight_layout()
    
    fig2.suptitle('Recorded Units Per Session')
    fig2.tight_layout()
    
    fig3.suptitle('Avg FR Histograms')
    fig3.tight_layout()
        
    return




def units_totals():
    
    areas = ('Cd','OFC')
    totals = {}
    sortd=[]
    incl=[]
    val=[]
    for area,area_code in zip(areas,['Unit C','Unit D']):
        
        s_counter=0
        i_counter=0
        v_counter=0
        for s,session in enumerate(SESSIONS):
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            neuron_type_df = utils.load_csv('NeuronTypes',session,DATA_FOLDER)
            
            value_all_subsess = neuron_type_df[neuron_type_df['Neuron Type'].str.contains('Q')] #get value encodings
            value_spikes = [s for s in value_all_subsess['Unit'] if 'all trials' in s] #get value encodings over all trials
            
            sorted_spikes = Spikes.unit_labels
            incl_spikes = Spikes.fr_df.columns
            
            # sortd.append(len([sp for sp in sorted_spikes if area_code in sp]))
            # incl.append(len([sp for sp in incl_spikes if area_code in sp]))
            # val.append(len([sp for sp in value_spikes if area_code in sp]))
            s_counter+=len([sp for sp in sorted_spikes if area_code in sp])
            i_counter+=len([sp for sp in incl_spikes if area_code in sp])
            v_counter+=len([sp for sp in value_spikes if area_code in sp])
        
        sortd.append(s_counter)
        incl.append(i_counter)
        val.append(v_counter)
    
    totals={'Sorted': sortd,
            'Included': incl,
            'Value': val}
    
    fig,ax=plt.subplots(layout='constrained')
    res = ax.grouped_bar(totals, tick_labels=areas, group_spacing=1)
    for container in res.bar_containers:
        ax.bar_label(container, padding=3)

    # Add some text for labels, title, etc.
    ax.set_ylabel('Number of units')
    ax.set_title('Unit Totals')
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0,375)
  
    return 

def units_by_ch():
    
    areas = ('Cd','OFC')
    n_chs_per_area=32
    dict_out={}
    for area,area_code in zip(areas,['Unit C','Unit D']):
        
        counter=np.zeros(n_chs_per_area)
        for s,session in enumerate(SESSIONS):
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)

            incl_spikes = Spikes.fr_df.columns
            
            area_spikes = [sp for sp in incl_spikes if area_code in sp]
            
            for ch in range(n_chs_per_area):
            
                if ch < 10:
                    unit_code = f'{area_code}-1-0{str(ch+1)}' #e.g. Unit D-1-09
                elif ch > 9:
                    unit_code = f'{area_code}-1-{str(ch+1)}' #e.g. Unit D-1-25
                counter[ch] += len([sp for sp in area_spikes if unit_code in sp])
                
        
        fig,ax=plt.subplots(layout='constrained')
        ax.bar(np.arange(n_chs_per_area)+1,counter)
        ax.set_title(f'{area} spike count per ch')
        ax.set_ylabel('Num Included Spikes')
        ax.set_xlabel('Ch num')
        ax.set_xticks(np.arange(n_chs_per_area,step=2)+1)
        
        dict_out[f'unit counts per ch, {area}'] = counter
        
        
    utils.save_pkl(dict_out,'UnitCountsByCh','',DATA_FOLDER)    
    return 


#### Works in Progress #######################################################

def run_lasso():
    
    # REGRESSORS = ['Q1','Choice1','Qhigh','Qchosen','Qdiff_','absQdiff','Choice_high','Side','Time']
    
    encodings_dict= {} #over all sessions
    
    for s,session in enumerate([SESSIONS[0]]):
        
        Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
        units = list(Spikes.fr_df.columns)
        behavior = Behav.behavior_df[REGRESSORS]

        
        for j,unit in enumerate(units):
            fr = Spikes.fr_df[unit]
                     
            # print(behavior.shape)
            # print(fr.shape)
            score,coefs = utils.lasso_regression(fr,behavior,verbose = False)
            encoding = utils.lasso_feature_selection(coefs,REGRESSORS)
            encodings_dict[unit] = encoding
            
    neuron_type_df = pd.DataFrame.from_dict(encodings_dict, orient='index', columns=['Encodings'])
    utils.save_csv(neuron_type_df,'lasso_encodings','',DATA_FOLDER)

    
    return neuron_type_df


def behavior_checks():
    
    print('BEHAVIOR CHECKS')
    
    all_trials=[]
    unique_trials=[]
    completed_trials=[]
    pct_rewarded=[]
    side_bias=[]
    rights=[]
    lefts=[]
    for s,session in enumerate(SESSIONS):
        
        print(f'{session}: {s+1}/{len(SESSIONS)}')
        #add in for subsession here
        
        
        #load hdf
        Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
        be = BehaviorAnalysis.ChoiceBehavior_Whitehall([Behav.hdf_file]) #method needs hdf filenames in a list
        
        #get info about trials
        all_ = len(be.ind_center_states)
        unique = len(be.ind_wait_states)
        completed = len(be.ind_check_reward_states)
        hold_pen = len(be.ind_hold_penalty_states)
        timeout = len(be.ind_timeout_penalty_states)
        rewarded = len(be.ind_reward_states)
        pct_rew = rewarded/completed
        side = Behav.behavior_df['Side'] # -1 or 1
        bias = np.mean(side)
        right = sum(np.nonzero(side==-1)[0])
        left = sum(np.nonzero(side==1)[0])
        assert right + left == len(side)
        
        assert abs(completed + hold_pen + timeout - all_) < 2 #making sure number of total trials add up (can be off by 1 since task can end abruptly)
        
        all_trials.append(all_)
        unique_trials.append(unique)
        completed_trials.append(completed)
        pct_rewarded.append(pct_rew)
        side_bias.append(bias)
        rights.append(right)
        lefts.append(left)
        print(f'All trials: {all_}')
        print(f'Unique trials: {unique}')
        print(f'Completed trials: {completed}')
        print(f'Percent completed trials rewarded: {pct_rew}')
        print(f'Side bias: {bias}')
        
        print('-'*30)
        
    # summarize info across sessions using histograms
    fig,axs = plt.subplots(2,2,layout='constrained')
    axs[0,0].hist(all_trials,color='tab:blue')
    axs[0,0].set_title('Num Trials')
    
    # axs[1,0].hist(side_bias,color='tab:orange')
    # axs[1,0].set_title('Side Bias')
    axs[1,0].scatter(lefts,rights,color='tab:orange')
    axs[1,0].set_xlabel('left choices')
    axs[1,0].set_ylabel('right choices')
    axs[1,0].set_title('Left/Right Choices')
    
    axs[0,1].hist(completed_trials,color='tab:green')
    axs[0,1].set_title('Num Completed Trials')
    
    axs[1,1].hist(pct_rewarded,color='tab:red')
    axs[1,1].set_title('Pct Trials Rewarded')
    
    

    fig.suptitle('Behavior Checks - Histograms over all sessions')
            
        

def behavior_checks_2():
    
    behavior_df_list=[]
    for s,session in enumerate(SESSIONS):
        
        start = time.time()
        print(session)
        
        Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
        
        
        #get Rxn time
        be = BehaviorAnalysis.ChoiceBehavior_Whitehall([Behav.hdf_file])
        target_prompt = be.ind_check_reward_states - 3
        hold_begin = be.ind_check_reward_states - 2
        RT = (be.state_time[hold_begin] - be.state_time[target_prompt]) / be.fs_hdf * 1000 #millisec
        
        #get side info
        side = np.array(Behav.behavior_df['Side']) # -1 or 1
        
        # get acc info
        acc=[]
        for trial in range(len(Behav.behavior_df)):
            if Behav.behavior_df['Choice1'].iloc[trial] == 1: #if choice 1
                if Behav.behavior_df['Rew_prob_1'].iloc[trial] > Behav.behavior_df['Rew_prob_2'].iloc[trial]: #if choice 1 actually more valuable
                    acc.append(1)
                else:
                    acc.append(0)
                    
            else: #if choice 2
                if Behav.behavior_df['Rew_prob_2'].iloc[trial] > Behav.behavior_df['Rew_prob_1'].iloc[trial]: #if choice 2 actually more valuable
                    acc.append(1)
                else:
                    acc.append(0)
        
        Behav.behavior_df['Rxn Time'] = RT
        behavior_df_list.append(Behav.behavior_df)
        
        # ## Per session plots
        # #bar charts
        # for metric_arr,metric in zip([side,np.array(acc),RT],['Side bias','Accuracy','Rxn Time']):
        #     print(metric)
        #     barchart_metric_by_rewprob(metric_arr,Behav.behavior_df,ax_title=metric,sup_title=session)
            
        # #sliding avg plots
        # for metric_arr,metric in zip([side,acc],['Side bias','Accuracy']):
        #     sliding_win_vs_trials(metric_arr,Behav.behavior_df,ax_title=metric,sup_title=session,num_trials_slide=10)
        
        # #rxn time vs trials plot
        # sliding_win_vs_trials(RT,Behav.behavior_df,ax_title='Rxn Time (ms)',sup_title=session,num_trials_slide=1)
        
        
        # fig,ax = plt.subplots(layout='constrained')
        # ax.hist(RT,bins=50)
        # ax.set_xlabel('Rxn time (ms)')
        # ax.set_title('Rxn Time')
        # fig.suptitle(session)
        # 
        # xxx
        
        utils.get_time_left(start,s,len(SESSIONS)) 
        
    
    
    #all sessions combined
    combined_df = utils.merge_sessions_df(behavior_df_list)
    
    #get side info
    side = combined_df['Side'] # -1 or 1
    
    # get acc info
    acc=[]
    for trial in range(len(combined_df)):
        if combined_df['Choice1'].iloc[trial] == 1: #if choice 1
            if combined_df['Rew_prob_1'].iloc[trial] > combined_df['Rew_prob_2'].iloc[trial]: #if choice 1 actually more valuable
                acc.append(1)
            else:
                acc.append(0)
                
        else: #if choice 2
            if combined_df['Rew_prob_2'].iloc[trial] > combined_df['Rew_prob_1'].iloc[trial]: #if choice 2 actually more valuable
                acc.append(1)
            else:
                acc.append(0)
    
    # get rxn time
    RT = combined_df['Rxn Time']
    
    #bar charts
    for metric_arr,metric in zip([side,np.array(acc),RT],['Side bias','Accuracy','Rxn Time']):
        barchart_metric_by_rewprob(metric_arr,combined_df,ax_title=metric,sup_title='All Sessions')
        
    return
    
def spike_inclusion_sanity():
    
    session = "braz20260911_03_te2608"
    _ = ProcessSpikes(session,True)
    
    return

class SubsessionVsFR_Analysis:
    
    def __init__(self):
    
        fr_dicts = {} #each item in dict is a list of subsession fr for a unit
        raw_fr_dicts = {} #each item in dict is a list of subsession fr for a unit
        for s, session in enumerate(SESSIONS):
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s, session)
            
            units = Spikes.fr_df.columns
            for unit in units:
            
                fr_dicts[unit] = []
                raw_fr_dicts[unit] = []
                for subsession in SUBSESSIONS:
                    trials = utils.get_trials(Behav.behavior_df,subsession)
                    
                    raw_fr = Spikes.raw_fr_df[unit].iloc[trials] #vector of FRs for subsession trials
                    raw_fr_dicts[unit].append(np.mean(raw_fr))
                    
                    fr = Spikes.fr_df[unit].iloc[trials] #vector of FRs for subsession trials
                    fr_dicts[unit].append(np.mean(fr))
       
        self.plot_paired_frs(fr_dicts,'Z-scored FR')
        self.plot_paired_frs(raw_fr_dicts,'Raw FR')
            
        return
    
    def plot_paired_frs(self,fr_dicts,title):
        print('\n\n')
        print(title)
        fig,ax=plt.subplots(layout='constrained')
        
        units = fr_dicts.keys()
        
        #paired dots
        for unit in units:
            ax.plot(fr_dicts[unit],'o-',alpha=0.2)
            
        # mean and sem
        fr_list=[]
        for i,subsession in enumerate(SUBSESSIONS):
            frs_subsess = [frs[i] for frs in fr_dicts.values()] #itr thru units and get fr for current subsess
            avg,sem = utils.get_avg_sem(frs_subsess,axis=0)
            fr_list.append(frs_subsess)
            
            ax.errorbar(i,avg,yerr=sem,color='k')
            
            print('-'*15)
            print(subsession)
            print(f'Avg +/- sem FR: {avg:.4f} +/- {sem:.4f}')
            
        print('-'*15)
        print('Stable vs Volatile')
        res = sp.stats.ttest_ind(fr_list[1], fr_list[2], equal_var=True)
        print(res)
            
        ax.set_xticks(np.arange(len(SUBSESSIONS)),SUBSESSIONS)
        ax.set_ylabel('Avg FR')
        ax.set_title(f'FR for each unit for each subsession\nn={len(units)}')
        fig.suptitle(title)
            
        
        return



class Behavior_Analysis:
    def __init__(self):
        return

    @staticmethod
    def value_dists():
        df_list = []
        for s, session in enumerate(SESSIONS):
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s, session)

            #for plotting single sess results
            # Behav.plot_values(save_flag=False)
            # Behav.plot_value_dist(save_flag=False) #good for exemplar plots

            df_list.append(Behav.behavior_df)

        print(f'len df_list: {len(df_list)}')
        Behavior_Analysis.plot_value_dist(df_list)

        return
    
    @staticmethod
    def plot_value_dist(behavior_df_list):

        combined_df = utils.merge_sessions_df(behavior_df_list)

        def get_values_rewprob_volatility(df, stable_or_volatile, rew_prob):
            if 'Q1' in df.columns:
                trials = utils.get_trials_rewprob_volatility(df, stable_or_volatile, rew_prob)
                Q1 = df['Q1'].iloc[trials]
            else:
                Q1 = np.full(len(df), np.nan)

            if 'Q2' in df.columns:
                trials = utils.get_trials_rewprob_volatility(df, stable_or_volatile,
                                                             1 - rew_prob)  # do 1 minus rew prob since this is for targ 2
                Q2 = df['Q2'].iloc[trials]
            else:
                Q2 = np.full(len(df), np.nan)

            if 'Qchosen' in df.columns:
                trials = utils.get_trials_rewprob_volatility_Qchosen(df, stable_or_volatile,rew_prob)
                Qch = df['Qchosen'].iloc[trials]
            else:
                Qch = np.full(len(df), np.nan)
                
            return np.concatenate([Q1, Q2])
            # return Qch

        ## Histogram plots
        bins = np.linspace(0.0, 1.0, num=20)
        
        def plot_valuedist_hist(rewprobs,rewprob_labels):
            
            fig, ax = plt.subplots(layout='constrained')
            for i, stable_or_volatile in enumerate(SUBSESSIONS[1:]):
                for ii, rew_prob in enumerate(rewprobs):
                    values = get_values_rewprob_volatility(combined_df, stable_or_volatile, rew_prob)
                    ax.hist(values, density=True, bins=bins, histtype='step',
                            label=stable_or_volatile, linestyle=LINESTYLES[i], color=COLORS[ii])
            ax.legend()
            ax.set_ylabel('Density')
            ax.set_xlabel('Subjective Value')
            ax.set_title(f'Target Reward Probabilities: {rewprob_labels}')
            return
        
        plot_valuedist_hist([0.,1.],'0/1') # rewprob = 0/1 plot
        plot_valuedist_hist([0.2,0.8],'0.2/0.8') # rewprob = 0.2/0.8 plot
        plot_valuedist_hist([0.4,0.6],'0.4/0.6') # rewprob = 0.4/0.6 plot
        

        ## Variance Bar plots
        vars_df = {'RewProb': [], 'StableOrVolatile': [], 'Variance': []}
        ylims = 0, 0.1
        colors = ['darkgray', 'dimgray']
        
        def plot_valuedist_var_bar(rewprobs,rewprob_labels):
            fig, ax = plt.subplots(layout='constrained')
            handles = []
            for i, stable_or_volatile in enumerate(SUBSESSIONS[1:]):
                for ii, rew_prob in enumerate(rewprobs):
                    vars_ = []
                    for behavior_df in behavior_df_list:  # loop thru all sessions individually
                        values = get_values_rewprob_volatility(behavior_df, stable_or_volatile, rew_prob)
                        if len(values) > 0:
                            var = np.nanvar(values)
                            vars_.append(var)
                            vars_df['RewProb'].append(rew_prob)
                            vars_df['StableOrVolatile'].append(stable_or_volatile)
                            vars_df['Variance'].append(var)

                    avg, sem = utils.get_avg_sem(vars_, axis=0)
                    # print(vars_)
                    # print(avg)
                    # print(sem)
                    ax.bar(i + (ii * 2.5), avg, label=stable_or_volatile, color=colors[i])
                    ax.errorbar(i + (ii * 2.5), avg, sem, ecolor='k', color='k')
                patch = mpatches.Patch(color=colors[i], label=stable_or_volatile)
                handles.append(patch)

            ax.legend(handles=handles)
            ax.set_ylabel('Value Distribution Variance')
            ax.set_xticks([0.5, 3.0], [f'Rew Prob: {rewprobs[0]}', f'Rew Prob: {rewprobs[1]}'])
            ax.set_title(f'Target Reward Probabilities: {rewprob_labels}\nn={len(vars_)}')
            ax.set_ylim(ylims)
            return

        # rewprob = 0/1 plot
        plot_valuedist_var_bar([0.,1.],'0/1') # rewprob = 0/1 plot
        plot_valuedist_var_bar([0.2,0.8],'0.2/0.8') # rewprob = 0.2/0.8 plot
        plot_valuedist_var_bar([0.4,0.6],'0.4/0.6') # rewprob = 0.4/0.6 plot
        

        ## Stats

        # 2way ANOVAs
        vars_df = pd.DataFrame.from_dict(vars_df)
        vars_df['StableOrVolatile'] = vars_df['StableOrVolatile'].astype('category')

        # print('High Value')
        # hv_df = vars_df.loc[vars_df['RewProb'] > 0.5]
        # model = sm_api.ols('Variance ~ C(RewProb) + C(StableOrVolatile) + C(RewProb):C(StableOrVolatile)',
        #                    data=hv_df).fit()
        # anova_result = sm.stats.anova_lm(model, typ=2)
        # print(anova_result)

        # print('\n\nLow Value')
        # lv_df = vars_df.loc[vars_df['RewProb'] < 0.5]
        # model = sm_api.ols('Variance ~ C(RewProb) + C(StableOrVolatile) + C(RewProb):C(StableOrVolatile)',
        #                    data=lv_df).fit()
        # anova_result = sm.stats.anova_lm(model, typ=2)
        # print(anova_result)

        # T-Tests
        for rew_prob in [0., 0.2, 0.4, 0.6, 0.8, 1.]:
            print(f'\n\nRewProb={rew_prob}  Stable vs Volatile')
            #get data
            stab_df = vars_df.loc[
                (vars_df['RewProb'] == rew_prob) & (vars_df['StableOrVolatile'] == SUBSESSIONS[1])]
            vol_df = vars_df.loc[
                (vars_df['RewProb'] == rew_prob) & (vars_df['StableOrVolatile'] == SUBSESSIONS[2])]
            
            #T-test
            res = sp.stats.ttest_ind(stab_df['Variance'].values, vol_df['Variance'].values, equal_var=True)
            print(res)
            
            
        # ANOVAs    
        for rew_probs in [(0.,1.), (0.2,0.8), (0.4,0.6)]:
            print(f'\n\nRewProbs={rew_probs}  Stable vs Volatile') 
            #get data
            stab_df1 = vars_df.loc[
                (vars_df['RewProb'] == rew_probs[0]) & (vars_df['StableOrVolatile'] == SUBSESSIONS[1])]
            vol_df1 = vars_df.loc[
                (vars_df['RewProb'] == rew_probs[0]) & (vars_df['StableOrVolatile'] == SUBSESSIONS[2])]
            
            stab_df2 = vars_df.loc[
                (vars_df['RewProb'] == rew_probs[1]) & (vars_df['StableOrVolatile'] == SUBSESSIONS[1])]
            vol_df2 = vars_df.loc[
                (vars_df['RewProb'] == rew_probs[1]) & (vars_df['StableOrVolatile'] == SUBSESSIONS[2])]
            
            #ANOVA
            model = sm_api.ols('Variance ~ C(RewProb) + C(StableOrVolatile) + C(RewProb):C(StableOrVolatile)',
                               data=pd.concat([stab_df1,vol_df1,stab_df2,vol_df2])).fit()
            anova_result = sm.stats.anova_lm(model, typ=2)
            print(anova_result)

        return

def plot_behavior_new(session_to_plot=''):  
    '''
    session_to_plot parameter is optional to plot only a specific session
    comment/uncomment plotting methods below to choose what to plot
    '''
    for s,session in enumerate(SESSIONS):
        
        #choose which session to do (optional)
        if session_to_plot:
            cond = session == session_to_plot
        else:
            cond = True
            
        if cond:    
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            # Behav.check_all_behavior()
            # Behav.plot_choices_and_rewards(save_flag=False)
            # Behav.plot_values(save_flag=False)
            # Behav.plot_choices(save_flag=False)
            Behav.plot_values_vs_choices_vs_rewprob(num_targs=1,num_trials_to_plot=60,save_flag=False)
            Behav.plot_values_vs_rewards(num_targs=1,num_trials_to_plot=60,save_flag=False)
            Behav.plot_values_vs_choices_vs_rewprob(num_targs=2,num_trials_to_plot=60,save_flag=False)
            Behav.plot_values_vs_rewards(num_targs=2,num_trials_to_plot=60,save_flag=False)
            
    return    

def plot_behavior():
    df_list =[]
    # corr_list=[]
    # aics_list=[]
    # kurt_vol_list=[]
    # kurt_stab_list=[]
    for s,session in enumerate(SESSIONS):
        
        #choose which sessions to do
        # if session == 'braz20260803_03_te2548':
        if 1:
            
            # Behav = ProcessBehavior(session)
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            # Behav.check_all_behavior()
            # Behav.plot_choices_and_rewards(save_flag=False)
            # Behav.plot_values(save_flag=False)
            # Behav.plot_choices(save_flag=False)
            # Behav.plot_values_vs_choices_vs_rewprob(num_targs=1,num_trials_to_plot=60,save_flag=False)
            # Behav.plot_values_vs_rewards(num_targs=1,num_trials_to_plot=60,save_flag=False)
            # Behav.plot_values_vs_choices_vs_rewprob(num_targs=2,num_trials_to_plot=60,save_flag=False)
            # Behav.plot_values_vs_rewards(num_targs=2,num_trials_to_plot=60,save_flag=False)
            
            ##see which q learning model fits the best
            # vmc = ValueModelingClass()
            # fig,ax,aics = vmc.plot_model_comparison(Behav.hdf_file, num_trials_A=0, num_trials_B=0)
            # aics_list.append(aics)
            
            ##re get values using rishi's method
            # Behav.get_behavior(Q_learning=True)
            
            ##analyze correlation of behavior variables
            # list_of_vars = ['Q1','Q2','Choice1','Qhigh','Qlow','Qchosen','Qdiff_','absQdiff','Choice_high','Side','Time']
            # corr = Behav.get_behavior_var_corr(list_of_vars)
            # plot_var_corr(corr,list_of_vars,Behav.session)
            # corr_list.append(corr)
            
            ##analyze distribution of values
            # Behav.plot_values(save_flag=False)
            # Behav.plot_value_dist(save_flag=False) #good for exemplar plots
            # kurt_vol_list.append(kurt_vol)
            # kurt_stab_list.append(kurt_stab)
            # xxx
            df_list.append(Behav.behavior_df)
      
    ##analyze distribution of values
    # df = utils.merge_sessions_df(df_list)
    plot_value_dist(df_list,save_flag=False)
    
    print(f'len df_list: {len(df_list)}')
    
    ##analyze correlation of behavior variables
    # corr_list = np.array(corr_list)
    # plot_var_corr(np.mean(corr_list,axis=0),list_of_vars,'Across Session Average')
    
    #shape aics_list = num_sess len list of dicts
    
    
    # fig,ax = plt.subplots(layout='constrained')
    # ax.bar(['volatile','stable'],[np.mean(kurt_vol_list),np.mean(kurt_stab_list)])
    # ax.scatter(1*np.ones(len(kurt_vol_list)),kurt_vol_list)
    # ax.scatter(2*np.ones(len(kurt_vol_list)),kurt_stab_list)
    
    
    
    return


#### Load Data

def load_data(overwrite_spikes_flag=False,overwrite_lfp_flag=False,verbose=False):
    
    sp_list=[]
    be_list=[]
    lfp_list=[]
    for s,session in enumerate(SESSIONS):
        
        start = time.time()
        print(session)
        
        
        ## Behavior
        if LOAD_BEHAV:
            Behav = ProcessBehavior(session,'all trials')
            assert np.all(np.isclose(np.unique(Behav.behavior_df['Rew_prob_1']),REW_PROBS)), f'Not all rew probs present! {session} {np.unique(Behav.behavior_df['Rew_prob_1'])}'
            
            #Apply session inclusion criteria
            # assert Behav.num_trials > MIN_NUM_TRIALS, f'Too few trials! Skipping! {session}'
            be_list.append(Behav)                
        else: 
            be_list.append('')
            
            
        ## Spikes
        if LOAD_SPIKES:
            if utils.does_pkl_exist('Spikes', session,DATA_FOLDER) and not overwrite_spikes_flag:  
                Spikes = utils.load_pkl('Spikes', session,DATA_FOLDER)
                
                #Check that params are the same 
                if (Spikes.alignment == ALIGNMENT) and np.all(Spikes.psth_t_vector == T_VECTOR):
                    sp_list.append(Spikes)
                else: #if params are new, overwrite
                    del Spikes
                    Spikes = ProcessSpikes(session, verbose)
                    utils.save_pkl(Spikes, 'Spikes', session,DATA_FOLDER)
                    sp_list.append(Spikes)
                    
            else: #if pkl doesnt exist
                Spikes = ProcessSpikes(session, verbose)
                utils.save_pkl(Spikes, 'Spikes', session,DATA_FOLDER)
                sp_list.append(Spikes)
        else:
            sp_list.append('')
            
            
        
        ## LFP
        if LOAD_LFP:
            if utils.does_pkl_exist('LFP', session,DATA_FOLDER) and not overwrite_lfp_flag:  
                LFP = utils.load_pkl('LFP', session,DATA_FOLDER)
                
                #Check that params are the same 
                if (LFP.alignment == ALIGNMENT) and np.all(LFP.t_vector == T_VECTOR):
                    lfp_list.append(LFP)
                else: #if params are new, overwrite
                    del LFP
                    LFP = ProcessLFP(session, verbose)
                    utils.save_pkl(LFP, 'LFP', session,DATA_FOLDER)
                    lfp_list.append(LFP)
                    
            else: #if pkl doesnt exist
                LFP = ProcessLFP(session, verbose)
                utils.save_pkl(LFP, 'LFP', session,DATA_FOLDER)
                lfp_list.append(LFP)
        else:
            lfp_list.append('')
        
        utils.get_time_left(start,s,len(SESSIONS)) 
    # xxx
    return sp_list,be_list,lfp_list



def get_Spikes_Behav_LFP(s: int, session: str):
    # s is idx counter representing the number session to load
    # session is the str name of the session to load
    
    assert SESSIONS[s] == session
    
    if SPIKES_LIST:
        Spikes = SPIKES_LIST[s]
        Behav = BEHAV_LIST[s]
        LFP = LFP_LIST[s]
    else:
        Spikes = ProcessSpikes(session,verbose=False)
        Behav = ProcessBehavior(session)
        LFP = ProcessLFP(session, verbose=False)
    return Spikes, Behav, LFP

#%% Run!

# SESSIONS = ["airp20251021_02_te2216"] #low units- test sesison
LOAD_LFP = False
LOAD_SPIKES = True
LOAD_BEHAV = True
SPIKES_LIST,BEHAV_LIST,LFP_LIST = load_data(overwrite_spikes_flag=False,overwrite_lfp_flag=False,verbose=False)


# lasso_param_gridsearch(lasso_or_ridge='ridge')

# run_temporal_encodings_LFP()

# run_cumulative_units()
# units_totals()
# units_by_ch()

# plot_behavior()


# Behavior_Analysis.value_dists()

# _ = SubsessionVsFR_Analysis()

# plot_value_dist_unitlevel()

# behavior_checks()
# behavior_checks_2()



# plot_behavior_new("braz20260824_03_te2584") # go to method and comment/uncomment plots you want to do


# DvS = DualVsSingleAlpha_Analysis()
# DvS.plot_rsqrs()

# run_tuning_curves()

# VALUE_MODEL = 'Baseline'
# BehaviorLearningRates.plot_behavior_learningrates_single()
# VALUE_MODEL = 'DualBaseline'
# BehaviorLearningRates.plot_behavior_learningrates_dual()

NT = NeuronType_Analysis(overwrite_flag=True)
#regression_rsqr_table()

# # Calculate Learning Rates
# LR = LearningRate_Analysis(overwrite_flag=True,do_even_num_trials=True
# # Plot LRs
# LR = LearningRate_Analysis(overwrite_flag=False)
# # Subsession order control analyses
# LR = LearningRate_Analysis(overwrite_flag=False,do_volatile_first_only=True)
# LR = LearningRate_Analysis(overwrite_flag=False,do_stable_first_only=True)
# Even number of trials control analysis
# LR = LearningRate_Analysis(overwrite_flag=True,do_even_num_trials=True)

# TemporalAnalysis = Temporal_Analysis()
# sessions = ["airp20251007_02_te2198","airp20250602_03_te2003","airp20260408_03_te2343"] #found by using units_by_session() and argmax
# areas =    ["OFC","vmPFC","Cd"]
# TemporalAnalysis.run_PCA(sessions,areas)
# TemporalAnalysis.run_LFP_vs_PCA(sessions,areas)
# TemporalAnalysis.run_spikes()
# TemporalAnalysis.run_LFP()

# sessions = ["airp20260708_05_te2496","airp20260625_05_te2453","airp20260624_06_te2448"]
# areas = ['OFC','OFC','OFC']
# TemporalAnalysis.run_PCA(sessions,areas)

# ENCODING_WINDOWs = [(0.0,0.2),(0.0,0.4),(0.0,0.6)] #window over which neuron encoding is determined [s]
# for ENCODING_WINDOW in ENCODING_WINDOWs:
#     SPIKES_LIST,BEHAV_LIST,LFP_LIST = load_data(True,False)
#     run_neuron_typing(True)
#     regression_rsqr_table()



# if __name__ == "__main__":
#     run_temporal_encodings_LFP()

# if __name__ == "__main__":
#    LR = LearningRate_Analysis(overwrite_flag=True)

# if __name__ == "__main__":
#    # run_dual_vs_single_alpha_rsqr()
#    plot_dual_vs_single_alpha_rsqr()




s=SPIKES_LIST[0]

