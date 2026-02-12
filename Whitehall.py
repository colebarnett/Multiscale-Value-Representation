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
#i should make a function called save_fig(path,fname) that just does plt.savefig and prints that it's done. would be slightly cleaner and easier.
# te1888 and te1927 have units with 300 Hz FRs?? There's gotta be an error somewhere
# find a better way to do the pie chart counting. There's gotta be a better one. Cheat and google it
# fix pie charts. I odn't think the value pie adds to the number of value units. There's overlap.
# turn pies into venns! Using sets should be easiest i think. subsets to do multipel value codings
# make it so I can switch time_align point more easily
# lfp ch numbers and/or elec ids?? are they consistent across recs? Which set of chs for which area?+
# turn lfp snips out into a df for better labelling ?
# can only do volatile blocks first atm for some things
# make q learning learning rate be fit separately for stable and volatile blocks
# get rid of plotting class
# make verbose tag for all mehtods and classes. shit be printing too mcuh
# change to just import scipy and then call the full call in the code to make it more explainy
# Change    units = Spikes.firing_rate_df['Unit_labels'][0]   to be    list(Spikes.firing_rate_df.columns) and then delete the unit labels col
# make there be an input to get times align to choose the alignment point

#### Define global things

# Paths
import os
import sys
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





import gif
import copy
import pickle
import neurodsp.utils
import neurodsp.spectral
# import neurodsp.timefrequency

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
from matplotlib_venn import venn2,venn3
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian
from scipy.signal import welch, iirnotch, filtfilt
from scipy.stats import zscore,pearsonr,kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor



import DecisionMakingBehavior_Whitehall as BehaviorAnalysis
from RishiValueModeling import ValueModelingClass
from file_info import get_block_info

# import riglib.ripple.pyns.pyns.nsparser
from riglib.blackrock.brpylib import NsxFile
# from riglib.ripple.pyns.pyns.nsparser import ParserFactory
# from riglib.ripple.pyns.pyns.nsexceptions import NeuroshareError, NSReturnTypes
# from riglib.ripple.pyns.pyns.nsentity import AnalogEntity, SegmentEntity, EntityType, EventEntity, NeuralEntity

# os.chdir(NS_FOLDER)
from nsfile import NSFile
    
os.chdir(PROJ_FOLDER)

#%% Main code block

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

SESSIONS = [
            "airp20250919_02_te2177",
            "airp20251015_04_te2206",
            "airp20251016_03_te2209",
            "airp20251020_05_te2214",
            "airp20251021_02_te2216",
            "airp20251023_03_te2219",
            "airp20251028_03_te2226",
            "airp20251029_05_te2231",
            "airp20251030_02_te2233",
            "airp20251104_02_te2242",
            "airp20251111_02_te2250"
            ]

REGRESSORS = [
              'Q1',
              'Choice1',
              # 'Qhigh',
              'Qchosen',
              # 'Qdiff_',
              # 'absQdiff',
              # 'Choice_high',
              'Side',
              'Time'
              ]

# REGRESSORS = ['Q1','Q2','Choice1',
#               'Qlow','Qhigh','Qdiff','absQdiff',
#               'Choice_low','Side','Time']

#Since Q1+Q2=1 and Qlow+Qhigh=1, only one regressor of each pair is needed.
# REGRESSORS = ['Q1','Choice1',
#               'Qhigh','Qdiff_','absQdiff',
#               'Choice_low','Side','Time']

ALPHA_THRESHOLD = 0.05

COLORS = ['red','blue','gold','green','orange','purple']
LINESTYLES = ['-','--',':','-.']
MARKERSTYLES = ['o','^','s','*','d','P','X','p','H','<','>']

# Channel keys
# something here about which chs are which area
# just do ^^ with ^^ good_chans file?

def get_avg_sem(arr,axis):
    avg = np.mean(arr,axis=axis)
    sem = np.std(arr,axis=axis) / np.sqrt(np.shape(arr)[axis])
    return avg,sem

def encoding_regression(spikes,behavior):
    '''
    Parameters
    ----------
    spikes : (1 x num_trials) array of spiking information (psth timebin info or firing rate)
    behavior : (num_regressors x num_trials) array of behavioral information (do not include const. This will be added internally)

    Returns
    -------
    f_pval : float. p-value for entire regression
    pvals : (1 x num_regressors) array of p-values for each regressor
    '''
    model = sm.OLS(spikes, sm.add_constant(behavior,has_constant='add'), missing='raise',hasconst=True)
    # #Variance inflation factor analysis
    # vif = np.array([variance_inflation_factor(behavior, i) for i in range(np.shape(behavior)[0])])
    # vif_threshold = 5
    # if np.sum(vif>vif_threshold) > 0:
    #     idx = np.nonzero(vif>vif_threshold)[0]
    #     print('-'*50)
    #     print('COLLINEARITY WARNING')
    #     # print(f'Regressor(s): {[self.regressors[i] for i in idx]}')
    #     # fig,ax = plt.subplots()
    #     # ax.plot(behavior)
    #     # xxx        
    res = model.fit()
    # print(res.pvalues)
    # pvals = res.pvalues[1:] #exclude constant (position 0)
    # f_pval = res.f_pvalue
    # rsqr = res.rsquared
    return res #f_pval, pvals, rsqr

def encoding_regression_feature_selection(regressor_list,spikes,behavior_df):
    f_pval, pvals, _ = encoding_regression(spikes,behavior_df[regressor_list])
    
    if f_pval < ALPHA_THRESHOLD:
        signif_regressors = [str(feat) for feat in np.array(regressor_list)[pvals < ALPHA_THRESHOLD]]
    else:
        signif_regressors = ['']
        
    return signif_regressors
        

def lasso_regression(spikes,behavior,regularization_strength=1e-5,tol=1e-4, verbose=False):
    '''
    Lasso (L1 regularization) is a technique to perform feature selection by penalizing model weights (coefficients)
    Parameters
    ----------
    spikes : (1 x num_trials) array of spiking information (psth timebin info or firing rate)
    behavior : (num_regressors x num_trials) array of behavioral information (do not include const. This will be added internally)
    regularization_strength : float on interval (0,inf), where 0 would indicate ordinary least squares

    Returns
    -------
    score : float. r-squared score for model fit
    coefs : (num_regressors x 1) parameter vector
    
    '''
    if verbose:
        print('='*15)
        print('Starting Lasso Regression..')
    model = Lasso(alpha=regularization_strength,tol=tol)
    model.fit(behavior, spikes) #no need to add constant since intercept is automatically fit by this method
    score = model.score(behavior, spikes) #r squared
    a=model.get_params()
    print(a.keys())
    xxx
    
    if verbose:
        print('Lasso model fit!\n')
        print(f'Num iterations: {model.n_iter_}')
        print(f'R^2 score: {score}')
        print(f'Num features seen: {model.n_features_in_}')
        if model.feature_names_in_ is not None:
            print(f'Feature names: {model.feature_names_in_}')
            print(f'Lasso fit coefficients: ')
            [print(f'{model.feature_names_in_[i]} : {model.coef_[i]}') for i in range(model.n_features_in_)]
        else:
            print('Feature names not found')
            print(f'Lasso fit coefficients: ')
            [print(f'Feature {i+1} : {model.coef_[i]}') for i in range(model.n_features_in_)]
        print('\n\n')
    return score, model.coef_

def lasso_feature_selection(coefs,regressor_list):
    thresh = 1E-5
    feats = coefs > thresh #bool arr i think
    return str([str(feat) for feat in np.array(regressor_list)[feats]])

def simple_regression_feature_selection(regressor_list,spikes,behavior_df):
    encoding_list = []
    for reg in regressor_list:
        reg_behav = behavior_df[reg]
        f_pval, pval, rsqr = encoding_regression(spikes,reg_behav)
        if f_pval < ALPHA_THRESHOLD:
            encoding_list.append(reg)
            
    if len(encoding_list)<1:
        encoding_list.append('') #in case list is empty
        
    return encoding_list

def simple_regression_plot(regressor,spikes,behavior_df):
    reg_behav = behavior_df[regressor]
    res = encoding_regression(spikes,reg_behav)
    
    fig,ax=plt.subplots()
    ax.plot(reg_behav.values,spikes,'o')
    start,stop = min(reg_behav.values), max(reg_behav.values)
    x = np.linspace(start,stop,50)
    y = res.predict(sm.add_constant(np.transpose(x),has_constant='add'))
    ax.plot(x,y,'k--')
    ax.set_xlabel(regressor)
    ax.set_ylabel('Z-scored FR')
    return

def simple_regression_best_rsqr(regressor_list,spikes,behavior_df):
    rsqr_list = []
    for reg in regressor_list:
        reg_behav = behavior_df[reg]
        f_pval, pval, rsqr = encoding_regression(spikes,reg_behav)
        rsqr_list.append(rsqr)
        
    return max(rsqr_list)
    
def area_parser(df,brain_area):
    
    if 'Unit_labels' in df.columns: #unit=row organization
        match brain_area:
            case 'vmPFC':
                subset_df = df[df['Unit_labels'].str.contains('Unit A')]
            case 'Cd':
                subset_df = df[df['Unit_labels'].str.contains('Unit C')]
            case 'OFC':
                subset_df = df[df['Unit_labels'].str.contains('Unit D')]
            case 'all areas':
                subset_df = df
            case _:
                raise ValueError('Invalid brain area')
                
    else: #unit=col organization
        match brain_area:
            case 'vmPFC':
                subset_df = df.filter(like='Unit A',axis=1)
            case 'Cd':
                subset_df = df.filter(like='Unit C',axis=1)
            case 'OFC':
                subset_df = df.filter(like='Unit D',axis=1)
            case 'all areas':
                subset_df = df
            case _:
                raise ValueError('Invalid brain area')
      
    return subset_df

def calc_percent_encoding(pval_df,alpha_threshold):
    pval_df = pval_df.drop('Unit_labels',axis=1)
    df_signif = pval_df < alpha_threshold
    num_signif = df_signif.sum(axis='index')
    num_units = pval_df.shape[0]
    percent_encoding = num_signif / num_units
    return percent_encoding, num_units

def save_out_csv(df,df_name,session):
    df.to_csv(os.path.join(DATA_FOLDER,session,f'{session}_{df_name}_df.csv'))
    print(f'{session}_{df_name}_df.csv saved!')
    return

def save_out_pkl(obj,obj_name,session):
    with open(os.path.join(DATA_FOLDER,session,f'{session}_{obj_name}.pkl'),'wb') as f:
        pickle.dump(obj,f)
    print(f'{session}_{obj_name}.pkl saved!')
    return

def load_pkl(obj_name,session):
    with open(os.path.join(DATA_FOLDER,session,f'{session}_{obj_name}.pkl'),'rb') as f:
        obj = pickle.load(f)
    print(f'{session}_{obj_name}.pkl loaded.')
    return obj

def does_pkl_exist(obj_name,session):
    return os.path.exists(os.path.join(DATA_FOLDER,session,f'{session}_{obj_name}.pkl'))




class ProcessSpikes:
    '''
    From .mat, .hdf, and .nev files, load spike times and calculate firing rate 
    aligned to task events.
    
    Output: DataFrame with firing rate for each unit for each trial
    '''
    
    def __init__(self, session: str, overwrite_flag=False, verbose=True):

        print('\n')
        print('*'*25)
        print(session)
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session, self.session)
        self.verbose=verbose
        
#         # [Initiate different data files]
#         self.ns2file = None
#         self.hdffile = None
#         self.matfile = None
#         self.pklfile = None
#         self.ns5file = None
#         self.nevfile = None
        
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
        if self.has_pkl and not overwrite_flag:
            self.load_spike_times()
        else:
            self.get_spike_times()
        
        ## Get Times Align
        self.times_align = None
        self.get_times_align()
        
        ## Get Firing Rates and PSTHs
        self.firing_rate_df = None
        self.psth_dict = None
        self.psth_df = None
        self.t_before = 0.1 #how far to look before time_align point [s]
        self.t_after = 0.7 #how far to look after time_align point [s]
        self.t_resolution = 0.1 #time resolution ; size of psth time bins [s]
        self.psth_t_vector = np.round(np.arange(-self.t_before+self.t_resolution, self.t_after, self.t_resolution),decimals=2)
        self.get_firing_rate() 
        self.get_psth()
        
        ## Save Out Processed Data
        # self.df = pd.concat([self.behavior_df,self.firing_rate_df],axis='columns') #merge behavior_df and firing_rate_df
        # self.dict_out = {'Name':session, 'df':self.df}
        self.dict_out = {'Session':session, 'fr_df':self.firing_rate_df, 'psth_dict':self.psth_dict, 'psth_t_vector':self.psth_t_vector}
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
        
        # Save out spike times so we don't need to load them from nev again
        spike_times_dict = {'unit_labels':self.unit_labels, 'spike_times':self.spike_times, 'spike_waveforms': spike_waveforms}
        save_out_pkl(spike_times_dict,'spike_times_dict',self.session)
        
        return


    def load_spike_times(self):
        
        # Load dict of previously saved spike times
        spike_times_dict = load_pkl('spike_times_dict',self.session)
            
        self.unit_labels = spike_times_dict['unit_labels']
        self.spike_times = spike_times_dict['spike_times']
        self.num_units = len(self.unit_labels)
            
        #print out FRs for funsies
        if self.verbose:
            for i,unit_spikes in enumerate(self.spike_times): #loop thru sorted unit
                print(f'{self.unit_labels[i]} - Avg FR: {len(unit_spikes)/(unit_spikes[-1]-unit_spikes[0]):.2f} Hz. ({i+1}/{len(self.unit_labels)})')
        
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
        
        # Find times of successful trials
        ind = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
#         ind_mvmt_period = cb.ind_check_reward_states - 3 #times corresponding to target prompt
#         ind_reward_period = cb.ind_check_reward_states #times corresponding to reward period onset
        # ind_target_hold = cb.ind_check_reward_states - 2 # times corresponding to target hold onset
        
        # align spike tdt times with hold center hdf indices using syncHDF files
        time_align_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind,cb.state_time,self.syncHDF_file)

        # Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
        assert len(time_align_TDT_ind) == len(ind), f'Repeat hold times! Session: {self.session}'
        assert len(time_align_TDT_ind) == self.num_trials

        self.times_align = time_align_TDT_ind / DIO_freq
        
        
        # #Plot to check things out
        # tdt_times = target_hold_TDT_ind / DIO_freq / 60  #convert from samples to seocnds to minutes
        # hdf_times = (cb.state_time[ind_target_hold]) / fs_hdf / 60  #convert from samples to seocnds to minutes
        # fig,ax=plt.subplots()
        # ax.set_title('TDT and HDF clock alignment')
        # ax.set_xlabel('Time of Reward Period on TDT clock (min)')
        # ax.set_ylabel('HDF clock time (min)')
        # ax.plot(tdt_times,hdf_times,'o',alpha=0.5)
        # ax.plot([0,np.max(tdt_times)],[0,np.max(tdt_times)],'k--') #unity line
        # fig.suptitle(self.session)
        # xxx
        print('Alignment loaded!')

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
            
            win_begin = self.times_align[trial] - self.t_before
            win_end = self.times_align[trial] + self.t_after

            for i in range(self.num_units):
                
                unit_spikes = np.array(self.spike_times[i])
                num_spikes = sum( (unit_spikes>win_begin) & (unit_spikes<win_end) )
                
                firing_rates[trial,i] = num_spikes / (self.t_before + self.t_after)
                
        print('Done counting spikes!')

        self.firing_rate_df = pd.DataFrame(zscore(firing_rates,axis=0),columns=self.unit_labels)
        # self.firing_rate_df['Trial'] = np.arange(self.num_trials)+1
        # self.firing_rate_df['Unit_labels'] = [self.unit_labels for i in range(self.num_trials)] 
                
        return 
            
    # def get_spike_times_per_trial(self): #NOT IN USE  tbh i dont remember what htis is for
        
    #     # self.nevfile = NSFile(self.file_prefix + '.nev')
    #     # self.fs = self.nevfile.get_file_data('nev').parser.timestamp_resolution
    #     self.fs = 30000
    #     samps_per_trial = self.fs * (self.t_before + self.t_after)
        
    #     print('Binning spikes..')
    #     spike_times_dataset = np.zeros((self.num_trials,self.num_units,samps_per_trial))
    #     ### WOULD NEED TO REDO SPIKE TIMES TO DO IT PER TRIAL AND NOT PER UNIT. NEED TO KEEP CH INFO
        
        
    #     return
    
    
    
    def get_psth(self):
        '''
        asdfsa

        Returns
        -------
        None.

        '''

        ## Loop thru units and use compute_psth
        print('Getting PSTHs..')
        # psth_length = int(np.rint((self.t_before + self.t_after)/self.t_resolution)) #num_timepoints
        # psths = np.zeros((self.num_units, self.num_trials, psth_length))
        self.psth_dict = dict()
        psth_df_dict = dict()
        for unit,unit_label in enumerate(self.unit_labels):
            
            unit_spikes = np.array(self.spike_times[unit])
            psth = self.compute_psth(unit_spikes) #psth = (num_trials x num_timepoints)
            self.psth_dict[unit_label] = psth
            psth_df_dict[unit_label] = np.ravel(psth)
            
                
        print('Done making PSTHs!')
        
        # self.psth_dict: keys=unit_labels, values=psth arrays (trials x timepoints)
        
        self.psth_df = pd.DataFrame.from_dict(psth_df_dict,orient='columns') #df of size (n_trials * n_timepoints) x n_units
                
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
        psth_length = int(np.rint((self.t_before + self.t_after + extra_t_before + extra_t_after)/self.t_resolution))
        num_timepoints = len(self.times_align)
        psth = np.zeros((num_timepoints, psth_length-1))
        smooth_psth = np.zeros(psth.shape)

        #set smoothing parameters
        # boxcar_length = 5
        # boxcar_window = boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
        b = gaussian(39, 1)

        for i, tp in enumerate(self.times_align):
            data = unit_spikes
            t_window = np.arange(tp - self.t_before - extra_t_before, tp + self.t_after + extra_t_after, self.t_resolution)
            hist, bins = np.histogram(data, bins = t_window)
            hist_fr = hist/self.t_resolution
            psth[i,:] = hist_fr[:psth_length-1]
            #smooth_psth[i,:] = np.convolve(hist_fr[:psth_length-1], boxcar_window,mode='same')/boxcar_length
            smooth_psth[i,:] = convolve1d(hist_fr[:psth_length-1], b/b.sum())
       
        extra_time_bins_before = int(np.rint(extra_t_before / self.t_resolution))
        extra_time_bins_after = int(np.rint(extra_t_after / self.t_resolution))
        
        smooth_psth_out = smooth_psth[:, extra_time_bins_before:-extra_time_bins_after]
        assert len(self.psth_t_vector) == np.shape(smooth_psth_out)[1], print(f'{len(self.psth_t_vector)} vs {np.shape(smooth_psth_out)[1]}')
        
        return smooth_psth_out



class ProcessLFP:
    '''
    From .mat, .hdf, and .ns2 files, load LFP data aligned to task events.
    
    Output: DataFrame with LFP trace for each ch for each trial
    '''
    
    def __init__(self, session: str):

        
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session, self.session)
        
#         # [Initiate different data files]
#         self.ns2file = None
#         self.hdffile = None
#         self.matfile = None
#         self.pklfile = None
#         self.ns5file = None
#         self.nevfile = None
        
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
        self.times_align_th = None #target hold onset (secs)
        self.times_align_cr = None #check reward (secs)
        self.get_times_align_ns2()
        np.save(self.file_prefix+'_times_align_th.npy',self.times_align_th)
        print('SAVED LETS GO')
        
        ## Get LFP snippets for Each Epoch
        #hold center (immediately before target hold times_align)
        times_align, t_before, t_after = self.times_align_th, 1.0, 0.0
        snips = self.get_lfp_snippets(times_align,t_before,t_after)
        self.hc_snips = {'data':snips,'t_before':t_before,'t_after':t_after}
        
        #mvmt period (around target hold times_align)
        times_align, t_before, t_after = self.times_align_th, 0.5, 0.5
        snips = self.get_lfp_snippets(times_align,t_before,t_after)
        self.mp_snips = {'data':snips,'t_before':t_before,'t_after':t_after}
        
        #reward period (immediately after check reward)
        times_align, t_before, t_after = self.times_align_cr, 0.2, 0.8
        snips = self.get_lfp_snippets(times_align,t_before,t_after)
        self.rp_snips = {'data':snips,'t_before':t_before,'t_after':t_after} 
        
        ## Save Out Processed Data
        self.dict_out = {'Session':session, 'hc_snips':self.hc_snips,
                         'mp_snips':self.mp_snips, 'rp_snips':self.rp_snips,
                         'fs':self.fs, 'n_chs':self.n_chs, 'n_trials':self.n_trials}
        
        

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
        
        # Find times of successful trials for desired epoch
        ind_target_hold = cb.ind_check_reward_states - 2 # times corresponding to target hold onset
#         ind_hold_center = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
#         ind_mvmt_period = cb.ind_check_reward_states - 3 #times corresponding to target prompt
        ind_reward_period = cb.ind_check_reward_states #times corresponding to reward period onset
        
        
        # align spike tdt times with hold center hdf indices using syncHDF files
        target_hold_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind_target_hold,cb.state_time,self.syncHDF_file)
        check_reward_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind_reward_period,cb.state_time,self.syncHDF_file)

        # Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
        assert len(target_hold_TDT_ind) == len(ind_target_hold), f'Repeat hold times! Session: {self.session}'
        assert len(target_hold_TDT_ind) == self.n_trials

        self.times_align_th = target_hold_TDT_ind / DIO_freq #convert from samples (30k sampling rate btw) to seconds
        self.times_align_cr = check_reward_TDT_ind / DIO_freq
        
        
        # #Plot to check things out
        # tdt_times = target_hold_TDT_ind / DIO_freq / 60  #convert from samples to seocnds to minutes
        # hdf_times = (cb.state_time[ind_target_hold]) / fs_hdf / 60  #convert from samples to seocnds to minutes
        # fig,ax=plt.subplots()
        # ax.set_title('TDT and HDF clock alignment')
        # ax.set_xlabel('Time of Reward Period on TDT clock (min)')
        # ax.set_ylabel('HDF clock time (min)')
        # ax.plot(tdt_times,hdf_times,'o',alpha=0.5)
        # ax.plot([0,np.max(tdt_times)],[0,np.max(tdt_times)],'k--') #unity line
        # fig.suptitle(self.session)
        # xxx
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
        #convert from secs to samples
        t_before = np.round(self.fs * t_before).astype(int)
        t_after = np.round(self.fs * t_after).astype(int)
        times_align = np.round(self.fs * times_align).astype(int)
        
        snippets = np.zeros((len(times_align), self.n_chs, t_before+t_after)) # (trials x chs x samples)

        #Loop thru trials
        for trial, time_align in enumerate(times_align):
        
            #Get window of samples for current trial (all units are now samples)
            time_begin = time_align - t_before
            time_end = time_align + t_after
            
            #Get time-aligned snippet for current trial from LFP data array
            snippets[trial,:,:] = self.data[:,time_begin:time_end]
            
            #filter out line noise
            snippets[trial,:,:] = self.filter_line_noise(snippets[trial,:,:])
               
        return snippets # (trials x chs x samples)


    def filter_line_noise(self,sig):
        
        #set up to notch filter out 60,120,180 Hz line noise
        Q=30 #quality factor
        b60,a60 = iirnotch(60.,Q,self.fs) #this is done after downsampling would be done
        b120,a120 = iirnotch(120.,Q,self.fs)
        b180,a180 = iirnotch(180.,Q,self.fs)
        
        # Filter line noise    
        sig = filtfilt(b60,a60,sig) #filter out 60hz noise
        sig = filtfilt(b60,a60,sig) #filter out 60hz noise, again
        sig = filtfilt(b120,a120,sig) #filter out 120hz noise
        sig = filtfilt(b180,a180,sig) #filter out 180hz noise
        # snippets = signal.decimate(snippets,q) #downsample down to 500Hz. This function applies anti-aliasing filters.
        
        return sig

        

class ExploreLFP():
    '''
    Get LFP snippets from .mat, .hdf, and .ns2 files using ProcessLFP and then
    plot various attributes of that LFP in order to check it out.
    '''
    
    def __init__(self, session: str):
        
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session, self.session)
        
        ## Get LFP Snippets
        self.snips_dict = ProcessLFP(self.session).dict_out
        self.fs = self.snips_dict['fs']
        self.n_chs = self.snips_dict['n_chs']
        self.n_trials = self.snips_dict['n_trials']
        

        ## PSD Calculation Params
        # self.f = neurodsp.utils.data.create_freqs(4.,200.,2.) #4-200Hz w 2Hz resolution
        # self.n_cycles = self.f/4
        
        ## Loop thru Each Set of Snippets
        i=0
        for key in self.snips_dict.keys():
            if 'snips' in key: #do for all the sets of snippets
                self.color = COLORS[i]
                i+=1
                print(key)
            
        ## Plot some LFP
                n_chs_to_plot = 12
                self.ylim = 400 #(uV)
                self.plot_raw_LFP_chs(self.snips_dict[key],n_chs_to_plot,title=key)
            
        ## Calculate PSD
                f,psds = self.calc_psd(self.snips_dict[key]['data'])
                
        ## Plot PSD by Trials and Channels
                self.plot_psd_each_ch(f,psds,title=key)
                self.plot_psd_each_tr(f,psds,title=key)
        
    
    def plot_raw_LFP_chs(self,snippets,n_chs_to_plot,title):
        
        print('Plotting raw snippets..')
        
        chs_to_plot = np.linspace(0,self.n_chs-1, num=n_chs_to_plot, dtype=int)
        for ch in chs_to_plot:
            self.plot_raw_LFP(snippets,ch,title)
        
        return
    
    
    def plot_raw_LFP(self,snippets,ch,title):
        
        n_trs_to_plot = 10
        trs_to_plot = np.linspace(0,self.n_trials-1, num=n_trs_to_plot, dtype=int)
        
        #plot 
        fig,axs=plt.subplots(5,2)
        fig.suptitle(f'Raw LFP for {title}\nCh {ch}')
        row=0
        col=0
        for tr in trs_to_plot:
            ax=axs[row,col]
            snip = snippets['data'][tr,ch,:]
            ax.plot(snip,self.color)
            ax.set_ylim([-self.ylim,self.ylim])
            ax.set_xlabel('Time (sec)')
            num_ticks = 5
            ax.set_xticks(np.linspace(0,len(snip),num_ticks),
                          np.round(np.linspace(-1*snippets['t_before'],snippets['t_after'],num_ticks),1))
            ax.set_title(f'Trial {tr}')
            if row==4: ax.set_xlabel('Freq (Hz)')
            if col==0: ax.set_ylabel('uV')
            if col==1: ax.set_yticks([])
            
            #step/reset counters
            if row==4:
                row=0
                col+=1
            else:
                row+=1

        return
    
    
    def calc_psd(self,snippets):
        
        print('Calculating PSD..')
        
        first_run = True
        
        for trial in range(self.n_trials):
            for ch in range(self.n_chs):
                
                # f, psd = neurodsp.spectral.compute_spectrum(snippets[trial,ch,:], 
                #         self.fs, method='wavelet',freqs=self.f,n_cycles=self.n_cycles)
                # f, psd = neurodsp.spectral.trim_spectrum(f,psd,[4.,200.])
                
                f, psd = welch(snippets[trial,ch,:], self.fs)
                
                #trim off freqs above 150 Hz
                f_trim = (f<150)
                f = f[f_trim]
                psd = psd[f_trim]
                
                #preallocate
                if first_run:
                    psds = np.zeros((self.n_trials,self.n_chs,len(f)))
                    self.f = f
                    first_run = False
                    
                assert np.all(f == self.f)
                
                psds[trial,ch,:] = psd
                
            if (trial+1)%100 == 0:
                print(f'\t{trial+1}/{self.n_trials} trials done.')
        
        return f,psds    # (trials x chs x f)
    
            
    def plot_psd_each_ch(self,f,psds,title):
        
        print('Plotting PSD for each ch..')
        
        #avg over trials (0th axis)
        avg,sem = get_avg_sem(psds,axis=0)
        
        #plot
        fig,ax=plt.subplots()
        ax.plot(f,avg.transpose(),self.color,alpha=0.5)
        # ax.fill_between(f,avg-sem,avg+sem,self.color,alpha=0.5)
        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title('PSD for each ch (avgd over trials)')
        fig.suptitle(title)
        
        return


    def plot_psd_each_tr(self,f,psds,title):
        
        print('Plotting PSD for each trial..')
        
        #avg over chs (1st axis)
        avg,sem = get_avg_sem(psds,axis=1)
        
        #plot
        fig,ax=plt.subplots()
        # for 
        ax.plot(f,avg.transpose(),self.color,alpha=0.5)
        # ax.fill_between(f,avg-sem,avg+sem,self.color,alpha=0.5)
        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title('PSD for each trials (avgd over chs)')
        fig.suptitle(title)
        
        return        


class ProcessBehavior:
    '''
    From .hdf file, load task and behavior details and calculate value with
    either Q-Learning or empirical-based methods.
    
    Output: DataFrame with behavioral details for regression.
    '''
    
    def __init__(self, session: str):
        

        
        self.session = session
        self.file_prefix = os.path.join(DATA_FOLDER, self.session, self.session)
        
#         # [Initiate different data files]
#         self.ns2file = None
#         self.hdffile = None
#         self.matfile = None
#         self.pklfile = None
#         self.ns5file = None
#         self.nevfile = None
        
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
        self.dict_out = {'Session':session, 'df':self.behavior_df}
        
        

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


        
        
        
        
    def get_behavior(self,Q_learning=True,learning_rate=None):
        '''
        Method which takes the hdf files for a session and computes key behavioral metrics for each 
        trial, such as value of each target and rpe. These along with information about each
        trial, such as whether it was free/forced and which target was chosen, are saved
        into a matrix which is ready to be plugged into a regression model to find what
        neurons are encoding throughout the course of a session.

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
        choices,rewards = behavior.GetChoicesAndRewards()
        Q1_choices = np.zeros_like(choices)
        Q1_choices[choices==1] = 1.0 #1 if he chose Q1, 0 otherwise
        Q2_choices = np.zeros_like(choices)
        Q2_choices[choices==2] = 1.0 #1 if he chose Q2, 0 otherwise
        
        # get block info (stable or volatile)
        if behavior.is_stable_block is not None:
            is_stable_block = behavior.is_stable_block
            is_volatile_block = behavior.is_volatile_block
        else:
            is_stable_block, is_volatile_block = get_block_info(self.session)
            is_stable_block, is_volatile_block = is_stable_block[:len(choices)], is_volatile_block[:len(choices)] #trim to actual num of trials
            
        
        # make time regressor 
        time = np.arange(len(choices))
        time = np.log(time+1)
        time -= np.mean(time) # to make centered at 0
        
        # Calculate value of targets throughout trials
        if Q_learning and learning_rate: #to do a specific learning rate
            [Q1,Q2] = BehaviorAnalysis.CalcValue_2Targs_QLearning(choices,rewards,learning_rate)
        elif Q_learning and not learning_rate: #do Rishi's best Q-learning method
            vmc = ValueModelingClass()
            value_dict = vmc.get_values(self.hdf_file, num_trials_A=0, num_trials_B=0, method='Kernel')
            [Q1,Q2] = [value_dict['Q_low'],value_dict['Q_high']]
        else:
            [Q1,Q2], _ = behavior.CalcValue_2Targs(choices,rewards,win_sz=10,smooth=True)
            
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
                    Qlow_choices[trial] = 1.0
                
                #if Q2 chosen, then Qhigh chosen
                else:                       
                    Qhigh_choices[trial] = 1.0
                    
            #Q1 = higher value -> Qlow=Q2 and Qhigh=Q1         
            else:
                Qlow[trial] = Q2[trial]
                Qhigh[trial] = Q1[trial]
                
                #if Q1 chosen, then Qhigh chosen
                if Q1_choices[trial] == 1: 
                    Qhigh_choices[trial] = 1.0
                
                #if Q2 chosen, then Qlow chosen
                else:                       
                    Qlow_choices[trial] = 1.0
            
        
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
                
            
            
            
            
            
            
        # Get which side (left or right) each choice was
        choice_side = behavior.GetTargetSideSelection()
        choice_side = (choice_side-0.5)*2 #make to be -1 or 1
        
#         # get which center holds were stimulated
#         stim_holds= (choices==1) * (instructed_or_freechoice==1) #get forced LV holds
#         stim_holds[:100] = False #only want blB and blAp holds
#         #print(stim_holds)
#         for i,el in enumerate(stim_holds):
#             if el:
#                 stim_holds[i]=1
#             else:
#                 stim_holds[i]=0


        assert len(choices) == len(choice_side) == len(time) == len(Q1) == len(Q2) == len(rpe) == len(is_stable_block)       
        
        self.behavior_df = pd.DataFrame.from_dict(
            {'Q1':Q1,       'Q2':Q2,        'Choice1':Q1_choices,     'Choice2':Q2_choices, 
             'Qlow':Qlow,   'Qhigh':Qhigh,  'Qdiff_':Qdiff,  'absQdiff':absQdiff,
             'Qchosen':Q_chosen,            'Qunchosen':Q_unchosen,
             'Choice_low':Qlow_choices,  'Choice_high':Qhigh_choices,
             'Reward':rewards, 'RPE':rpe, 'Side':choice_side, 'Time':time,
             'Stable':is_stable_block, 'Volatile':is_volatile_block}) #, 'Stim':stim_holds})
        
        self.num_trials = len(choices)
        
        if not Q_learning:
            print('Behavior loaded!')

        
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
        
        fig,axs=plt.subplots(2,1)
        
        #Q1 vs Q2
        ax=axs[0]
        ax.plot(self.behavior_df['Q1'],color='red',label='Q1')
        ax.plot(self.behavior_df['Q2'],color='blue',label='Q2')
        # ax.legend()
        
        #Qlow vs Qhigh
        ax=axs[1]
        ax.plot(self.behavior_df['Qlow'],color='green',label='Qlow')
        ax.plot(self.behavior_df['Qhigh'],color='gold',label='Qhigh')
        # ax.legend()
        
        fig.suptitle(self.session)
        
        return
        
    
    def plot_value_dist(self,save_flag=False):
        
        df = self.behavior_df
        Q1_vol = df['Q1'].loc[df['Volatile']==1]
        Q1_stab = df['Q1'].loc[df['Stable']==1]
        # Q2_vol = df['Q2'].loc[df['Volatile']==1]
        # Q2_stab = df['Q2'].loc[df['Stable']==1]
        
        bins=np.linspace(0.0,1.0,num=15)
        
        fig,ax=plt.subplots()
        
        ax.hist(Q1_vol,density=True,bins=bins,label='Volatile',color='red',alpha=0.5)
        ax.hist(Q1_stab,density=True,bins=bins,label='Stable',color='blue',alpha=0.5)
        ax.legend()
        ax.set_title('Actual Value Distributions')
        fig.suptitle(self.session)
        
        #flip around to make bimodal dist look normal
        Q1_vol=np.array(Q1_vol)
        Q1_vol[Q1_vol>0.5] = 1.5-Q1_vol[Q1_vol>0.5]
        Q1_vol[Q1_vol<0.5] = 0.5-Q1_vol[Q1_vol<0.5]
        Q1_stab=np.array(Q1_stab)
        Q1_stab[Q1_stab>0.5] = 1.5-Q1_stab[Q1_stab>0.5]
        Q1_stab[Q1_stab<0.5] = 0.5-Q1_stab[Q1_stab<0.5]
        
        fig,ax=plt.subplots()
        
        ax.hist(Q1_vol,density=True,bins=bins,label='Volatile',color='red',alpha=0.5)
        ax.hist(Q1_stab,density=True,bins=bins,label='Stable',color='blue',alpha=0.5)
        ax.legend()
        ax.set_title('Flipped Value Distributions')
        fig.suptitle(self.session)
        
        #Q1
        # ax.hist([Q1_vol,Q1_stab], histtype='bar', stacked=True, density=True, label=['Volatile','Stable'])
        # ax.legend()
        # fig.suptitle(self.session)
        
        # #Q2
        # ax=axs[1]
        # ax.hist(np.array(Q2_vol,Q2_stab),label=['Volatile','Stable'])
        # ax.legend()
        
        
        
        if save_flag:
            fname=f'{self.file_prefix}_value_dist.svg'
            plt.savefig(fname,format='svg')
            print(f'{fname} saved!')
        
        
        return
    
    
    def plot_choices_and_rewards(self,save_flag=False):
            
        color1 = 'gold'
        color2 = 'blue'
        
        ## choice behavior and rewards figure for Fig 1
        #smooth choice data 
        window_length=50
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
#         P_MV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_M'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)

        fig,ax = plt.subplots()

        ax.plot(P_LV,color=color1,label='Target 1')
#         ax.plot(P_MV,color='orange',label='MV Target')
        ax.plot(P_HV,color=color2,label='Target 2')

        #Get reward data for each choice
        LV_rew = np.nonzero((self.behavior_df['Choice1']==1) & (self.behavior_df['Reward']==1))
        LV_unrew = np.nonzero((self.behavior_df['Choice1']==1) & (self.behavior_df['Reward']==0))

#         MV_rew = np.nonzero((behavior_dict['Choice_M']==1) & (behavior_dict['Rewarded']==1))
#         MV_unrew = np.nonzero((behavior_dict['Choice_M']==1) & (behavior_dict['Rewarded']==0))

        HV_rew = np.nonzero((self.behavior_df['Choice1']!=1) & (self.behavior_df['Reward']==1))
        HV_unrew = np.nonzero((self.behavior_df['Choice1']!=1) & (self.behavior_df['Reward']==0))

        ymin_hi, ymax_hi = 1.1, 1.2 #for rewarded trials
        ymin_lo, ymax_lo = -0.2, -0.1 #for unrewarded trials

        ax.vlines(LV_rew,ymin_hi,ymax_hi,color=color1)
        ax.vlines(LV_unrew,ymin_lo,ymax_lo,color=color1)
#         ax.vlines(MV_rew,ymin_hi,ymax_hi,color='orange')
#         ax.vlines(MV_unrew,ymin_lo,ymax_lo,color='orange')
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
        fig.tight_layout()
        
        if save_flag:
            fname=f'{self.file_prefix}_choice_rew_behav.svg'
            plt.savefig(fname,format='svg')
            print(f'{fname} saved!')
        
        return    
    
    def plot_choices(self,save_flag=False):
        
        color1 = 'gold'
        color2 = 'blue'
        
        ## choice behavior and rewards figure for Fig 1
        #smooth choice data 
        window_length=50
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
#         P_MV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_M'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)

        fig,ax = plt.subplots()

        ax.plot(P_LV,color=color1,label='Target 1')
#         ax.plot(P_MV,color='orange',label='MV Target')
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
        fig.tight_layout()
        
        if save_flag:
            fname=f'{self.file_prefix}_choice_behav.svg'
            plt.savefig(fname,format='svg')
            print(f'{fname} saved!')
            
        
        
        
        return 
    
    
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
                res = pearsonr(ind_vars[i,:],ind_vars[j,:])
                corr_matrix[i,j] = res.statistic
                
        corr_matrix = np.multiply(corr_matrix,corr_matrix) #make it r squared
        
        return corr_matrix
        
 

        
# def merge_firingrate_behavior_dfs(session:str, firing_rate_df:pd.DataFrame, behavior_df:pd.DataFrame):
   
#     ## Save Out Processed Data
#     df = pd.concat([behavior_df,firing_rate_df],axis='columns') #merge behavior_df and firing_rate_df
#     processed_data_dict = {'Name':session, 'df':df}
    
#     return processed_data_dict  
        
        

        
        
        
class NeuronTypeRegressions:
    
    def __init__(self,spike_data_dict,behavior_data_dict):
        
        ## Get Spike and Behavior Data
        assert spike_data_dict['Session'] == behavior_data_dict['Session'], 'Sessions do not match!'
        self.session = spike_data_dict['Session']
        self.spike_df = spike_data_dict['fr_df']
        self.behavior_df = behavior_data_dict['df']
        
        self.alpha_threshold = 0.05

        ## Regress Spiking Activity against Behavior to Determing Neuron Type
        self.neuron_type_df = None
        self.neuron_type_regression()
        
        ## Save Out
        self.dict_out = {'Session':self.session, 'df':self.neuron_type_df}
        
        return
    
        
    def neuron_type_regression(self):
        
        print('Performing neuron type regression..')

        counter=0
        encodings = []
        unit_subsession_labeled = []
        for ii,subsession in enumerate(SUBSESSIONS):
            
            assert len(self.behavior_df) == len(self.spike_df), 'Number of trials do not match!'
            num_trials = len(self.spike_df)
            
            trials = get_stable_volatile_trials(self.session)[subsession] #get trials for specified subsession
            trials = trials[trials<num_trials] #trim to match actual session length
            
            if not len(trials) > 1: #if no trials for current subsession
                print(f'No trials for {subsession}. Skipping!')
                continue
                
            else:        
                        
                # Make array of behavioral data
                # regressor_labels = ['Choice1','Side','Q1','Q2','Time']
                # regressor_labels = [col for col in self.behavior_df.columns if not col.startswith("Unit")]
                regressor_matrix = self.behavior_df[REGRESSORS].iloc[trials]
                
                
                units = self.spike_df['Unit_labels'][0]
                # Regress FR of each unit against behavior
                for i,unit in enumerate(units): #loop thru all units of session
                
                    FR = self.spike_df[unit].iloc[trials] #vector of FRs for each trial
                    
                    assert len(FR)>1
                    assert len(regressor_matrix)>1
        
                    f_pval,pvals = encoding_regression(FR,regressor_matrix)
                    
                    # print(res.f_pvalue)
                    if f_pval < self.alpha_threshold: #if regression is statistically significant
                        signif_regressors_bools = pvals<self.alpha_threshold #get which regressors were statistically signficant
                        signif_regressors = [reg for indx,reg in enumerate(regressor_matrix.cols) if signif_regressors_bools[indx]] #get names of those regressors
                        encodings.append(signif_regressors) 
                        
        #                 signif_regressors_posneg = res.params[signif_regressors_bools] #get the sign of the coefficient of the significant regressors
        #                 encoding_posneg[chann][unit] = signif_regressors_posneg
        #                 #within nested dict, unit number is the key and the significant regressors are the values
                    
                    else: #if regression isn't statistically significant, there's no encoding
                        encodings.append(['Non-encoding'])
                    
                    unit_subsession_labeled.append(f'{unit} {subsession}')
                
                    # print(f'{unit_subsession_labeled[counter]} done. Neuron type: {encodings[counter]}. ({counter+1}/{len(units)*len(SUBSESSIONS)})')
                
                    counter+=1
      
        self.neuron_type_df = pd.DataFrame.from_dict({'Unit':unit_subsession_labeled, 'Neuron Type':encodings})
        print('Neuron type regressions done!')

        return
    
    
      
    


        
class TemporalEncodingRegressions:
    
    def __init__(self,spike_data_dict,behavior_data_dict,overwrite_flag):
        
        self.obj_str = 'temporal_encoding_pval_dict'
        
        ## Get Spike and Behavior Data
        assert spike_data_dict['Session'] == behavior_data_dict['Session'], 'Sessions do not match!'
        self.session = spike_data_dict['Session']
        self.psth_dict = spike_data_dict['psth_dict'] #keys=unit_labels, values=psth arrays (trials x timepoints)
        self.psth_t_vector = spike_data_dict['psth_t_vector']
        self.behavior_df = behavior_data_dict['df']
        
        self.alpha_threshold = 0.05

        ## Regress Spiking Activity against Behavior to Determing Encoding for each Regressor
        # self.regressors = ['Q1','Qhigh','Qdiff_','Choice1']
        self.areas = ['OFC','vmPFC','Cd']
        self.pval_dict = dict()
        
        self.has_pval_dict = does_pkl_exist(self.obj_str,self.session)
        
        if self.has_pval_dict and not overwrite_flag:
            self.pval_dict = load_pkl(self.obj_str,self.session)
        else:
            self.temporal_encoding_regression()
            save_out_pkl(self.pval_dict,self.obj_str,self.session)
            
        self.plot_temporal_encoding() 
        
        return
    
    

    
    
    def temporal_encoding_regression(self):
        
        print('Performing temporal encoding regression..')

        counter=0
        
        units = list(self.psth_dict.keys())
        num_units = len(units)
        num_timebins = self.psth_dict[units[0]].shape[1]
        num_trials = len(self.behavior_df)
        trials = np.arange(num_trials)
  
                
        # Make array of behavioral data
        # regressor_labels = [col for col in self.behavior_df.columns if not col.startswith("Unit")]
        # regressor_matrix = self.behavior_df[REGRESSORS].iloc[trials]
        # regressor_labels = ['Choice1','Side','Q1','Q2','Time']
        regressor_matrix = self.behavior_df[REGRESSORS].iloc[trials]

        
        
        ## Regress FR of each unit against behavior for each timebin
        for j,regressor in enumerate(REGRESSORS):
            temporal_pvals = np.zeros((num_units,num_timebins))
        
            for i,unit in enumerate(units): #loop thru all units of session
                psth = self.psth_dict[unit]
                
                
                for k in range(num_timebins): #loop thru timebins
                    timebin_data = psth[:,k] #timebin k for all trials
                    
                    _, pvals = encoding_regression(timebin_data,regressor_matrix)
                    temporal_pvals[i,k] = pvals[REGRESSORS.index(regressor)]
    
    
                # print(f'{i+1}/{num_units}')
  
    
                ## Save out
                timebin_labels = [f'Timebin_{t}' for t in range(num_timebins)]
                pval_df = pd.DataFrame(temporal_pvals,columns=timebin_labels)
                pval_df['Unit_labels'] = units
                self.pval_dict[f'{regressor}_pval_df'] = pval_df
        
        print('Temporal encoding regressions done!')

        return
    

        
    
    def plot_temporal_encoding(self):
        
        fig,ax = plt.subplots()
        
        num_units_list = []
        for i,area in enumerate(self.areas):
            for j,regressor in enumerate(REGRESSORS):
                
                # get pval_dict for current regressor
                df_list = list(self.pval_dict.keys())
                regressor_index = [i for i in range(len(df_list)) if regressor in df_list[i]][0]
                pval_df = self.pval_dict[df_list[regressor_index]]
                
                # get % encoding for current area 
                pval_df = area_parser(pval_df,area)
                percent_encoding, num_units = calc_percent_encoding(pval_df,self.alpha_threshold)
                if j==0:
                    num_units_list.append(num_units)

                #plot                    
                ax.plot(percent_encoding,color=COLORS[i],linestyle=LINESTYLES[j])
                
        # Plot details
        ax.set_ylabel('% Units Encoding')
        ax.set_xlabel('Time relative to Choice')
        ax.set_xticks(np.arange(len(percent_encoding)),self.psth_t_vector)
        ax.set_title('Temporal Encoding Dynamics')
        fig.suptitle(self.session)
        
        # legend
        custom_lines = [Line2D([0], [0], color=COLORS[0]),
                        Line2D([0], [0], color=COLORS[1]),
                        Line2D([0], [0], color=COLORS[2]),
                        Line2D([0], [0], color='gray', linestyle=LINESTYLES[0]),
                        Line2D([0], [0], color='gray', linestyle=LINESTYLES[1]),
                        Line2D([0], [0], color='gray', linestyle=LINESTYLES[2]),
                        Line2D([0], [0], color='gray', linestyle=LINESTYLES[3])]
        
        area_labels = [f'{self.areas[i]}, {num_units_list[i]} units' for i in range(len(self.areas))]
        ax.legend(custom_lines, area_labels + REGRESSORS,ncols=4)
        
        return
        
    
      
 

     
            
class NeuronTypePieCharts:
    
    def __init__(self, sessions: list, save_flag: bool):
        
        dfs = []
        for session in sessions:
            dfs.append(load_neuron_type_df(session))
        
        self.neuron_type_df = merge_sessions_df(dfs)
        # self.neuron_type_df.astype({'Neuron Type': list}).dtypes
        
        self.piecharts(save_flag)
        
       
    def count_encodings(self,subsession):
        
        self.unit_list = self.neuron_type_df['Unit'].to_list()
        self.unit_list = [unit for unit in self.unit_list if subsession in unit]
        
        self.total_counts = {
                  'num_units':0, 
                  'num_value':0,
                  'num_value1':0,
                  'num_value2':0,
                  'num_value1&2':0,
                  'num_value_low':0,
                  'num_value_high':0,
                  'num_value_diff':0,
                  'num_abs_value_diff':0,
                  'num_choice_targ':0,
                  'num_choice_value':0,
                  'num_side':0,
                  'num_time':0,
                  'non_encoding':0}
        
        for unit in self.unit_list:
            unit_encoding = get_unit_data(self.neuron_type_df,unit,'Neuron Type')
            # if not isinstance(unit_encoding, str):
            #     unit_encoding = ','.join(unit_encoding) #join function is to turn the list of strs into a single str to make it easier to search for keywords

            # self.total_counts = self.count(unit_encoding,self.total_counts)
            
            self.total_counts = self.count_v2(unit_encoding,self.total_counts)
            
        
        return
    
    def count(self,unit_encoding,total_counts):
    
        total_counts['num_units'] +=1
        
        
        
        if 'Q1' in unit_encoding:
            if 'Q2' in unit_encoding:
                total_counts['num_value1&2'] +=1
            else:
                total_counts['num_value1'] +=1
        elif 'Q2' in unit_encoding:
            total_counts['num_value2'] +=1
            
        if 'Qlow' in unit_encoding:
            total_counts['num_value_low'] +=1
        if 'Qhigh' in unit_encoding:
            total_counts['num_value_high'] +=1
            
        if 'Qdiff_' in unit_encoding:
            total_counts['num_value_diff'] +=1
        if 'absQdiff' in unit_encoding:
            total_counts['num_abs_value_diff'] +=1
        
        if 'Q' in unit_encoding:
            total_counts['num_value'] +=1    
        elif 'Choice_low' in unit_encoding:
            total_counts['num_choice_value'] +=1
        elif 'Choice1' in unit_encoding:
            total_counts['num_choice_targ'] +=1
        elif 'Side' in unit_encoding:
            total_counts['num_side'] +=1
        elif 'Time' in unit_encoding:
            total_counts['num_time'] +=1
        else:
            total_counts['non_encoding'] +=1
        
        return total_counts


    def count_v2(self,unit_encoding,total_counts):
    
        total_counts['num_units'] +=1
        
        
        targ_value = 'Q1' in unit_encoding #neuron tracks value of the targets
        opt_value = 'Qhigh' in unit_encoding #neuron tracks value of whichever target is optimal
        value_diff = 'Qdiff_' in unit_encoding
        abs_value_diff = 'absQdiff' in unit_encoding
        
        # if targ_value and not opt_value and not
        
        if targ_value or opt_value or value_diff or abs_value_diff:
            total_counts['num_value'] +=1
        elif 'Choice_low' in unit_encoding:
            total_counts['num_choice_value'] +=1
        elif 'Choice1' in unit_encoding:
            total_counts['num_choice_targ'] +=1
        elif 'Side' in unit_encoding:
            total_counts['num_side'] +=1
        elif 'Time' in unit_encoding:
            total_counts['num_time'] +=1
        else:
            total_counts['non_encoding'] +=1
            
            
        if 'Qlow' in unit_encoding:
            total_counts['num_value_low'] +=1
        if 'Qhigh' in unit_encoding:
            total_counts['num_value_high'] +=1
            
        # if :
        #     total_counts['num_value_diff'] +=1
        # if :
        #     total_counts['num_abs_value_diff'] +=1
        
        if 'Q' in unit_encoding:
            total_counts['num_value'] +=1    
        
        
        return total_counts    
    
    def piecharts(self,save_flag):
        
        for subsession in SUBSESSIONS:
            self.count_encodings(subsession)
            
            num_units = self.total_counts['num_units']
            num_value_units = self.total_counts["num_value"]
    
            fig,axs = plt.subplots(1,2)
            axs[0].set_title(f'Num Neurons: {num_units}')
            axs[1].set_title(f'Num Value: {num_value_units}')
            
            #general type pie
            if num_units>0:
                ax=axs[0]
                
                slices=[self.total_counts["num_value"],
                        self.total_counts['num_choice_targ'],
                        self.total_counts['num_choice_value'],
                        self.total_counts['num_side'],
                        self.total_counts['num_time'],
                        self.total_counts['non_encoding']]
                explode=[0.2,0,0,0,0,0] #to make the value slice stick out slightly
                labels=['Value','Target-based Choice','Value-based Choice','Action','Time','Non-encoding']
                colors=['#74c476',    'pink',             'purple',         'red',  'blue',   'gray']
    
                ax.pie(slices,explode,labels,colors)
                
            #value type pie
            if num_value_units>0:
                ax=axs[1]
                
                slices=[self.total_counts['num_value1&2'],
                        self.total_counts['num_value1'],
                        self.total_counts['num_value2'],
                        self.total_counts['num_value_low'],
                        self.total_counts['num_value_high'],
                        self.total_counts['num_value_diff'],
                        self.total_counts['num_abs_value_diff']]
                labels=['Q1&Q2',    'Q1',      'Q2',     'Qlow',  'Qhigh' , 'Qdiff', 'absQdiff']
                colors=['#edf8e9','#bae4b3','#74c476','#31a354','#006d2c',  'blue',    'purple']
    
                ax.pie(slices,labels=labels,colors=colors)
            
            
            fig.suptitle('All sessions\n' + subsession)
            fig.tight_layout()
            
            if save_flag:
                fname=f'{PROJ_FOLDER}\\Figures\\allsess_pie_{subsession}.svg'
                plt.savefig(fname,format='svg')
                print(f'{fname} saved!')
        
        return

        



class NeuronTypeVenns:
    
    def __init__(self, sessions: list, save_flag: bool):
        
        dfs = []
        for session in sessions:
            dfs.append(load_neuron_type_df(session))
        
        self.neuron_type_df = merge_sessions_df(dfs)
        # self.neuron_type_df.astype({'Neuron Type': list}).dtypes
        
        self.piecharts(save_flag)
        
       
    def count_encodings(self,subsession):
        
        self.unit_list = self.neuron_type_df['Unit'].to_list()
        self.unit_list = [unit for unit in self.unit_list if subsession in unit]
        
        self.total_counts = {
                  'num_units':0, 
                  'num_value':0,
                  'num_value1':0,
                  'num_value2':0,
                  'num_value1&2':0,
                  'num_value_low':0,
                  'num_value_high':0,
                  'num_value_diff':0,
                  'num_abs_value_diff':0,
                  'num_choice_targ':0,
                  'num_choice_value':0,
                  'num_side':0,
                  'num_time':0,
                  'non_encoding':0}
        
        for unit in self.unit_list:
            unit_encoding = get_unit_data(self.neuron_type_df,unit,'Neuron Type')
            # if not isinstance(unit_encoding, str):
            #     unit_encoding = ','.join(unit_encoding) #join function is to turn the list of strs into a single str to make it easier to search for keywords

            # self.total_counts = self.count(unit_encoding,self.total_counts)
            
            self.total_counts = self.count_v2(unit_encoding,self.total_counts)
            
        
        return
    
    def count(self,unit_encoding,total_counts):
    
        total_counts['num_units'] +=1
        
        
        
        if 'Q1' in unit_encoding:
            if 'Q2' in unit_encoding:
                total_counts['num_value1&2'] +=1
            else:
                total_counts['num_value1'] +=1
        elif 'Q2' in unit_encoding:
            total_counts['num_value2'] +=1
            
        if 'Qlow' in unit_encoding:
            total_counts['num_value_low'] +=1
        if 'Qhigh' in unit_encoding:
            total_counts['num_value_high'] +=1
            
        if 'Qdiff_' in unit_encoding:
            total_counts['num_value_diff'] +=1
        if 'absQdiff' in unit_encoding:
            total_counts['num_abs_value_diff'] +=1
        
        if 'Q' in unit_encoding:
            total_counts['num_value'] +=1    
        elif 'Choice_low' in unit_encoding:
            total_counts['num_choice_value'] +=1
        elif 'Choice1' in unit_encoding:
            total_counts['num_choice_targ'] +=1
        elif 'Side' in unit_encoding:
            total_counts['num_side'] +=1
        elif 'Time' in unit_encoding:
            total_counts['num_time'] +=1
        else:
            total_counts['non_encoding'] +=1
        
        return total_counts


    def count_v2(self,unit_encoding,total_counts):
    
        total_counts['num_units'] +=1
        
        
        targ_value = 'Q1' in unit_encoding #neuron tracks value of the targets
        opt_value = 'Qhigh' in unit_encoding #neuron tracks value of whichever target is optimal
        value_diff = 'Qdiff_' in unit_encoding
        abs_value_diff = 'absQdiff' in unit_encoding
        
        # if targ_value and not opt_value and not
        
        if targ_value or opt_value or value_diff or abs_value_diff:
            total_counts['num_value'] +=1
        elif 'Choice_low' in unit_encoding:
            total_counts['num_choice_value'] +=1
        elif 'Choice1' in unit_encoding:
            total_counts['num_choice_targ'] +=1
        elif 'Side' in unit_encoding:
            total_counts['num_side'] +=1
        elif 'Time' in unit_encoding:
            total_counts['num_time'] +=1
        else:
            total_counts['non_encoding'] +=1
            
            
        if 'Qlow' in unit_encoding:
            total_counts['num_value_low'] +=1
        if 'Qhigh' in unit_encoding:
            total_counts['num_value_high'] +=1
            
        # if :
        #     total_counts['num_value_diff'] +=1
        # if :
        #     total_counts['num_abs_value_diff'] +=1
        
        if 'Q' in unit_encoding:
            total_counts['num_value'] +=1    
        
        
        return total_counts    
    
    def piecharts(self,save_flag):
        
        for subsession in SUBSESSIONS:
            self.count_encodings(subsession)
            
            num_units = self.total_counts['num_units']
            num_value_units = self.total_counts["num_value"]
    
            fig,axs = plt.subplots(1,2)
            axs[0].set_title(f'Num Neurons: {num_units}')
            axs[1].set_title(f'Num Value: {num_value_units}')
            
            #general type pie
            if num_units>0:
                ax=axs[0]
                
                slices=[self.total_counts["num_value"],
                        self.total_counts['num_choice_targ'],
                        self.total_counts['num_choice_value'],
                        self.total_counts['num_side'],
                        self.total_counts['num_time'],
                        self.total_counts['non_encoding']]
                explode=[0.2,0,0,0,0,0] #to make the value slice stick out slightly
                labels=['Value','Target-based Choice','Value-based Choice','Action','Time','Non-encoding']
                colors=['#74c476',    'pink',             'purple',         'red',  'blue',   'gray']
    
                ax.pie(slices,explode,labels,colors)
                
            #value type pie
            if num_value_units>0:
                ax=axs[1]
                
                slices=[self.total_counts['num_value1&2'],
                        self.total_counts['num_value1'],
                        self.total_counts['num_value2'],
                        self.total_counts['num_value_low'],
                        self.total_counts['num_value_high'],
                        self.total_counts['num_value_diff'],
                        self.total_counts['num_abs_value_diff']]
                labels=['Q1&Q2',    'Q1',      'Q2',     'Qlow',  'Qhigh' , 'Qdiff', 'absQdiff']
                colors=['#edf8e9','#bae4b3','#74c476','#31a354','#006d2c',  'blue',    'purple']
    
                ax.pie(slices,labels=labels,colors=colors)
            
            
            fig.suptitle('All sessions\n' + subsession)
            fig.tight_layout()
            
            if save_flag:
                fname=f'{PROJ_FOLDER}\\Figures\\allsess_pie_{subsession}.svg'
                plt.savefig(fname,format='svg')
                print(f'{fname} saved!')
        
        return
            
        
        
class LearningRateRegressions:
    
    def __init__(self,session,spike_data_dict,neuron_type_dict):
        
        ## Get Spike Info (Firing Rates)
        assert spike_data_dict['Session'] == neuron_type_dict['Session'] == session, 'Sessions do not match!'
        self.spike_df = spike_data_dict['fr_df']
        self.session = spike_data_dict['Session']
        
        ## Learning Rates for all Units
        self.learning_rates_df = None
        self.learning_rates = np.linspace(start=0.01,stop=0.99,num=50)
        self.Behavior = ProcessBehavior(session) #need to instantiate behavior
        self.get_learning_rates()
        
        ## Filter Learning Rates by Neuron Type
        self.neuron_type_df = neuron_type_dict['df']
        # self.filter_learning_rates()
        
        ## Save Out
        self.dict_out = {'Session':self.session, 'df':self.learning_rates_df}
        
        return



    def get_learning_rates(self):
        
        print('Performing learning rate regressions..')
            
        df_rows_learning_rates = []
        
        counter=0
        for ii,subsession in enumerate(SUBSESSIONS):           
        
            assert len(self.Behavior.behavior_df) == len(self.spike_df), 'Number of trials do not match!'
            num_trials = len(self.spike_df)
            
            trials = get_stable_volatile_trials(self.session)[subsession] #get trials for specified subsession
            trials = trials[trials<num_trials] #trim to match actual session length
        
            if not len(trials) > 1: #if no trials for current subsession
                print(f'No trials for {subsession}. Skipping!')    
                continue
        
            # Get FR for each neuron
            units = self.spike_df['Unit_labels'][0]
            for i,unit in enumerate(units):
                
                FR = self.spike_df[unit].iloc[trials] #vector of FRs for subsession trial
                
                self.regression_weights_df = self.learning_rate_regression(FR,trials)
                
                regressor_names = self.regression_weights_df.columns
                
                # Find learning rate which maximizes each regression weight
                unit_subsession_labeled = f'{unit} {subsession}'
                row_dict_learning_rates = {'Unit':unit_subsession_labeled}
                for j in range(len(regressor_names)):
                    
                    # idx of max regression weight for current regressor
                    idx_max_regressor = self.regression_weights_df[regressor_names[j]].idxmax()
                    
                    # Find and store learning rate which corresponds to max regression weight
                    row_dict_learning_rates['Learning rate for ' + regressor_names[j]] = self.learning_rates[idx_max_regressor]
                    
                df_rows_learning_rates.append(row_dict_learning_rates)
                
                # print(f'{unit_subsession_labeled} learning rate found! ({counter+1}/{len(units)*len(SUBSESSIONS)})')
                counter+=1
                
        self.learning_rates_df = pd.DataFrame(df_rows_learning_rates)
        # print(self.learning_rates_df)
        
        return
    
                
                
    def learning_rate_regression(self,FR,trials):
        
        # Get behavior regressor for each learning rate
        df_rows_regression_weights = []
        for learning_rate in self.learning_rates:
            
            self.Behavior.get_behavior(Q_learning=True, learning_rate=learning_rate)
            behavior_df = self.Behavior.behavior_df
            
            # Make array of behavioral data
            # regressor_labels = behavior_df.columns
            # regressor_labels = ['Choice1','Side','Q1','Q2','Time']
            regressor_matrix = sm.add_constant(behavior_df[REGRESSORS].iloc[trials],has_constant='raise') #.to_numpy() 
            #raise flag will let us know if one of the variables is already a constant (which would be bad)

            
            # Regress FR against behavior
            model = sm.OLS(FR,regressor_matrix,hasconst=True)
            res = model.fit()
            
            row_dict_regression_weights = {}
            
            # Get weights for each regressor
            regressor_names = res.params.keys()
            for j in range(len(regressor_names)):
                row_dict_regression_weights[regressor_names[j]] = res.params.values[j]
                
            df_rows_regression_weights.append(row_dict_regression_weights)
        
        # Organize regression weights for each learning rate into a df  
        return pd.DataFrame(df_rows_regression_weights)
    
    
                
    # def filter_learning_rates(self):
        
    #     units = self.neuron_type_df['Unit'].values
    #     cols = self.learning_rates_df.columns
        
    #     for i,unit in enumerate(units):
            
    #         neuron_type = get_unit_data(self.neuron_type_df,unit,'Neuron Type')
            
    #         for col in cols: #loop thru "Learning rate for <regressor>" df columns
    #             if col.startswith('Learning rate for '):
                    
    #                 # if this neuron's type does not include the regressor specified by this column,
    #                 # mark it as Not Applicable (N/A)
    #                 if col.removeprefix('Learning rate for ') not in neuron_type: 
    #                     self.learning_rates_df.loc[i,col] = 'N/A' 
                    
    #     return           
        

        
#     def learning_rate_regression_loop(self):        
        
#         print('Performing learning rate regressions..')
        
#         df_rows = []
        
#         # Get behavior for each learning rate and then do regression with resultant behavior regressors
#         for learning_rate in self.learning_rates:
            
#             #calculate value using current learning rate and get subsequent df
#             self.Behavior.get_behavior(Q_learning=True, learning_rate=learning_rate)
#             behavior_df = self.Behavior.behavior_df
            
#             #combine behavior and spike dfs
#             df = pd.concat([behavior_df,self.spike_df],axis='columns') #merge behavior_df and firing_rate_df
#             # self.dict_out = {'Name':session, 'df':self.df}
            
#             #find regression weights using current learning rate
#             weights_dict = self.learning_rate_regression(df)
#             df_rows.append(weights_dict) #each row of the df is a dict
            
#         self.weights_df = pd.DataFrame(df_rows) #compile rows into a df 
        
#         return
        
    
    
    
#     def learning_rate_regression(self,df):
#         '''
#         Parameters
#         ----------
#         df : combined dataframe that contains both firing rate and behavior/task info for each trial

#         Returns
#         -------
#         Resultant regression weights in a labelled dictionary

#         '''

#         # Make array of behavioral data
#         regressor_labels = ['Choice1','Side','Q1','Q2','Time']
#         regressor_matrix = df[regressor_labels].to_numpy()
        
#         encodings = np.full(len(df['Unit_labels'][0]),None)
#         units = df['Unit_labels'][0]
        
        
        
        
#         # Regress FR of each unit against behavior
#         for i,unit in enumerate(units): #loop thru all units of session
        
#             FR = df[unit].to_numpy() #vector of FRs for each trial

#             model = sm.OLS(FR, sm.add_constant(regressor_matrix),hasconst=True)
#             res = model.fit()
            
# #             print(FR)
# #             print(sm.add_constant(regressor_matrix))
            
#             # print(res.f_pvalue)
#             if res.f_pvalue < self.alpha_threshold: #if regression is statistically significant
#                 signif_regressors_bools = res.pvalues<self.alpha_threshold #get which regressors were statistically signficant
#                 signif_regressors = [reg for indx,reg in enumerate(regressor_labels) if signif_regressors_bools[indx]] #get names of those regressors
#                 encodings[i] = signif_regressors 
                
# #                 signif_regressors_posneg = res.params[signif_regressors_bools] #get the sign of the coefficient of the significant regressors
# #                 encoding_posneg[chann][unit] = signif_regressors_posneg
# #                 #within nested dict, unit number is the key and the significant regressors are the values
            
#             else: #if regression isn't statistically significant, there's no encoding
#                 encodings[i] = 'Non-encoding'
                
                
#             print(f'{unit} done. Neuron type: {encodings[i]}. ({i+1}/{len(df["Unit_labels"][0])})')
            
#         print('Neuron type regressions done!')    
        
#         self.neuron_type_df = pd.DataFrame.from_dict({'Unit':units, 'Neuron Type':encodings})
#         # print(self.neuron_type_df)
        
        
#         return
            

    
    
class Plotting:
    
    
    def __init__(self,neuron_type_dict,learning_rates_dict):
        
        small_fontsize = 8
        med_fontsize = 12
        large_fontsize = 18
        
        assert neuron_type_dict['Session'] == learning_rates_dict['Session'], 'Sessions do not match!'
        self.session = neuron_type_dict['Session']
        self.neuron_type_df = neuron_type_dict['df']
        self.learning_rates_df = learning_rates_dict['df']
        self.unit_list = self.neuron_type_df['Unit'].to_list()
        
        self.total_counts = None
        self.count_encodings()
        
        self.piecharts()
        
        return

        

        
       
    def learning_rate_violins(self,volatile_learning_rates,stable_learning_rates):
        
        fig,ax=plt.subplots()
        ax.violinplot([volatile_learning_rates,stable_learning_rates])
        ax.set_title(self.session)

        return
        
        
        
    def summary_behavior(self):
        return
        
        
    def session_behavior(self, session):
        return



def make_datasets_for_modeling():
    #msdynode takes input data as pkl files with structure:
    # {'LFP': array of shape 1 x n_trials x n_channels x data_size,
    #'Spikes': array of shape 1 x n_trials x n_channels x data_size}
    
    
    
    return


def merge_sessions_df(dfs: list): # 'dfs' is a list of DataFrames
   
    return pd.concat(dfs,axis='index',ignore_index=True)


# def plot_learning_rates_old(learning_rates_df): 
    
#     cols = learning_rates_df.columns
    
#     for col in cols:
#         if col.startswith('Learning rate for '):
            
#             learning_rates = np.array(learning_rates_df[col].values)
#             learning_rates = [item for item in learning_rates if isinstance(item, (int, float))]
            
#             num_units = len(learning_rates)
            
#             if num_units > 0:
                
#                 fig,ax=plt.subplots()
#                 ax.hist(learning_rates,bins=10,range=(0,1))
#                 ax.set_xlabel(col)
#                 ax.set_ylabel('Number of neurons')
#                 ax.set_xlim([0,1])
#                 fig.suptitle(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
#                 # ax.set_title(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
#                 # fig.suptitle(self.name)
#                 fig.tight_layout()
            
#     return




def load_learning_rate_df(session: str):
    
    return pd.read_csv(os.path.join(PROJ_FOLDER,session,f'{session}_learning_rate_df.csv'))

def load_neuron_type_df(session: str):
    
    return pd.read_csv(os.path.join(PROJ_FOLDER,session,f'{session}_type_df.csv'))

def combine_pval_dicts(list_of_pval_dicts):
    
    pval_df_keys = list_of_pval_dicts[0].keys()
    
    # Unpack list and dicts to get the dfs for each regressor
    combined_pval_dict = {}
    for key in pval_df_keys:
        
        list_of_dfs = []
        for pval_dict in list_of_pval_dicts:
            
            list_of_dfs.append(pval_dict[key])
            
        combined_pval_dict[key] = pd.concat(list_of_dfs,ignore_index=True)

    return combined_pval_dict

def plot_temporal_encoding(pval_dict,temporal_encoding_regression_obj):
    # temporal_encoding_regression_obj is passed purely to get plotting parameters (see below)
    areas = temporal_encoding_regression_obj.areas
    regressors = temporal_encoding_regression_obj.regressors
    t_vector = temporal_encoding_regression_obj.psth_t_vector
    alpha_threshold = temporal_encoding_regression_obj.alpha_threshold
    
    num_units_list = []
    for i,area in enumerate(areas):
        
        fig,ax = plt.subplots()
        
        for j,regressor in enumerate(regressors):
            
            # get pval_dict for current regressor
            df_list = list(pval_dict.keys())
            regressor_index = [i for i in range(len(df_list)) if regressor in df_list[i]][0]
            pval_df = pval_dict[df_list[regressor_index]]
            
            # get % encoding for current area 
            pval_df = area_parser(pval_df,area)
            percent_encoding, num_units = calc_percent_encoding(pval_df,alpha_threshold)
            if j==0: #only do once for each area
                num_units_list.append(num_units)
                
            #plot                    
            ax.plot(percent_encoding,color=COLORS[i],linestyle=LINESTYLES[j])
    
        # Plot details
        ax.set_ylabel('% Units Encoding')
        ylo,yhi,dy = 0.0,0.8,0.1
        ax.set_ylim([ylo,yhi])
        ax.set_yticks(np.arange(ylo,yhi+dy,dy),(np.arange(ylo,yhi+dy,dy)*100).astype(int))
        ax.set_xticks(np.arange(len(percent_encoding)),t_vector)
        ax.set_xlabel('Time relative to Choice (sec)')
        ax.set_xticks(np.arange(len(percent_encoding)),t_vector)
        ax.set_title('Temporal Encoding Dynamics')
    
        # legend
        # custom_lines = [Line2D([0], [0], color=COLORS[0]),
        #                 Line2D([0], [0], color=COLORS[1]),
        #                 Line2D([0], [0], color=COLORS[2]),
        #                 Line2D([0], [0], color='gray', linestyle=LINESTYLES[0]),
        #                 Line2D([0], [0], color='gray', linestyle=LINESTYLES[1]),
        #                 Line2D([0], [0], color='gray', linestyle=LINESTYLES[2]),
        #                 Line2D([0], [0], color='gray', linestyle=LINESTYLES[3])]     
        # area_labels = [f'{areas[i]}, {num_units_list[i]} units' for i in range(len(areas))]
        # ax.legend(custom_lines, area_labels + regressors,ncols=4)
        
        
        custom_lines = [Line2D([0], [0], color=COLORS[i], linestyle=LINESTYLES[0]),
                        Line2D([0], [0], color=COLORS[i], linestyle=LINESTYLES[1]),
                        Line2D([0], [0], color=COLORS[i], linestyle=LINESTYLES[2]),
                        Line2D([0], [0], color=COLORS[i], linestyle=LINESTYLES[3])]     
        ax.legend(custom_lines,regressors,ncols=4,loc='upper right')
        ax.set_title(f'{area}, {num_units_list[i]} units')
    
    return


def plot_var_corr(corr_matrix,list_of_vars,title):
    num_vars = len(list_of_vars)
    fig,ax=plt.subplots()
    im = ax.imshow(corr_matrix,vmin=0.5,vmax=0.9,cmap='Reds')
    ax.set_title('Behavior Variable Correlation (R^2)')
    ax.set_aspect(0.9)#to make labels readable
    c=plt.colorbar(im)
    plt.xticks(range(num_vars),list_of_vars)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15) #
    plt.yticks(range(num_vars),list_of_vars)
    fig.suptitle(title)
    fig.tight_layout()
    return
    
def plot_value_dist(behavior_df,save_flag=False):
    
    df = behavior_df
    Q1_vol = df['Q1'].loc[df['Volatile']==1]
    Q1_stab = df['Q1'].loc[df['Stable']==1]
    # Q2_vol = df['Q2'].loc[df['Volatile']==1]
    # Q2_stab = df['Q2'].loc[df['Stable']==1]
    
    #calculate kurtosis
    kurt_vol = kurtosis(Q1_vol)
    kurt_stab = kurtosis(Q1_stab)
    
    bins=np.linspace(0.0,1.0,num=15)
    
    fig,ax=plt.subplots()
    
    ax.hist(Q1_vol,density=True,bins=bins,label=f'Volatile, Kurt = {kurt_vol}',color='red',alpha=0.5)
    ax.hist(Q1_stab,density=True,bins=bins,label=f'Stable, Kurt = {kurt_stab}',color='blue',alpha=0.5)
    ax.legend()
    ax.set_title('Actual Value Distributions')
    # fig.suptitle(self.session)
    
    #flip around to make bimodal dist look normal
    Q1_vol=np.array(Q1_vol)
    Q1_vol[Q1_vol>0.5] = 1.5-Q1_vol[Q1_vol>0.5]
    Q1_vol[Q1_vol<0.5] = 0.5-Q1_vol[Q1_vol<0.5]
    Q1_stab=np.array(Q1_stab)
    Q1_stab[Q1_stab>0.5] = 1.5-Q1_stab[Q1_stab>0.5]
    Q1_stab[Q1_stab<0.5] = 0.5-Q1_stab[Q1_stab<0.5]
    
    # fig,ax=plt.subplots()
    
    # ax.hist(Q1_vol,density=True,bins=bins,label='Volatile',color='red',alpha=0.5)
    # ax.hist(Q1_stab,density=True,bins=bins,label='Stable',color='blue',alpha=0.5)
    # ax.legend()
    # ax.set_title('Flipped Value Distributions')
    # fig.suptitle(self.session)
    
    #Q1
    # ax.hist([Q1_vol,Q1_stab], histtype='bar', stacked=True, density=True, label=['Volatile','Stable'])
    # ax.legend()
    # fig.suptitle(self.session)
    
    # #Q2
    # ax=axs[1]
    # ax.hist(np.array(Q2_vol,Q2_stab),label=['Volatile','Stable'])
    # ax.legend()
    
    
    
    if save_flag:
        fname=f'value_dist.svg'
        plt.savefig(fname,format='svg')
        print(f'{fname} saved!')
    
    
    return
def plot_learning_rates(learning_rates_df,neuron_type_df): 
    
    cols = learning_rates_df.columns
    unique_units = get_unique_units(learning_rates_df["Unit"])
    
    for col in cols:
        if 'Q' in col: #only do for value related stuff
            
            learning_rates_list = []
            subsession_labels = []
            for subsession in SUBSESSIONS:
                
                # Get learning rate for all units that encode this col for this subsession
                learning_rates = []
                for unit in unique_units:
                    
                    # if this unit encodes the current column during this subsession
                    if col.removeprefix('Learning rate for ') in get_unit_data(neuron_type_df,f'{unit} {subsession}','Neuron Type'):

                        
                        learning_rates.append(learning_rates_df[col][learning_rates_df["Unit"]==f'{unit} {subsession}'].values)
                        # print(f'{unit} {subsession}')
                        # print(col)
                        # print(get_unit_data(neuron_type_df,f'{unit} {subsession}','Neuron Type'))
                        # print(learning_rates_df[col][learning_rates_df["Unit"]==f'{unit} {subsession}'].values)
                        # xxx
        
                if len(learning_rates)>0:
                    learning_rates_list.append(np.array(learning_rates).flatten())
                    subsession_labels.append(subsession + f'\nnum_units: {len(learning_rates)}')
            
            if len(learning_rates_list) > 0:
                
                fig,ax=plt.subplots()
                ax.violinplot(learning_rates_list)#,bins=10,range=(0,1))
                ax.set_xticks(np.arange(1,len(learning_rates_list)+1),subsession_labels)
                ax.set_ylabel(col)
                ax.set_ylim([0,1])
                fig.suptitle(f'{col.removeprefix("Learning rate for ")} Learning Rates by subsession')
                fig.tight_layout()
                
                plt.savefig(os.path.join(PROJ_FOLDER,'Figures',f'{col.removeprefix("Learning rate for ")} Learning Rates.svg'),format="svg")
                
                
            else:
                if len(learning_rates_list[0]) > 0:
                    print(f'No volatile block {col.removeprefix("Learning rate for ")} units found!')
                else:
                    print(f'No stable block {col.removeprefix("Learning rate for ")} units found!')
    return



# def filter_learning_rates(learning_rates_df,neuron_type_df):
    
#     units = neuron_type_df['Unit'].values
#     cols = learning_rates_df.columns
    
#     for i,unit in enumerate(units):
        
#         neuron_type = get_unit_data(neuron_type_df,unit,'Neuron Type')
        
#         for col in cols: #loop thru "Learning rate for <regressor>" df columns
#             if col.startswith('Learning rate for '):
                
#                 # if this neuron's type does not include the regressor specified by this column,
#                 # mark it as Not Applicable (N/A)
#                 if col.removeprefix('Learning rate for ') not in neuron_type: 
#                     learning_rates_df.loc[i,col] = 'N/A' 
                
#     return    



# def get_learning_rate_change_old(learning_rates_df):
    
#     df = learning_rates_df
    
#     # get names of units (without subsession specification suffix)
#     unique_units = get_unique_units(df["Unit"])

#     for unit in unique_units:
        
#         #find learning rates for current unit's volatile and stable block
#         idx_volatile = df.index[df['Unit'] == f'{unit} volatile block' ]
#         idx_stable = df.index[df['Unit'] == f'{unit} stable block' ]
        
#         if len(idx_volatile)>0 and len(idx_stable)>0:
            
            
#             # to make indexes match for subtraction
#             row_stable = df.iloc[idx_stable].reset_index(drop=True) 
#             row_volatile = df.iloc[idx_volatile].reset_index(drop=True)
            
#             # to deal with strings/nans
#             row_stable = row_stable.apply(pd.to_numeric, errors='coerce') 
#             row_volatile = row_volatile.apply(pd.to_numeric, errors='coerce')

#             #find learning rate change (delta) by finding difference between stable and volatile blocks
#             delta = row_stable.sub(row_volatile)
#             delta['Unit'] = f'{unit} delta'
            
#             df = pd.concat([df,delta], ignore_index=True)
            
#             print('delta made!')

#     return df 


def plot_learning_rate_change(learning_rates_df,neuron_type_df):
    
    cols = learning_rates_df.columns
    
    # get names of units (without subsession specification suffix)
    unique_units = get_unique_units(learning_rates_df["Unit"])
    
    for col in cols:
        if 'Q' in col: #only do for value related stuff
            
            # Get change in learning rate for all units that encode the current col
            deltas = []
            for unit in unique_units:
                
                # if this unit encodes the current column over all trials
                if col.removeprefix('Learning rate for ') in get_unit_data(neuron_type_df,f'{unit} all trials','Neuron Type'):
    
                    volatile = learning_rates_df[col][learning_rates_df["Unit"]==f'{unit} volatile block'].values
                    stable = learning_rates_df[col][learning_rates_df["Unit"]==f'{unit} stable block'].values
                    
                    if len(volatile)>0 and len(stable)>0:
    
                        deltas.append(stable[0] - volatile[0])
            
                    
            num_units = len(deltas)
            
            
            if num_units > 0:
                
                fig,ax=plt.subplots()
                ax.hist(deltas,bins=20,range=(-1,1))
                ax.set_xlabel(f'Delta {col}\n(Stable - Volatile)')
                ax.set_ylabel('Number of neurons')
                ax.set_xlim([-1,1])
                [ylo,yhi] = ax.get_ylim()
                ax.vlines(0,ylo,yhi,color='k',linestyle='--',label='No change')
                ax.legend()
                fig.suptitle(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
                # ax.set_title(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
                # fig.suptitle(self.name)
                fig.tight_layout()
                
                plt.savefig(os.path.join(PROJ_FOLDER,'Figures',f'{col.removeprefix("Learning rate for ")} Learning Rate Deltas.svg'),format="svg")
                    

    return
        
        
# def plot_learning_rate_change(learning_rates_df,neuron_type_df):
    
#     idxs_delta = learning_rates_df.index[learning_rates_df['Unit'].str.contains('delta')]
#     assert idxs_delta.any(), 'No delta values found!'
    
#     cols = learning_rates_df.columns
    
#     for col in cols:
#         if col.startswith('Learning rate for '):
            
            
#             learning_rates = np.array(learning_rates_df[col].iloc[idxs_delta].values)
#             learning_rates = [item for item in learning_rates if isinstance(item, (int, float))]
#             learning_rates = [item for item in learning_rates if ~np.isnan(item)]
#             learning_rates = [item for i,item in enumerate(learning_rates) if col.removeprefix("Learning rate for ") in get_unit_data(neuron_type_df,unit,'Neuron Type')]
            
#             num_units = len(learning_rates)
            
#             if num_units > 0:
                
#                 fig,ax=plt.subplots()
#                 ax.hist(learning_rates,bins=10,range=(0,1))
#                 ax.set_xlabel(col)
#                 ax.set_ylabel('Number of neurons')
#                 ax.set_xlim([0,1])
#                 fig.suptitle(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
#                 # ax.set_title(f'Total number of {col.removeprefix("Learning rate for ")} units: {num_units}')
#                 # fig.suptitle(self.name)
#                 fig.tight_layout()
            
#     return


def get_unique_units(unit_list):

    unit_list = [unit.removesuffix(' stable block') for unit in unit_list]
    unit_list = [unit.removesuffix(' volatile block') for unit in unit_list]
    unit_list = [unit.removesuffix(' all trials') for unit in unit_list]
    
    return np.unique(np.array(unit_list))

def get_unit_data(df,unit,heading):
    
    desired_row = df[df['Unit']==unit]

    if len(desired_row)>0:
        desired_value = desired_row[heading].values[0]
    else:
        desired_value = ''

    return desired_value


def get_stable_volatile_trials(session):
    
     if session == 'braz20240927_01_te5384':
         stable_first = True
     elif session == 'braz20241001_03_te5390':
         stable_first = True
     elif session == 'braz20241002_04_te5394':
         stable_first = True
     elif session == 'braz20241004_02_te5396':
         stable_first = False
     elif session == 'braz20250221_03_te1873':
         stable_first = False
     elif session == 'braz20250225_04_te1880':
         stable_first = False
     elif session == 'braz20250228_03_te1888':
         stable_first = False
     elif session == 'braz20250326_04_te1923':
         stable_first = False
     elif session == 'braz20250327_04_te1927':
         stable_first = False
     
     if stable_first:
         stable_trials = np.arange(0,100*6,step=1)
         volatile_trials = np.arange(0,20*24,step=1) + stable_trials[-1] + 1
     else:
         volatile_trials = np.arange(0,20*24,step=1)
         stable_trials = np.arange(0,100*6,step=1) + volatile_trials[-1] + 1
         
     all_trials = np.arange(0,len(volatile_trials)+len(stable_trials),step=1)
     
     return {'stable block':stable_trials, 'volatile block':volatile_trials, 'all trials':all_trials}

#### Run Methods

# # session = 'braz_test'
# session = 'braz20250225_04_te1880'
# # session = 'braz20250326_04_te1923'

# Spikes = ProcessSpikes(session)
# # Spikes.dict_out['df'].to_csv(os.path.join(PROJ_FOLDER,'spike_df.csv'))
# # xxx
# Behav = ProcessBehavior(session)
# Type = NeuronTypeRegressions(Spikes.dict_out,Behav.dict_out)
# LR = LearningRateRegressions(session,Spikes.dict_out,Type.dict_out)
# LR.plot_learning_rates()

# class SanityChecks:
#     def __init__(self):
#         sdkfj
      

# sessions = ['braz20250225_04_te1880','braz20250326_04_te1923']


def run_lrs():
    lrs_dfs_out = []
    nt_dfs_out = []
    for session in SESSIONS:
        Spikes = ProcessSpikes(session)
        Behav = ProcessBehavior(session)
        Type = NeuronTypeRegressions(Spikes.dict_out,Behav.dict_out)
        save_out_csv(Type.neuron_type_df,'UnitType',Type.session)
        nt_dfs_out.append(Type.dict_out['df'])
        LR = LearningRateRegressions(session,Spikes.dict_out,Type.dict_out)
        save_out_csv(LR.learning_rates_df,'LearningRate',LR.session)
        lrs_dfs_out.append(LR.dict_out['df'])
        print(f'{session} done!\n\n')

    plot_learning_rates(merge_sessions_df(lrs_dfs_out),merge_sessions_df(nt_dfs_out))
    plot_learning_rate_change(merge_sessions_df(lrs_dfs_out),merge_sessions_df(nt_dfs_out))
    return
    
    
def load_lrs():
    lrs_dfs_out = []
    nt_dfs_out = []
    for session in SESSIONS:
        lrs_dfs_out.append(load_learning_rate_df(session))
        nt_dfs_out.append(load_neuron_type_df(session))
        print(f'{session} loaded!')

    plot_learning_rates(merge_sessions_df(lrs_dfs_out),merge_sessions_df(nt_dfs_out))
    plot_learning_rate_change(merge_sessions_df(lrs_dfs_out),merge_sessions_df(nt_dfs_out))
    return
    
    
    
def plot_neuron_types_eachsess():
    for session in SESSIONS:
        
        #choose which sessions to save
        if session == 'braz20241002_04_te5394':
            save_flag=True
        else:
            save_flag=False
            #TODO make it try to load the neurotype df and if it cant then it processes spieks and behav. 
        Spikes = ProcessSpikes(session)
        Behav = ProcessBehavior(session)
        Type = NeuronTypeRegressions(Spikes.dict_out,Behav.dict_out)
        # Type.piecharts(save_flag)
        print(f'{session} done!')
    return
        
def plot_neuron_types_allsess():
    NeuronTypePieCharts(SESSIONS, save_flag=True)
    return
    

def plot_behavior():
    df_list =[]
    corr_list=[]
    for session in SESSIONS:
        
        #choose which sessions to do
        # if session == 'braz20250228_03_te1888':
        
        if 1:
            Behav = ProcessBehavior(session)
            # Behav.check_all_behavior()
            # Behav.plot_choices_and_rewards(save_flag=False)
            
            ##see which q learning model fits the best
            vmc = ValueModelingClass()
            vmc.plot_model_comparison(Behav.hdf_file, num_trials_A=0, num_trials_B=0)
            
            ##re get values using rishi's method
            # Behav.get_behavior(Q_learning=True)
            
            ##analyze correlation of behavior variables
            # list_of_vars = ['Q1','Choice1','Qhigh','Qchosen','Qdiff_','absQdiff','Choice_high','Side','Time']
            # corr = Behav.get_behavior_var_corr(list_of_vars)
            # plot_var_corr(corr,list_of_vars,Behav.session)
            # corr_list.append(corr)
            
            ##analyze distribution of values
            # Behav.plot_values(save_flag=False)
            # Behav.plot_value_dist(save_flag=False)
            # df_list.append(Behav.behavior_df)
      
    ##analyze distribution of values
    # df = merge_sessions_df(df_list)
    # plot_value_dist(df,save_flag=False)
    
    ##analyze correlation of behavior variables
    # corr_list = np.array(corr_list)
    # plot_var_corr(np.mean(corr_list,axis=0),list_of_vars,'Across Session Average')
    return
    
        
            
def explore_LFP():
    session = 'braz20250327_04_te1927'
    ExploreLFP(session)
    return

def do_process_spikes(session):
    s=ProcessSpikes(session,overwrite_flag=True)    
    return s
    
def run_temporal_encodings(overwrite_flag = False):
    
    pval_dicts=[]
    for i,session in enumerate(SESSIONS):
        
        if does_pkl_exist('temporal_encoding_pval_dict',session) and not overwrite_flag:
            pval_dict = load_pkl('temporal_encoding_pval_dict',session)
            if i==0: #do once to get TER object for plotting parameters
                Spikes = ProcessSpikes(session)
                Behav = ProcessBehavior(session)
                TER = TemporalEncodingRegressions(Spikes.dict_out,Behav.dict_out,overwrite_flag)
        else:
            Spikes = ProcessSpikes(session)
            Behav = ProcessBehavior(session)
            TER = TemporalEncodingRegressions(Spikes.dict_out,Behav.dict_out,overwrite_flag)
            pval_dict = TER.pval_dict
            
        pval_dicts.append(pval_dict)
        
    all_pval_dict = combine_pval_dicts(pval_dicts)
    plot_temporal_encoding(all_pval_dict,TER)
    
    return


def run_lasso():
    
    # REGRESSORS = ['Q1','Choice1','Qhigh','Qchosen','Qdiff_','absQdiff','Choice_high','Side','Time']
    
    encodings_dict= {} #over all sessions
    
    for i,session in enumerate([SESSIONS[0]]):
        
        Spikes = ProcessSpikes(session)
        units = list(Spikes.firing_rate_df.columns)
        
        Behav = ProcessBehavior(session)
        behavior = Behav.behavior_df[REGRESSORS]

        
        for j,unit in enumerate(units):
            fr = Spikes.firing_rate_df[unit]
                     
            # print(behavior.shape)
            # print(fr.shape)
            score,coefs = lasso_regression(fr,behavior,verbose = False)
            encoding = lasso_feature_selection(coefs,REGRESSORS)
            encodings_dict[unit] = encoding
            
    encodings_df = pd.DataFrame.from_dict(encodings_dict, orient='index', columns=['Encodings'])
    save_out_csv(encodings_df,'lasso_encodings','')

    
    return encodings_df

def count_encodings_frthistime(encodings_df):
    # uppercase = contains
    # lowercase = does not contain
    # * = AND
    # + = OR
    
    def count(encoding_str):
        return sum(encodings_df['Encodings'].str.contains(encoding_str))
    
    def contains(encoding_str):
        return encodings_df['Encodings'].str.contains(encoding_str)
    
    def no_contains(encoding_str):
        return ~encodings_df['Encodings'].str.contains(encoding_str)
    
    def plot_venn2(Ab, aB, AB, labelA, labelB):
        fig,ax = plt.subplots()
        venn2(subsets = (Ab, aB, AB),
              set_labels = (labelA, labelB))
        return
    
    def plot_venn3(Abc, aBc, ABc, abC, AbC, aBC, ABC, labelA, labelB, labelC):
        fig,ax = plt.subplots()
        venn3(subsets = (Abc, aBc, ABc, abC, AbC, aBC, ABC),
              set_labels = (labelA, labelB, labelC))
        return
    
    # value (V) vs choice (C) vs action/time (O)
    Vco = sum( contains('Q') *       no_contains('Choice') * (no_contains('Side') *   no_contains('Time')) )
    vCo = sum( no_contains('Q') *    contains('Choice') *    (no_contains('Side') *   no_contains('Time')) )
    VCo = sum( contains('Q') *       contains('Choice') *    (no_contains('Side') *   no_contains('Time')) )
    vcO = sum( no_contains('Q') *    no_contains('Choice') * (contains('Side') +      contains('Time')) )
    VcO = sum( contains('Q') *       no_contains('Choice') * (contains('Side') +      contains('Time')) )
    vCO = sum( no_contains('Q') *    contains('Choice') *    (contains('Side') +      contains('Time')) )
    VCO = sum( contains('Q') *       contains('Choice') *    (contains('Side') +      contains('Time')) )
    plot_venn3(Vco,vCo,VCo,vcO,VcO,vCO,VCO, 'Value','Choice','Other')
    
    # relative choice (R) vs target choice (T)
    Rt = sum( contains('Choice_high') * no_contains('Choice1') )
    rT = sum( no_contains('Choice_high') * contains('Choice1') )
    RT = sum( contains('Choice_high') * contains('Choice1') )
    plot_venn2(Rt,rT,RT, 'Relative Choice','Target Choice')
    
    # Qdiff_ (D) vs Q1 (I)
    Di = sum( contains('Qdiff_') *   no_contains('Q1') )
    dI = sum( no_contains('Qdiff_') *contains('Q1') )
    DI = sum( contains('Qdiff_') *   contains('Q1') )
    plot_venn2(Di,dI,DI, 'Qdiff','Q1')
    
    # Qhigh (H) vs |Qdiff| (A) vs Qchosen (C)
    Hac = sum( contains('Qhigh') *       no_contains('absQdiff') * no_contains('Qchosen') )
    hAc = sum( no_contains('Qhigh') *    contains('absQdiff') *    no_contains('Qchosen') )
    HAc = sum( contains('Qhigh') *       contains('absQdiff') *    no_contains('Qchosen') )
    haC = sum( no_contains('Qhigh') *    no_contains('absQdiff') * contains('Qchosen') )
    HaC = sum( contains('Qhigh') *       no_contains('absQdiff') * contains('Qchosen') )
    hAC = sum( no_contains('Qhigh') *    contains('absQdiff') *    contains('Qchosen') )
    HAC = sum( contains('Qhigh') *       contains('absQdiff') *    contains('Qchosen') )
    plot_venn3(Hac,hAc,HAc,haC,HaC,hAC,HAC, 'Qhigh','|Qdiff|','Qchosen')
    
    # Qdiff_/Q1 (N) vs Qhigh/Qchosen/|Qdiff| (M)
    Nm = sum( (contains('Qdiff_') +  contains('Q1')) * \
            (no_contains('Qhigh') * no_contains('Qchosen') * no_contains('absQdiff')) )
            
    nM = sum( (no_contains('Qdiff_') *  no_contains('Q1')) * \
            (contains('Qhigh') + contains('Qchosen') + contains('absQdiff')) )
            
    NM = sum( (contains('Qdiff_') +  contains('Q1')) * \
            (contains('Qhigh') + contains('Qchosen') + contains('absQdiff')) )
    plot_venn2(Nm,nM,NM, 'Qdiff/Q1','Qhigh/Qchosen/|Qdiff|')
    
    
    # Bar chart of all regressors 
    counts=[]
    for reg in REGRESSORS:
        counts.append(count(reg))
    
    idx_sort = np.argsort(counts)   
    fig,ax = plt.subplots()
    ax.bar(np.array(REGRESSORS)[idx_sort[::-1]],np.array(counts)[idx_sort[::-1]])
    ax.set_ylabel('Num Neurons')
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    
    return
    

# def regressor_selection_table():
    
#     master_regressor_list = ['Q1','Choice1','Qhigh','Qchosen','Qdiff_','absQdiff','Choice_high','Side','Time']
#     df_dict = dict()
    
#     for i,session in enumerate(SESSIONS):
        
#         Spikes = ProcessSpikes(session)
#         units = Spikes.firing_rate_df['Unit_labels'][0]
        
#         Behav = ProcessBehavior(session)
#         behavior_df = Behav.behavior_df

        
#         for j,unit in enumerate(units):
#             fr = Spikes.firing_rate_df[unit]
                     
#             # lasso (regularization feature selection)
#             score,coefs = lasso_regression(fr,behavior_df[master_regressor_list],regularization_strength = 0.1)
#             lasso_res = lasso_feature_selection(coefs,master_regressor_list)
            
#             # full regressions
#             regressor_list_1 = ['Q1','Qchosen','Choice1','Side','Time']
#             full_reg_res1 = encoding_regression_feature_selection(regressor_list_1,fr,behavior_df)
#             regressor_list_2 = ['Q1','absQdiff','Choice1','Side','Time']
#             full_reg_res2 = encoding_regression_feature_selection(regressor_list_2,fr,behavior_df)
            
#             # simple regressions
#             simple_reg_res = simple_regression_feature_selection(master_regressor_list,fr,behavior_df)
                
                
            
#             df_dict[unit] = [lasso_res, full_reg_res1, full_reg_res2, simple_reg_res]

#     df = pd.DataFrame.from_dict(df_dict, orient='index', columns = ['Lasso','Full Regression 1', 'Full Regression 2', 'Simple Regressions'])
#     save_out_csv(df,'FeatureSelectionTable','')
    
#     return df


def regression_rsqr_table():
    
    df_dict = dict()
    
    for i,session in enumerate(SESSIONS):
        
        Spikes = ProcessSpikes(session)
        units = list(Spikes.firing_rate_df.columns)
        
        Behav = ProcessBehavior(session)
        behavior_df = Behav.behavior_df

        
        for j,unit in enumerate(units):
            fr = Spikes.firing_rate_df[unit]
                     
            # lasso (regularization feature selection)
            score,coefs = lasso_regression(fr,behavior_df[REGRESSORS],verbose=False)
            lasso_res = score
            
            # full regressions
            f_pval,pvals,rsqr = encoding_regression(fr,behavior_df[['Q1','Qchosen','Choice1','Side','Time']])
            Q1Qch_res = rsqr
            
            f_pval,pvals,rsqr = encoding_regression(fr,behavior_df[['Q1','absQdiff','Choice1','Side','Time']])
            Q1absQdiff_res = rsqr
            
            f_pval,pvals,rsqr = encoding_regression(fr,behavior_df[['Q1','Qhigh','Choice1','Side','Time']])
            Q1Qhigh_res = rsqr
            
            f_pval,pvals,rsqr = encoding_regression(fr,behavior_df[['Qdiff_','Qchosen','Choice1','Side','Time']])
            QdiffQch_res = rsqr
            
            # simple regressions
            simple_reg_res = simple_regression_best_rsqr(REGRESSORS,fr,behavior_df)
                
                
            
            df_dict[unit] = [lasso_res, Q1Qch_res, Q1absQdiff_res, Q1Qhigh_res, QdiffQch_res, simple_reg_res]

    col_names = ['Lasso', 'Q1 Qchosen', 'Q1 |Qdiff|', 'Q1 Qhigh', 'Qdiff Qchosen', 'Simple']
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=col_names)
    save_out_csv(df,'RegressionRSqrTable','')
    
    fig,ax = plt.subplots()
    ax.bar(col_names,df.mean(axis=0))
    data = [df[col_names[i]] for i in range(len(col_names))]
    ax.plot(range(len(col_names)),data,'o',alpha=0.1)
    ax.set_ylabel('R-Squared')
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15) #
    
    return df

def lasso_gridsearch():
    
    # tols = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
    # reg_strengths = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    tols = [1e-4,1e-5,1e-6]
    reg_strengths = [1e-7,5e-7,1e-6,5e-6,1e-5,5e-5,1e-4,5e-4]
    ntols = len(tols)
    nregs = len(reg_strengths)
    nsess = len(SESSIONS)
    
    master_regressor_list = ['Q1','Choice1','Qhigh','Qchosen','Qdiff_','absQdiff','Choice_high','Side','Time']
    
    result_matrix = np.zeros((ntols,nregs))

    for s,session in enumerate(SESSIONS):
        Spikes = ProcessSpikes(session)
        units = Spikes.firing_rate_df['Unit_labels'][0]
        Behav = ProcessBehavior(session)
        behavior_df = Behav.behavior_df
        
        print(f'{s+1}/{nsess}')
        
        for i,tol in enumerate(tols):
            for j,reg_strength in enumerate(reg_strengths):
                
                scores=[]
                for unit in units:
                    fr = Spikes.firing_rate_df[unit]
                    score,coefs = lasso_regression(fr,behavior_df[master_regressor_list],reg_strength,tol,verbose=False)
                    scores.append(score)
                result_matrix[i,j] += np.mean(score)
                
    result_matrix = result_matrix / nsess 
    save_out_pkl(result_matrix,'lasso_gridsearch2','')
    
    fig,ax=plt.subplots()
    im=ax.matshow(result_matrix)
    ax.set_xlabel('Regularization Strength')
    ax.set_ylabel('Tolerance')
    ax.set_xticks(range(nregs),reg_strengths)
    ax.set_yticks(range(ntols),tols)
    ax.set_title('Lasso hyperparameter grid search')
    c=fig.colorbar(im)
    c.ax.set_ylabel('R^2')
    
    return


class PCA_Spike_Analysis:
    
    def __init__(self, session: str, brain_area: str, Spikes=None, Behav=None):
        '''
        Can pass in ProcessSpikes and ProcessBehavior objects, otherwise will load them.
        
        '''
        # Get Spike and Behavior Data
        if not Spikes:
            Spikes = ProcessSpikes(session)
        if not Behav:
            Behav = ProcessBehavior(session)
        self.behavior = Behav.behavior_df[REGRESSORS]
        self.area_psth_df = area_parser(Spikes.psth_df,brain_area)
        self.psth_dict = Spikes.psth_dict
        self.session = session
        self.area = brain_area
        
        # print(Spikes.psth_df)
        # print(self.area_psth_df)
        
        # Get Shape and Data information
        self.units = list(self.area_psth_df.columns)
        print(self.units)
        self.n_units = len(self.units)
        if self.n_units < 1:
            print('!'*25,f'\nNo units found for {self.area} for {self.session}!\nSkipping..')
            return
        self.t_vector = Spikes.psth_t_vector
        self.n_timebins = len(self.t_vector)
        self.n_trials = len(self.behavior)
        self.trials = np.arange(self.n_trials)
        assert len(self.area_psth_df) == self.n_timebins*self.n_trials
        self.n_regressors = len(REGRESSORS)
        
        # Fit PCA model on all trials. 
        # This defines the population manifold for the particular area (over the entire session)
        self.pca = PCA()
        self.pca.fit(self.area_psth_df.values) #n_obs x n_components = (n_trials * n_timebins) x n_units
        
        # Determine manifold dimensionality
        self.plot_scree()
        self.n_comp = int(input("How many components to keep? "))
        
        # Find how behavioral regressors temporally vary with the population activity components
        self.betas, self.pvals = self.get_temporal_encoding_pca()
        self.plot_temporal_encoding_pca(self.betas,self.pvals)
        
        # Find population trajectory across timebins, avg across trials
        self.avg_traj = self.get_traj()
        
        return
        
        
    def plot_scree(self):
        
        # Scree plot
        cov = self.pca.get_covariance() #n_components x n_components
        eigvals = np.linalg.eigvals(cov)
        fig,ax=plt.subplots()
        ax.plot(np.arange(len(eigvals))+1,eigvals,'o-')
        ax.set_xlabel('Principal Component #')
        ax.set_ylabel('Variance Explained (eigvals of cov mat)')
        ax.set_title('Scree Plot')
        fig.suptitle(f'{self.session}\n{self.area}, {self.n_units} units')
        fig.tight_layout()
        plt.show()
        
        return
    
    def get_temporal_encoding_pca(self):
        
        temporal_betas = np.zeros((self.n_comp,self.n_regressors,self.n_timebins))
        temporal_pvals = np.zeros_like(temporal_betas)
        for k in range(self.n_timebins):
            
            timebin_data = np.zeros((self.n_trials,self.n_units))
            for i,unit in enumerate(self.units):
                
                timebin_data[:,i] = self.psth_dict[unit][:,k] #for each unit, get timebin k data for all trials
            
            # get the timebin data for principal components
            timebin_data_transformed = self.pca.transform(timebin_data)
            
            # get encoding for this timebin for each dominant component
            for i in range(self.n_comp): #loop through components that we want to keep
                res = encoding_regression(timebin_data_transformed[:,i],self.behavior)
                betas = res.params
                pvals = res.pvalues
                
                for j,regressor in enumerate(REGRESSORS):
                    temporal_betas[i,j,k] = abs(betas[regressor])
                    temporal_pvals[i,j,k] = pvals[regressor]
        
                    
        #Plot a simple regression of largest beta regressor for illustration
        avg_beta = np.mean(temporal_betas,axis=(0,2))
        max_reg = np.argmax(avg_beta)
        simple_regression_plot(REGRESSORS[max_reg],timebin_data_transformed[:,0],self.behavior)
        
        
        return temporal_betas, temporal_pvals #shape: (n_components, n_regressors, n_timebins)
    
    def plot_temporal_encoding_pca(self, temporal_betas, temporal_pvals):
        
        # Plotting: fig per PC
        for i in range(self.n_comp):
            fig,ax = plt.subplots() 
            ax.plot(np.arange(self.n_timebins),np.zeros((self.n_timebins)),'k--')
            for j,regressor in enumerate(REGRESSORS):
                to_label=True # for labels
                
                for k in range(self.n_timebins):
                    
                    if temporal_pvals[i,j,k]<=0.05:
                        color='red' #significant
                    elif temporal_pvals[i,j,k]<0.1:
                        color='blue' #trending
                    else:
                        color='grey' #nonsignificant
                        
                    if to_label:
                        label=regressor
                        to_label=False
                    else:
                        label=''
                        
                    ax.plot(k,temporal_betas[i,j,k],label=label,color=color,marker=MARKERSTYLES[j]) 
                
            ax.set_title(f'Temporal Encoding Dynamics\nPrincipal Component #{i+1}')
            ax.set_xticks(np.arange(self.n_timebins),self.t_vector)
            ax.set_ylabel('Beta coef magn')
            ax.set_xlabel('Time relative to Choice (sec)')
            ax.legend()
            fig.suptitle(f'{self.session}\n{self.area}, {self.n_units} units')
            fig.tight_layout()
            plt.show()
        return
    
    
    def get_traj(self):
        
        n_dims = 3
        # need at least three units for a n_dims-d traj
        if self.n_units < n_dims:
            print(f'Not enough units for a {n_dims}d trjaectory! (Need at least {n_dims})')
            return
                  
        trajs = np.zeros((self.n_timebins,n_dims,self.n_trials))
        for tr in range(self.n_trials):
            
            # Get timebin data for all units for each trial
            tr_data = np.zeros((self.n_timebins,self.n_units))
            for i,unit in enumerate(self.units):
                tr_data[:,i] = self.psth_dict[unit][tr,:] #for each unit, get all timebin data for trial tr
            
            # Transform all timebins for current trial into principal components
            # (principal components over timebins is a trajectory)
            tr_data_transformed = self.pca.transform(tr_data)
            
            # Keep top n_dims comps (to make n_dims-d traj)
            trajs[:,:,tr] = tr_data_transformed[:,:n_dims]
            
        return np.mean(trajs,axis=-1) #avg over trials to get avg trajectory

def plot_trajs(trajs, labels):
    '''
    Parameters
    ----------
    trajs : list of 2D Arrays
        each array is of shape n_timebins x n_components.
    labels : list of labels for each trajectory (e.g. OFC)

    Returns
    -------
    None.

    '''
    
    n_timebins,n_comps = np.shape(trajs[0])
    n_trajs = len(trajs)
    
    # 2D plot
    if n_comps == 2: 
        fig,ax=plt.subplots()
        
        for i,traj,label in zip(range(n_trajs),trajs,labels):
            ax.plot(traj[:,0],traj[:,1],color=COLORS[i],label=label)
        ax.legend()
        ax.set_xlabel('PC #1')
        ax.set_ylabel('PC #2')
        ax.set_title('PCA Trajectory')
        fig.tight_layout()
    
    # 3D plot
    if n_comps >= 3:
        ax = plt.figure().add_subplot(projection='3d')
        
        for i,traj,label in zip(range(n_trajs),trajs,labels):
            ax.plot(traj[:,0],traj[:,1],traj[:,2],color=COLORS[i],label=label)
        ax.legend()
        ax.set_xlabel('PC #1')
        ax.set_ylabel('PC #2')
        ax.set_zlabel('PC #3')
        ax.set_title('PCA Trajectory')
        
        elev = 25
        azim = 50
        roll = 0
        ax.view_init(elev, azim, roll)
        # fig.tight_layout()

    return
 

def plot_trajs_gif(trajs, labels):
    '''
    Parameters
    ----------
    trajs : list of 2D Arrays
        each array is of shape n_timebins x n_components.
    labels : list of labels for each trajectory (e.g. OFC)

    Returns
    -------
    None.

    '''
    
    n_timebins,n_comps = np.shape(trajs[0])
    n_trajs = len(trajs)
    
    # 2D plot
    if n_comps == 2: 
        fig,ax=plt.subplots()
        
        for i,traj,label in zip(range(n_trajs),trajs,labels):
            ax.plot(traj[:,0],traj[:,1],color=COLORS[i],label=label)
        ax.legend()
        ax.set_xlabel('PC #1')
        ax.set_ylabel('PC #2')
        ax.set_title('PCA Trajectory')
        fig.tight_layout()
    
    # 3D plot
    if n_comps >= 3:     
        
        @gif.frame
        def plot_trajs_frame(k):
            fig = plt.figure(figsize=(5, 3), dpi=100)
            ax = fig.add_subplot(projection="3d")
            
            xmins,xmaxs = [], []
            ymins,ymaxs = [], []
            zmins,zmaxs = [], []
            for i,traj,label in zip(range(n_trajs),trajs,labels):
                x = traj[:,0]
                y = traj[:,1]
                z = traj[:,2]
                # ax.plot(x[:k], y[:k], z[:k], lw=2, color=COLORS[i], label=label)
                ax.plot(x,y,z, lw=1.5, color=COLORS[i], label=label)
                
                elev = 30
                azim = k*2
                roll = 0
                ax.view_init(elev, azim, roll)
            
                xmins.append(min(x))
                xmaxs.append(max(x))
                ymins.append(min(y))
                ymaxs.append(max(y))
                zmins.append(min(z))
                zmaxs.append(max(z))
            
            ax.set_xlim(min(xmins), max(xmaxs))
            ax.set_ylim(min(ymins), max(ymaxs))
            ax.set_zlim(min(zmins), max(zmaxs))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.legend()
            ax.set_xlabel('PC #1')
            ax.set_ylabel('PC #2')
            ax.set_zlabel('PC #3')
            ax.set_title('PCA Trajectory')
            # fig.tight_layout()
        
        
        frames = []
        # for k in range(n_timebins):
        for k in range(180):
            frame = plot_trajs_frame(k)
            frames.append(frame)
        
        gif.save(frames, "Figures/trajs.gif", duration=10)

    return           


def do_PCA(): #turn into a class so we can keep the data after it runs
    
    for i,session in enumerate(SESSIONS):
        
        # pval_dict = {}
        # pval_dict['Session'] = session
        
        Spikes = ProcessSpikes(session)
        Behav = ProcessBehavior(session)
        behavior = Behav.behavior_df[REGRESSORS]
        units = list(Spikes.psth_df.columns)
        n_units = len(units)
        print(Spikes.psth_df)
        
        
    
        pca = PCA()
        X_new = pca.fit_transform(Spikes.psth_df) #n_obs x n_components = (n_trials * n_timebins) x n_units
        
        # Scree plot
        cov = pca.get_covariance() #n_components x n_components
        eigvals = np.linalg.eigvals(cov)
        fig,ax=plt.subplots()
        ax.plot(np.arange(len(eigvals))+1,eigvals,'o-')
        ax.set_xlabel('Principal Component #')
        ax.set_ylabel('Variance Explained (eigvals of cov mat)')
        ax.set_title('Scree Plot')
        fig.suptitle(session)
        fig.tight_layout()
        plt.show()
        
        n_comp = int(input("How many components to keep? "))
        # X_new_trimmed = X_new[:,:n_comp]
        
        
        # Temporal Encoding Regression Analysis
        n_timebins = Spikes.psth_dict[units[0]].shape[1]
        n_trials = len(behavior)
        trials = np.arange(n_trials)
        n_regressors = len(REGRESSORS)
        assert n_trials*n_timebins == len(Spikes.psth_df)
        
        
        
        #get specific timebin data for each trial
        temporal_betas = np.zeros((n_comp,n_regressors,n_timebins))
        temporal_pvals = np.zeros_like(temporal_betas)
        for k in range(n_timebins):
            
            timebin_data = np.zeros((n_trials,n_units))
            for i,unit in enumerate(units):
                
                timebin_data[:,i] = Spikes.psth_dict[unit][:,k] #for each unit, get timebin k data for all trials
            
            # get the timebin data for principal components
            timebin_data_transformed = pca.transform(timebin_data)
            
            # get encoding for this timebin for each dominant component
            for i in range(n_comp): #loop through components that we want to keep
                res = encoding_regression(timebin_data[:,i],behavior)
                betas = res.params
                pvals = res.pvalues
                
                for j,regressor in enumerate(REGRESSORS):
                    temporal_betas[i,j,k] = abs(betas[regressor])
                    temporal_pvals[i,j,k] = pvals[regressor]
        
        
        # Plotting: fig per PC
        for i in range(n_comp):
            fig,ax = plt.subplots() 
            ax.plot(np.arange(n_timebins),np.zeros((n_timebins)),'k--')
            for j,regressor in enumerate(REGRESSORS):
                to_label=True # for labels
                
                for k in range(n_timebins):
                    
                    if temporal_pvals[i,j,k]<=0.05:
                        color='red' #significant
                    elif temporal_pvals[i,j,k]<0.1:
                        color='blue' #trending
                    else:
                        color='grey' #nonsignificant
                        
                    if to_label:
                        label=regressor
                        to_label=False
                    else:
                        label=''
                        
                    ax.plot(k,temporal_betas[i,j,k],label=label,color=color,marker=MARKERSTYLES[j]) 
                
            ax.set_title(f'Temporal Encoding Dynamics\nPrincipal Component #{i+1}')
            ax.set_xticks(np.arange(n_timebins),Spikes.psth_t_vector)
            ax.set_ylabel('Beta coef magn')
            ax.set_xlabel('Time relative to Choice (sec)')
            ax.legend()
            fig.suptitle(session)
            fig.tight_layout()
            plt.show()

            
                    
            # ## Save out
            # timebin_labels = [f'Timebin_{t}' for t in range(n_timebins)]
            # pval_df = pd.DataFrame(temporal_pvals,columns=timebin_labels)
            # pval_df['Principal Component #'] = [f'PC {comp+1}' for comp in range(n_comp)]
            # pval_dict[f'{regressor}_pval_df'] = pval_df
            
        
        
        # encoding_dict = {}
        # for comp in range(n_comp):
        #     fr = X_new[:,comp]
                     
        #     # print(behavior.shape)
        #     # print(fr.shape)
        #     score,coefs = lasso_regression(fr,behavior,verbose = False)
        #     encoding = lasso_feature_selection(coefs,REGRESSORS)
        #     encodings_dict[comp] = encoding



def do_PCA2():
    
    list_of_objs=[]
    
    areas = ['all areas','OFC','vmPFC','Cd']
    
    for session in SESSIONS:
        
        Spikes = ProcessSpikes(session, verbose=False)
        Behav = ProcessBehavior(session)
        
        for area in areas:
            
            pca_analysis = PCA_Spike_Analysis(session,area,Spikes,Behav)
            
            list_of_objs.append(pca_analysis)
            
    return list_of_objs


def do_PCA_trajs():
    
    sessions = ["airp20251029_05_te2231"]#,"airp20251023_03_te2219","airp20251023_03_te2219"]
    areas =    ["OFC"]#,"vmPFC","Cd"]
    
    trajs = []
    labels = []
    for session,area in zip(sessions,areas):
        
        pca_ = PCA_Spike_Analysis(session,area)
        
        trajs.append(pca_.avg_traj)
        labels.append(area)
    
    
    plot_trajs_gif(trajs,labels)
    
    return trajs,labels

    
#### Run

## value distribution        
# sess = "airp20251111_02_te2250"
# b = ProcessBehavior(sess)
# b.get_behavior(Q_learning=True)
# b.plot_value_dist()

## temporal encoding
# session = "braz20240927_01_te5384"
# Spikes = ProcessSpikes(session)
# Behav = ProcessBehavior(session)
# aaa = TemporalEncodingRegressions(Spikes.dict_out,Behav.dict_out)



# dfs_out = []
# for session in SESSIONS:
#     dfs_out.append(load_learning_rate_df(session))
#     print(f'{session} loaded!\n\n')

# plot_learning_rate_change(get_learning_rate_change(merge_sessions_df(dfs_out)))

    
# for session in SESSIONS:
#     Behav = ProcessBehavior(session)
#     # Behav.check_behavior()
#     Behav.plot_values()