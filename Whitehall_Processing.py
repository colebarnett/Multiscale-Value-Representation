# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:26:52 2026

@author: coleb
"""
#### Libraries, Paths, and Imports
import os
import sys
# import gif
# import copy
import time
# import math
# import pickle
# import sklearn
# import neurodsp
# import matplotlib
import numpy as np
import scipy as sp
import pandas as pd
# import matplotlib_venn
# import statsmodels.api as sm
# import statsmodels.formula.api as sm_api
# import matplotlib.lines as mlines
from matplotlib import pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import BoundaryNorm
# from numpy.polynomial.polynomial import polyfit

import Whitehall_utils as utils
from file_info import get_block_info, get_area_info
import DecisionMakingBehavior_Whitehall as BehaviorAnalysis
# from Whitehall_Processing import ProcessSpikes,ProcessBehavior,load_data
from Whitehall_Globals import (SUBSESSIONS,SESSIONS,
PROJ_FOLDER,BMI_FOLDER,DATA_FOLDER,NS_FOLDER,NSX_FOLDER,
REGRESSOR_SUPERSET_DICT,ALL_REGRESSORS,FREQ_BANDS,AREAS,REW_PROBS,
VALUE_MODEL,ALIGNMENT,HOLD_DURATION,ENCODING_WINDOW,
T_BEFORE,T_AFTER,T_BOXCAR_SIZE,T_BOXCAR_STEP,T_VECTOR,N_TIMEBINS,MIN_FR_THRESH,MIN_NUM_TRIALS,
LEARNING_RATES,ALPHA_THRESHOLD,
COLORS,LINESTYLES,MARKERSTYLES,HATCHSTYLES)

from RishiValueModeling import ValueModelingClass
from TwoStageQLearning_DualBaselineSide import Calc_DistrQlearning_2Targs_DualBaselineSide
from TwoStageQLearning_BaselineSide import Calc_DistrQlearning_2Targs_BaselineSide

# Add paths for necessary packages to search path
sys.path.insert(1,BMI_FOLDER) 
sys.path.insert(2, NS_FOLDER) 
sys.path.insert(3,os.path.join(PROJ_FOLDER,'utils'))
# from riglib.blackrock.brpylib import NsxFile
from nsfile import NSFile,NsxFile

#%% Main

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
        self.psth_dict = None #not zscored
        self.psth_df = None #zscored
        self.excluded_units = []
        # self.psth_t_vector = T_VECTOR
        # self.alignment = ALIGNMENT
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
        # if TEST: unit_idxs=unit_idxs[:3] #to make runtime shorter for testing
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
    

    def apply_inclusion_criteria(self, fr):
        #fr = np.mean(psth[:,win],axis=1) = shape: 1xnum_trial
        
        if self.verbose:
            print(f'Hold avg FR: {np.mean(fr):.2f} Hz')
            print(f'Hold med FR: {np.median(fr):.8f} Hz')
            print(f'Num holds with low fr: {sum(fr<0.1)}')
        
        # overall avg and median FR of holds
        if np.mean(fr)>=MIN_FR_THRESH and np.median(fr)>0.01:
            return True
        else:
            if self.verbose: print('EXCLUDED')
            return False
        
        if self.verbose: print('\n')
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
        # win = (T_VECTOR>=ENCODING_WINDOW[0]) * (T_VECTOR<=ENCODING_WINDOW[1]) #to get num_spikes only over defined window
        
        for unit,unit_label in enumerate(self.unit_labels):
            
            unit_spikes = np.array(self.spike_times[unit])
            # psth = self.compute_psth(unit_spikes) #psth = (num_trials x num_timebins)
            # fr = np.mean(psth[:,win],axis=1) # shape = 1 x n_trials
            psth = self.count_spikes_boxcar(unit_spikes) #psth = (num_trials x num_timebins)
            fr = self.count_spikes_in_time_window(unit_spikes,win_begin=ENCODING_WINDOW[0],win_end=ENCODING_WINDOW[1]) # shape = 1 x n_trials
            
            #see if unit passes inclusion criteria
            if self.verbose:
                print(unit_label)
                print(f'Overall avg FR: {len(unit_spikes)/(unit_spikes[-1]-unit_spikes[0]):.2f} Hz')
            include_unit_flag = self.apply_inclusion_criteria(fr)
            
            if self.verbose:
                fig,axs=plt.subplots(2,1,layout="constrained")
                ax=axs[0]#fr vs trials
                ax.plot(fr)
                ax.set_xlabel('Trial')
                ax.set_ylabel('Hold center FR')
                ax.set_title('FR vs time')
                ax=axs[1] #hist
                ax.hist(fr,bins=np.arange(0,10,0.1))
                ax.set_xlabel('Hold center FR')
                ax.set_ylabel('Num trials')
                ax.set_title(f'FR Hist, Avg={np.mean(fr):.2f}, Med={np.median(fr):.2f}')
                fig.suptitle(f'{unit_label}\nIncluded: {include_unit_flag}')
                
            #only include current unit if it passed inclusion criteria
            if include_unit_flag:
                # self.psth_dict[unit_label] = sp.stats.zscore(psth,axis=0)
                self.psth_dict[unit_label] = psth #raw fr
                psth_df_dict[unit_label] = np.ravel(sp.stats.zscore(psth,axis=0))
                fr_df_dict[unit_label] = sp.stats.zscore(fr) # shape = 1 x n_trials
                raw_fr_df_dict[unit_label] = fr # shape = 1 x n_trials
                assert fr_df_dict[unit_label].shape[0] == self.num_trials
            else:
                self.excluded_units.append(unit_label)
            
        
            
        if self.verbose:
            print('Units excluded:')
            print(self.excluded_units)
            print('-'*20)
            
        print(f'{len(self.excluded_units)}/{self.num_units} units excluded using mean and median.')
        print('Done making PSTHs!')
        
        # self.psth_dict: keys=unit_labels, values=psth arrays (trials x timepoints)
        
        self.psth_df = pd.DataFrame.from_dict(psth_df_dict,orient='columns') #df of size (n_trials * n_timepoints) x n_units
        self.fr_df = pd.DataFrame.from_dict(fr_df_dict,orient='columns') #df of size n_trials x n_units
        self.raw_fr_df = pd.DataFrame.from_dict(raw_fr_df_dict,orient='columns') #df of size n_trials x n_units
                
        return   
    
    
    # def compute_psth(self,unit_spikes):
    #     '''
    #     Method that returns an array of psths for spiking activity aligned to the sample numbers indicated in samples_align
    #     with firing rates quantized to bins of size samp_resolution.

    #     Input:
    #     - unit_spikes: arr of all spike times

    #     Output: 
    #     - psth: T x N array containing the average firing rate over a window of total length N samples for T different
    #             time points (trials)
    #     '''        
        
    #     # Add 0.5 sec both before and after in order to incorporate spiking info on either side of the actual window 
    #     # for smoothing purposes. PSTH is trimmed back down to proper length after smoothing
    #     extra_t_before = 0.5
    #     extra_t_after = 0.5
    #     psth_length = int(np.rint((T_BEFORE + T_AFTER + extra_t_before + extra_t_after)/T_RES))
    #     num_timepoints = len(self.times_align)
    #     psth = np.zeros((num_timepoints, psth_length-1))
    #     # psth = np.zeros((num_timepoints, psth_length))
    #     smooth_psth = np.zeros(psth.shape)

    #     #set smoothing parameters
    #     # boxcar_length = 5
    #     # boxcar_window = boxcar(boxcar_length)  # 2 bins before, 2 bins after for boxcar smoothing
    #     b = sp.signal.windows.gaussian(39, 1)

    #     for i, tp in enumerate(self.times_align):
    #         data = unit_spikes
    #         t_window = np.arange(tp - T_BEFORE - extra_t_before, tp + T_AFTER + extra_t_after, T_RES)
    #         hist, bins = np.histogram(data, bins = t_window)
    #         hist_fr = hist/T_RES
    #         psth[i,:] = hist_fr[:psth_length-1]
    #         # psth[i,:] = hist_fr[:psth_length]
    #         #smooth_psth[i,:] = np.convolve(hist_fr[:psth_length-1], boxcar_window,mode='same')/boxcar_length
    #         smooth_psth[i,:] = sp.ndimage.convolve1d(hist_fr[:psth_length-1], b/b.sum())
    #         # smooth_psth[i,:] = sp.ndimage.convolve1d(hist_fr[:psth_length], b/b.sum())
       
    #     extra_time_bins_before = int(np.rint(extra_t_before / T_RES))
    #     extra_time_bins_after = int(np.rint(extra_t_after / T_RES))
        
    #     smooth_psth_out = smooth_psth[:, extra_time_bins_before:-extra_time_bins_after]
    #     assert len(T_VECTOR) == np.shape(smooth_psth_out)[1], f'{len(T_VECTOR)} vs {np.shape(smooth_psth_out)[1]}'
        
    #     return smooth_psth_out
    
    def count_spikes_boxcar(self,unit_spike_times):
        
        num_trials = len(self.times_align)
        psth = np.zeros((num_trials,N_TIMEBINS))
        
        for i, tp in enumerate(self.times_align):
            for j in range(N_TIMEBINS):
                
                #define boxcar edges
                begin_edge = tp - T_BEFORE + j*T_BOXCAR_STEP
                end_edge = begin_edge + T_BOXCAR_SIZE
                
                #count spikes within boxcar
                psth[i,j] = sum((unit_spike_times>begin_edge) & (unit_spike_times<end_edge)) / T_BOXCAR_SIZE
        
        return psth
    
    def count_spikes_in_time_window(self,unit_spike_times,win_begin,win_end):
        # win_begin and win_end are defined in relation to the time alignment point and must be in seconds
        # e.g. -0.2, 0.6
        
        num_trials = len(self.times_align)
        fr = np.zeros((num_trials))
        
        for i, tp in enumerate(self.times_align):
            
            #count spikes within boxcar
            fr[i] = sum((unit_spike_times>tp+win_begin) & (unit_spike_times<tp+win_end)) / (win_end - win_begin)
        
        return fr






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
        rew_probs_2, rew_probs_1 = behavior.reward_probabilityH, behavior.reward_probabilityL
        
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
            
        
        # construct State variable (-1 for stab, +1 for vol)
        state = np.where(is_stable_block,-1,1)
        
        
        # Calculate value of targets throughout trial
        if Q_learning and learning_rates is not None: #to do a specific learning rate (uses modified version of Rishi's method)
            beta=1 #beta doesn't matter since we are only interested in value here
            kappa=1 #same for kappa ^^
            Q0 = 0.5*np.ones(2)
            # instructed_or_freechoice = np.full_like(choices,2) #all whitehall trials are free choice
            if len(learning_rates)==2: #dual
                parameters = [learning_rates[0],learning_rates[1],beta,kappa]
                # Q1,Q2,_,_,_,_ = BehaviorAnalysis.Calc_DistrQlearning_2Targs_DualBaseline(learning_rates[0],learning_rates[1],beta,Q0,choices,rewards,instructed_or_freechoice)
                Q1,Q2,_,_,_,_ = Calc_DistrQlearning_2Targs_DualBaselineSide(parameters, Q0, choices, rewards, instructed_or_freechoice, choice_side_raw)
            elif len(learning_rates)==1: #single
                parameters = [learning_rates[0],beta,kappa]
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
        Q1_choices = np.where(choices==1,1,0)
        Q2_choices = np.where(choices==2,1,0)
        choices_targ = np.where(choices==1,1,-1)
        
        
        # make time regressor 
        time = np.linspace(0,1,len(choices))
        log_time = np.log(time+1)
        
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
        
        # reward of prev trial, relevant for choice epoch
        # prev reward for first trial doesnt exist, so I put 0.
        # also need to chop off last elemnt from array to make lengths the same
        # also make -1/1
        prev_reward_temp =  np.concat([np.array([0]),rewards[:-1]])    
        prev_reward = np.where(prev_reward_temp,1,-1)
                    
        # Split Q1 and Q2 into Qlow and Qhigh. Also do for choices
        Q1_is_low = Q1 < Q2
        chose_Q1 = np.asarray(Q1_choices) == 1
        
        Qlow = np.where(Q1_is_low, Q1, Q2)
        Qhigh = np.where(Q1_is_low, Q2, Q1)
        
        # Low option was chosen when "Q1 is low" and "Q1 was chosen" agree
        chose_low = Q1_is_low == chose_Q1
        Qlow_choices = chose_low.astype(Q1.dtype)
        Qhigh_choices = (~chose_low).astype(Q2.dtype)
        choices_high = np.where(Qhigh_choices,1,-1)
        

        # Split Q1 and Q2 into Qchosen and Qunchosen            
        Q_chosen = np.where(chose_Q1, Q1, Q2)
        Q_unchosen = np.where(chose_Q1, Q2, Q1)
        
        #assign QL and QR
        left_choice = choice_side == 1
        Q_L = np.where(left_choice,Q_chosen,Q_unchosen)
        Q_R = np.where(left_choice,Q_unchosen,Q_chosen)

        assert len(choices) == len(choice_side) == len(time) == len(Q1) == len(Q2) == len(rpe) == len(is_stable_block) == len(rew_probs_1) == len(prev_reward)

        
        self.behavior_df = pd.DataFrame.from_dict(
            {'Q1':Q1,       'Q2':Q2,        'Choice1':Q1_choices,     'Choice2':Q2_choices, 
             'Qlow':Qlow,   'Qhigh':Qhigh,  'Qdiff_':Qdiff,  'absQdiff':absQdiff,
             'Qchosen':Q_chosen,            'Qunchosen':Q_unchosen,
             'Qright':Q_R,                  'Qleft':Q_L,
             'Choice_low':Qlow_choices,  'Choice_high':Qhigh_choices,
             'Reward':rewards, 'Prev_Reward':prev_reward, 
             'Prev_Rew ⊥ Qch':utils.residualize(prev_reward,Q_chosen),
             'Prev_Rew ⊥ Q1':utils.residualize(prev_reward,Q1),
             'Prev_Rew ⊥ Q2':utils.residualize(prev_reward,Q2),
             'Qch ⊥ Prev_Rew':utils.residualize(Q_chosen,prev_reward),
             'RPE':rpe, 'absRPE':abs(rpe), 
             'ChoiceSide':choice_side, 'ChoiceTarg':choices_targ, 'ChoiceExploit':choices_high,
             'ChoiceExploit ⊥ Qch':utils.residualize(choices_high,Q_chosen),
             'ChoiceExploit ⊥ Q1':utils.residualize(choices_high,Q1),
             'ChoiceExploit ⊥ Q2':utils.residualize(choices_high,Q2),
             'Time':time, 'Log_Time':log_time,
             'Rew_prob_1':rew_probs_1, 'Rew_prob_2':rew_probs_2,
             'Stable':is_stable_block, 'Volatile':is_volatile_block, 
             'State':state}) #, 'Stim':stim_holds})
        
        if self.subsession == 'all trials':
            self.behavior_df['State ⊥ Time'] = utils.residualize(state,time)
        
        self.num_trials = len(choices)
        
        if not Q_learning:
            print('Behavior loaded!')
            
        if self.behavior_df.isna().values.any():
            print('!'*20,f'\nNans found in behavior df for {self.session}',self.behavior_df)
            

        return
    
    def get_learning_rate_grid_behavior(self,learning_rate_grid):
        
        
        
        if 'Dual' in VALUE_MODEL:
            
            dict_out = {'dict layout':'(pos alpha,neg alpha) : behavior_df',
                        'learning rate grid':learning_rate_grid,
                        'data': dict() }
        
            for pos_alpha in learning_rate_grid:
                for neg_alpha in learning_rate_grid:
                    
                    # get beahvior according to current learning rates
                    self.get_behavior(Q_learning=True, learning_rates=[pos_alpha,neg_alpha])
                    
                    dict_out['data'][(np.round(pos_alpha,2),np.round(neg_alpha,2))] = self.behavior_df
                    
        if 'Dual' not in VALUE_MODEL:
            
            dict_out = {'dict layout':'(alpha) : behavior_df',
                        'learning rate grid':learning_rate_grid,
                        'data': dict() }
        
            for alpha in learning_rate_grid:
                
                # get beahvior according to current learning rates
                self.get_behavior(Q_learning=True, learning_rates=[alpha])
                
                dict_out['data'][(np.round(alpha,2))] = self.behavior_df
                
                
        utils.save_pkl(dict_out,'LR_Grid_behavior_dfs_dict',self.session,DATA_FOLDER)
                
        return 
    
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


def units_totals_new():
    
    areas = ('Cd','OFC')
    totals = {}
    sortd=[]
    incl=[]
    for area,area_code in zip(areas,['Unit C','Unit D']):
        
        s_counter=0
        i_counter=0
        for s,session in enumerate(SESSIONS):
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            sorted_spikes = Spikes.unit_labels
            incl_spikes = Spikes.fr_df.columns
            
            # sortd.append(len([sp for sp in sorted_spikes if area_code in sp]))
            # incl.append(len([sp for sp in incl_spikes if area_code in sp]))
            # val.append(len([sp for sp in value_spikes if area_code in sp]))
            s_counter+=len([sp for sp in sorted_spikes if area_code in sp])
            i_counter+=len([sp for sp in incl_spikes if area_code in sp])
        
        sortd.append(s_counter)
        incl.append(i_counter)
    
    totals={'Sorted': sortd,
            'Included': incl}
    
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
    
    return


def spike_inclusion_sanity():
    
    session = "braz20260729_03_te2539"
    _ = ProcessSpikes(session,True)
    
    # for session in SESSIONS[:3]:
    #     _=ProcessSpikes(session,True)
    
    return


def load_data(overwrite_spikes=False,overwrite_lfp=False,load_behav=True,load_spikes=True,load_lfp=False,verbose=False):
    
    sp_list=[]
    be_list=[]
    lfp_list=[]
    for s,session in enumerate(SESSIONS):
        
        start = time.time()
        print(session)
        
        
        ## Behavior
        if load_behav:
            Behav = ProcessBehavior(session,'all trials')
            assert np.all(np.isclose(np.unique(Behav.behavior_df['Rew_prob_1']),REW_PROBS)), f'Not all rew probs present! {session} {np.unique(Behav.behavior_df['Rew_prob_1'])}'
            
            #Apply session inclusion criteria
            # assert Behav.num_trials > MIN_NUM_TRIALS, f'Too few trials! Skipping! {session}'
            be_list.append(Behav)                
        else: 
            be_list.append('')
            
            
        ## Spikes
        if load_spikes:
            if utils.does_pkl_exist('Spikes', session,DATA_FOLDER) and not overwrite_spikes:  
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
        if load_lfp:
            if utils.does_pkl_exist('LFP', session,DATA_FOLDER) and not overwrite_lfp:  
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

#%% Load Data

SPIKES_LIST,BEHAV_LIST,LFP_LIST = load_data(
                                            overwrite_spikes_flag=True,
                                            overwrite_lfp_flag=False,
                                            load_behav=True,
                                            load_spikes=True,
                                            load_flp=False,
                                            verbose=False
                                            )


def get_Spikes_Behav_LFP(s: int, session: str):
    # s is idx counter representing the number session to load
    # session is the str name of the session to load: e.g. "braz20260728_03_te2536""
    assert SESSIONS[s] == session
    Spikes = SPIKES_LIST[s]
    Behav = BEHAV_LIST[s]
    LFP = LFP_LIST[s]
    return Spikes, Behav, LFP

#### Control Analyses


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
