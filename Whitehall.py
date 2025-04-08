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

#### Define global things

import os
import sys
import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
# from matplotlib_venn import venn2,venn3

import DecisionMakingBehavior_Whitehall as BehaviorAnalysis






# Constants




# Paths
PROJECT_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis"
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
# PROJECT_FOLDER = r"F:\cole"
# BMI_FOLDER = r"C:\Users\crb4972\Desktop\bmi_python"

NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
# if not (BMI_FOLDER in sys.path):
sys.path.insert(1,BMI_FOLDER) #add bmi_python folder to package search path
# if not (BMI_FOLDER in sys.path):
sys.path.insert(2, NS_FOLDER) #add neuroshare python folder to package search path

# os.chdir(BMI_FOLDER)
# import riglib.ripple.pyns.pyns.nsparser
# from riglib.blackrock.brpylib import NsxFile
# from riglib.ripple.pyns.pyns.nsparser import ParserFactory
# from riglib.ripple.pyns.pyns.nsexceptions import NeuroshareError, NSReturnTypes
# from riglib.ripple.pyns.pyns.nsentity import AnalogEntity, SegmentEntity, EntityType, EventEntity, NeuralEntity

# os.chdir(NS_FOLDER)
from nsfile import NSFile
    
os.chdir(PROJECT_FOLDER)



# Channel keys
# something here about which chs are which area
# just do ^^ with ^^ good_chans file?



class ProcessSpikes:
    '''
    From .mat, .hdf, and .nev files, load spike times and calculate firing rate 
    aligned to task events.
    
    Output: DataFrame with firing rate for each unit for each trial
    '''
    
    def __init__(self, session: str):
        

        
        self.session = session
        self.file_prefix = os.path.join(PROJECT_FOLDER, self.session)
        
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
        
        ## Get Times Align
        self.times_align = None
        self.get_times_align()
        
        ## Get Spike Times
        self.spike_times = None
        self.get_spike_times()
        
        ## Get Firing Rates
        self.firing_rate_df = None
        self.t_before = 0.2 #how far to look before time_align point [s]
        self.t_after = 0.0 #how far to look after time_align point [s]
        self.get_firing_rate() 
        
        ## Save Out Processed Data
        # self.df = pd.concat([self.behavior_df,self.firing_rate_df],axis='columns') #merge behavior_df and firing_rate_df
        # self.dict_out = {'Name':session, 'df':self.df}
        self.dict_out = {'Name':session, 'df':self.firing_rate_df}
        
        

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
#         ind_hold_center = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
#         ind_mvmt_period = cb.ind_check_reward_states - 3 #times corresponding to target prompt
#         ind_reward_period = cb.ind_check_reward_states #times corresponding to reward period onset
        ind_target_hold = cb.ind_check_reward_states - 2 # times corresponding to target hold onset
        
        # align spike tdt times with hold center hdf indices using syncHDF files
        target_hold_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind_target_hold,cb.state_time,self.syncHDF_file)

        # Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
        assert len(target_hold_TDT_ind) == len(ind_target_hold), f'Repeat hold times! Session: {self.session}'
        assert len(target_hold_TDT_ind) == self.num_trials

        self.times_align = target_hold_TDT_ind / DIO_freq
        
        
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
        # unit_idxs=unit_idxs[:3] #to make runtime shorter for testing
        self.num_units = len(unit_idxs)
        self.unit_labels = [b"Unit " + h[b'NEUEVLBL'].label[:7] for h in headers[unit_idxs] ] #get labels of all sorted units
        num_units_per_idx = [h[b'NEUEVWAV'].number_sorted_units for h in headers[unit_idxs]]
        recording_duration = self.nevfile.get_file_info().time_span # [sec]


        self.spike_times = [] #each element is a list spike times for a sorted unit
        spike_waveforms = [] #each element is a list of waveforms for a sorted unit
        for i,unit_idx in enumerate(unit_idxs): #loop thru sorted unit
            unit = spike_entities[unit_idx]
            self.spike_times.append([]) #initiate list of spike times for this unit
            
            for spike_idx in range(unit.item_count):
                self.spike_times[i].append(unit.get_segment_data(spike_idx)[0])
#                 spike_waveforms.append(unit.get_segment_data(spike_idx)[1])

            print(f'{self.unit_labels[i]}: {unit.item_count} spikes. Avg FR: {unit.item_count/recording_duration:.2f} Hz. ({i+1}/{len(unit_idxs)})')
        print('All spike times loaded!')
        
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

        self.firing_rate_df = pd.DataFrame(firing_rates,columns=self.unit_labels)
        self.firing_rate_df['Trial'] = np.arange(self.num_trials)+1
        self.firing_rate_df['Unit_labels'] = [self.unit_labels for i in range(self.num_trials)] 
                
        return 
            
        





class ProcessBehavior:
    '''
    From .hdf file, load task and behavior details and calculate value with
    either Q-Learning or empirical-based methods.
    
    Output: DataFrame with behavioral details for regression.
    '''
    
    def __init__(self, session: str):
        

        
        self.session = session
        self.file_prefix = os.path.join(PROJECT_FOLDER, self.session)
        
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
        self.dict_out = {'Name':session, 'df':self.behavior_df}
        
        

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


        
        
        
        
    def get_behavior(self,Q_learning=False,learning_rate=None):
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
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'        
        
        #load behavior over all hdf files at once for behavioral data section
        behavior = BehaviorAnalysis.ChoiceBehavior_Whitehall([self.hdf_file]) #method needs hdf filenames in a list
        
        # get choices, rewards, and trial type
        choices,rewards = behavior.GetChoicesAndRewards()
        L_choices = np.zeros_like(choices)
        L_choices[choices==1] = 1 #1 if he chose LV, 0 otherwise
        H_choices = np.zeros_like(choices)
        H_choices[choices==2] = 1 #1 if he chose HV, 0 otherwise
        
        # make time regressor 
        time = np.arange(len(choices))
        time = np.log(time+1)
        time -= np.mean(time) # to make centered at 0
        
        # calculate value of targets throughout trials
        if Q_learning:
            values = BehaviorAnalysis.CalcValue_2Targs_QLearning(choices,rewards,learning_rate)
        else:
            values, _ = behavior.CalcValue_2Targs(choices,rewards,win_sz=10,smooth=True)
        
        # calculate RPE: rpe(t) = r(t) - Q(t). Note: Q just means value. Doesn't have to use q-learning specifically.
        rpe=[]
        for trial,choice in enumerate(choices):
            if choice==1: #lv choice
                rpe.append(rewards[trial] - values[0][trial]) #lv value
            elif choice==2: #hv choice
                rpe.append(rewards[trial] - values[1][trial]) #hv value
        rpe = np.transpose(np.array(rpe))
        
        # #split rpe into positive and negative rpe signals
        # pos_rpe = copy.deepcopy(rpe) #use deep copy so I can change one without changing the other
        # neg_rpe = copy.deepcopy(rpe)
        # pos_rpe[pos_rpe<0] = 0 #for positive rpe, set all negative rpes to zero
        # neg_rpe[neg_rpe>0] = 0 #for negative rpe, set all positive rpes to zero        
            
                
        # get which side (left or right) each choice was
        choice_side = behavior.GetTargetSideSelection()
        choice_side = ((choice_side-0.5)*2).astype(int) #make to be -1 or 1
        
#         # get which center holds were stimulated
#         stim_holds= (choices==1) * (instructed_or_freechoice==1) #get forced LV holds
#         stim_holds[:100] = False #only want blB and blAp holds
#         #print(stim_holds)
#         for i,el in enumerate(stim_holds):
#             if el:
#                 stim_holds[i]=1
#             else:
#                 stim_holds[i]=0


        assert len(choices) == len(choice_side) == len(time) == len(values[0]) == len(values[1]) == len(rpe)        
        
        self.behavior_df = pd.DataFrame.from_dict(
            {'Choice1':L_choices, 'Choice2':H_choices, 'Side':choice_side, 'Q1':values[0], 'Q2':values[1],
            'Reward':rewards, 'RPE':rpe, 'Time':time}) #, 'Stim':stim_holds})
        
        self.num_trials = len(choices)
        
        if not Q_learning:
            print('Behavior loaded!')

        
        return
        
    
    def check_behavior(self):
        
        assert self.has_hdf, FileNotFoundError(f'.hdf file not found! Session: {self.session}')
        
        self.hdf_file = self.file_prefix + '.hdf'        
        
        #load behavior over all hdf files at once for behavioral data section
        behavior = BehaviorAnalysis.ChoiceBehavior_Whitehall([self.hdf_file]) #method needs hdf filenames in a list
        
        # get rxn times
        behavior.GetRxnTime_ToggleSwitch()
        return
    
    
    def plot_choices(self):
        
        
        ## choice behavior and rewards figure for Fig 1
        #smooth choice data 
        window_length=50
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
#         P_MV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_M'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)

        fig,ax = plt.subplots()

        ax.plot(P_LV,color='red',label='Target 1')
#         ax.plot(P_MV,color='orange',label='MV Target')
        ax.plot(P_HV,color='blue',label='Target 2')

        #Get reward data for each choice
        LV_rew = np.nonzero((self.behavior_df['Choice1']==1) & (self.behavior_df['Reward']==1))
        LV_unrew = np.nonzero((self.behavior_df['Choice1']==1) & (self.behavior_df['Reward']==0))

#         MV_rew = np.nonzero((behavior_dict['Choice_M']==1) & (behavior_dict['Rewarded']==1))
#         MV_unrew = np.nonzero((behavior_dict['Choice_M']==1) & (behavior_dict['Rewarded']==0))

        HV_rew = np.nonzero((self.behavior_df['Choice1']!=1) & (self.behavior_df['Reward']==1))
        HV_unrew = np.nonzero((self.behavior_df['Choice1']!=1) & (self.behavior_df['Reward']==0))

        ymin_hi, ymax_hi = 1.1, 1.2 #for rewarded trials
        ymin_lo, ymax_lo = -0.2, -0.1 #for unrewarded trials

        ax.vlines(LV_rew,ymin_hi,ymax_hi,color='red')
        ax.vlines(LV_unrew,ymin_lo,ymax_lo,color='red')
#         ax.vlines(MV_rew,ymin_hi,ymax_hi,color='orange')
#         ax.vlines(MV_unrew,ymin_lo,ymax_lo,color='orange')
        ax.vlines(HV_rew,ymin_hi,ymax_hi,color='blue')
        ax.vlines(HV_unrew,ymin_lo,ymax_lo,color='blue')
        
        tick_marks = [np.mean([ymin_lo,ymax_lo]),0,0.25,0.5,0.75,1,np.mean([ymin_hi,ymax_hi])]
        tick_labels = ['Unrewarded',0,0.25,0.5,0.75,1,'Rewarded']
        ax.set_yticks(tick_marks,tick_labels)
        
        ax.set_ylabel('Choice Probability')
        ax.set_xlabel('Trials')
        
        # ax.set_xlim([0,100])

        ax.legend()
        ax.set_title(self.session)
        fig.tight_layout()
        
        return    
 

        
def merge_firingrate_behavior_dfs(session:str, firing_rate_df:pd.DataFrame, behavior_df:pd.DataFrame):
   
    ## Save Out Processed Data
    df = pd.concat([behavior_df,firing_rate_df],axis='columns') #merge behavior_df and firing_rate_df
    processed_data_dict = {'Name':session, 'df':df}
    
    return processed_data_dict  
        
        
def merge_sessions_processed_data_df(): #or this should probly be a dict of dfs huh since each will have diff num of trials
    return        #uses sessions to find dfs 
        
        
        
class NeuronTypeRegressions:
    
    def __init__(self,processed_data_dict):
        
        self.name = processed_data_dict['Name']
        self.df = processed_data_dict['df']
        
        self.alpha_threshold = 0.05

        self.neuron_type_df = None
        self.neuron_type_regression()
        
        self.dict_out = {'Name':self.name, 'df':self.neuron_type_df}
        
        return
    
        
    def neuron_type_regression(self):
        
        print('Performing neuron type regression..')

        # Make array of behavioral data
        # regressor_labels = ['Choice1','Side','Q1','Q2','Time']
        regressor_labels = [col for col in self.df.columns if not col.startswith(b"Unit")]
        regressor_matrix = self.df[regressor_labels].to_numpy()
        
        encodings = np.full(len(self.df['Unit_labels'][0]),None)
        units = self.df['Unit_labels'][0]
        
        
        
        
        # Regress FR of each unit against behavior
        for i,unit in enumerate(units): #loop thru all units of session
        
            FR = self.df[unit].to_numpy() #vector of FRs for each trial

            model = sm.OLS(FR, sm.add_constant(regressor_matrix),hasconst=True)
            res = model.fit()
            
#             print(FR)
#             print(sm.add_constant(regressor_matrix))
            
            # print(res.f_pvalue)
            if res.f_pvalue < self.alpha_threshold: #if regression is statistically significant
                signif_regressors_bools = res.pvalues<self.alpha_threshold #get which regressors were statistically signficant
                signif_regressors = [reg for indx,reg in enumerate(regressor_labels) if signif_regressors_bools[indx]] #get names of those regressors
                encodings[i] = signif_regressors 
                
#                 signif_regressors_posneg = res.params[signif_regressors_bools] #get the sign of the coefficient of the significant regressors
#                 encoding_posneg[chann][unit] = signif_regressors_posneg
#                 #within nested dict, unit number is the key and the significant regressors are the values
            
            else: #if regression isn't statistically significant, there's no encoding
                encodings[i] = 'Non-encoding'
                
                
            print(f'{unit} done. Neuron type: {encodings[i]}. ({i+1}/{len(self.df["Unit_labels"][0])})')
            
        print('Neuron type regressions done!')    
        
        self.neuron_type_df = pd.DataFrame.from_dict({'Unit':units, 'Neuron Type':encodings})
        # print(self.neuron_type_df)
        

        return
        
   
            
  
        
class LearningRateRegressions:
    
    def __init__(self,session):
        
        
        ## Get Spike Info (Firing Rates)
        Spikes = ProcessSpikes(session)
        self.spike_df = Spikes.dict_out['df']
        self.name = Spikes.dict_out['Name']
        
        ## Learning Rates for all Units
        self.learning_rates_df = None
        self.learning_rates = np.linspace(start=0.01,stop=0.99,num=50)
        self.Behavior = ProcessBehavior(session) #instantiate behavior
        self.get_learning_rates()
        
        ## Save Out
        self.dict_out = {'Name':self.name, 'df':self.learning_rates_df}
        
        return



    def get_learning_rates(self):
        
        print('Performing learning rate regressions..')
        
        df_rows_learning_rates = []
        
        # Get FR for each neuron
        units = self.spike_df['Unit_labels'][0]
        for i,unit in enumerate(units):
            
            FR = self.spike_df[unit].to_numpy() #vector of FRs for each trial
            
            self.regression_weights_df = self.learning_rate_regression(FR)
            
            regressor_names = self.regression_weights_df.columns
            
            # Find learning rate which maximizes each regression weight
            row_dict_learning_rates = {'Unit':unit}
            for j in range(len(regressor_names)):
                
                # idx of max regression weight for current regressor
                idx_max_regressor = self.regression_weights_df[regressor_names[j]].idxmax()
                
                # Find and store learning rate which corresponds to max regression weight
                row_dict_learning_rates['Learning rate for ' + regressor_names[j]] = self.learning_rates[idx_max_regressor]
                
            df_rows_learning_rates.append(row_dict_learning_rates)
            
            print(f'{unit} learning rate found!')
            
        self.learning_rates_df = pd.DataFrame(df_rows_learning_rates)
        print(self.learning_rates_df)
        
        return
    
                
                
    def learning_rate_regression(self,FR):
        
        # Get behavior regressor for each learning rate
        df_rows_regression_weights = []
        for learning_rate in self.learning_rates:
            
            self.Behavior.get_behavior(Q_learning=True, learning_rate=learning_rate)
            behavior_df = self.Behavior.behavior_df
            
            # Make array of behavioral data
            # regressor_labels = behavior_df.columns
            regressor_labels = ['Choice1','Side','Q1','Q2','Time']
            regressor_matrix = sm.add_constant(behavior_df[regressor_labels],has_constant='raise') #.to_numpy() 
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
        
        assert neuron_type_dict['Name'] == learning_rates_dict['Name'], 'Sessions do not match!'
        self.name = neuron_type_dict['Name']
        self.neuron_type_df = neuron_type_dict['df']
        self.learning_rates_df = learning_rates_dict['df']
        self.unit_list = self.neuron_type_df['Unit'].to_list()
        
        self.total_counts = None
        self.count_encodings()
        
        self.piecharts()
        
        
        
    def get_unit_data(self,df,unit,heading):
        desired_row = df[df['Unit']==unit]
        desired_value = desired_row[heading].values[0]

        return desired_value
        
    
    def count_encodings(self):
        
        self.total_counts = {'num_units':0, 'num_value':0,
                  'num_value1':0,
                  'num_value2':0,
                  'num_value1&2':0,
                  'num_choice':0,
                  'num_side':0,
                  'num_time':0}
        
        for unit in self.unit_list:
            unit_encoding = self.get_unit_data(self.neuron_type_df,unit,'Neuron Type')
            unit_encoding = ','.join(unit_encoding) #join function is to turn the list of strs into a single str to make it easier to search for keywords
        
            self.total_counts = self.count(unit_encoding,self.total_counts)
        
        return
    
    
    def count(self,unit_encoding,total_counts):
    
        total_counts['num_units'] +=1
        
        if 'Q' in unit_encoding:
            total_counts['num_value'] +=1
        
        if 'Q1' in unit_encoding:
            if 'Q2' in unit_encoding:
                total_counts['num_value1&2'] +=1
            else:
                total_counts['num_value1'] +=1
        elif 'Q2' in unit_encoding:
            total_counts['num_value2'] +=1
            
        elif 'Choice' in unit_encoding:
            total_counts['num_choice'] +=1
        elif 'Side' in unit_encoding:
            total_counts['num_side'] +=1
        elif 'Time' in unit_encoding:
            total_counts['num_time'] +=1
        
        return total_counts
    
    
    def piecharts(self):
        
        fig,ax = plt.subplots()

        num_units = self.total_counts['num_units']
        
        if num_units>0:
            
            slices=[self.total_counts['num_value1&2'],self.total_counts['num_value1'],
                    self.total_counts['num_value2'],self.total_counts['num_choice'],
                    self.total_counts['num_side'],self.total_counts['num_time']]
            explode=[0.2,0.2,0.2,0,0,0] #make the value slices stick out slightly
            labels=['Q1&Q2','Q1','Q2','Choice','Action','Time']
            colors=['#42f57e','#42f557','#42f5b0','red','gold','gray']
            
            ax.pie(slices,explode,labels,colors)
            
        ax.set_title(f'Num Neurons: {num_units}, Num Value: {self.total_counts["num_value"]}')
        fig.suptitle(self.name)
        fig.tight_layout()
        
        return
    
        
    def learning_rate_violins(self,volatile_learning_rates,stable_learning_rates):
        
        fig,ax=plt.subplots()
        ax.violinplot([volatile_learning_rates,stable_learning_rates])
        
        
        
        return
        
        
        
    def summary_behavior(self):
        return
        
        
    def session_behavior(session):
        return


#### Run

# session = 'braz_test'
session = 'braz20250225_04_te1880'
# session = 'braz20250326_04_te1923'

d=LearningRateRegressions(session)
xxx

a=ProcessSpikes(session).dict_out
xxx
b=Regressions(a.dict_out)
c=Plotting(b.dict_out,None,None)

# class SanityChecks:
#     def __init__(self):
#         sdkfj
        