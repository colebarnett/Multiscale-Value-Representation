# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:52:04 2025

@author: coleb
"""

#### Define global things

import os
import sys
import copy
import tables

import numpy as np
import pandas as pd
import statsmodels.api as sm

import DecisionMakingBehavior_Whitehall as BehaviorAnalysis






# Constants




# Paths
PROJECT_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis"
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')

sys.path.insert(1,BMI_FOLDER) #add bmi_python folder to package search path
sys.path.insert(2, NS_FOLDER) #add neuroshare python folder to package search path

from nsfile import NSFile





# Channel keys
# something here about which chs are which area
# just do ^^ with ^^ good_chans file?



session = 'braz_test'






class Processing:
	
	def __init__(self, session: str):
		
		"""
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
		
		self.session = session
		self.file_prefix = os.path.join(PROJECT_FOLDER, self.session)
		
# 		# [Initiate different data files]
# 		self.ns2file = None
# 		self.hdffile = None
# 		self.matfile = None
# 		self.pklfile = None
# 		self.ns5file = None
# 		self.nevfile = None
		
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
		
		## Get Times Align
		self.times_align = None
		self.get_times_align()
		
		## Get Spike Counts
		self.firing_rate_df = None
		self.t_before = 0.2 #how far to look before time_align point [s]
		self.t_after = 0.0 #how far to look after time_align point [s]
		self.get_firing_rate() 
		
		## Save Out Processed Data
		print(self.firing_rate_df)
		print(self.behavior_df)
		self.df = pd.concat([self.behavior_df,self.firing_rate_df],axis='columns') #merge behavior_df and firing_rate_df
		
		

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


		
		
		
		
	def get_behavior(self):
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
		values, win_sz = behavior.CalcValue_2Targs(choices,rewards,win_sz=10,smooth=True)
		
		# calculate RPE: rpe(t) = r(t) - Q(t). Note: Q just means value. Doesn't have to use q-learning specifically.
		rpe=[]
		for trial,choice in enumerate(choices):
			if choice==1: #lv choice
				rpe.append(rewards[trial] - values[0][trial]) #lv value
			elif choice==2: #hv choice
				rpe.append(rewards[trial] - values[1][trial]) #hv value
		rpe = np.transpose(np.array(rpe))
		

				
		# get which side (left or right) each choice was
		choice_side = behavior.GetTargetSideSelection()
		choice_side = ((choice_side-0.5)*2).astype(int) #make to be -1 or 1
		
# 		# get which center holds were stimulated
# 		stim_holds= (choices==1) * (instructed_or_freechoice==1) #get forced LV holds
# 		stim_holds[:100] = False #only want blB and blAp holds
# 		#print(stim_holds)
# 		for i,el in enumerate(stim_holds):
# 			if el:
# 				stim_holds[i]=1
# 			else:
# 				stim_holds[i]=0


		assert len(choices) == len(choice_side) == len(time) == len(values[0]) == len(values[1]) == len(rpe)
		
		self.behavior_df = pd.DataFrame.from_dict(
			{'Choice1':L_choices, 'Choice2':H_choices, 'Side':choice_side, 'Q1':values[0], 'Q2':values[1],
			'Reward':rewards, 'RPE':rpe, 'Time':time}) #, 'Stim':stim_holds})
		
		self.num_trials = len(choices)
		
		print('Behavior loaded!')

		
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
		
# 		fs_hdf = 60 #hdf fs is always 60
		
		# load behavior data
		cb = BehaviorAnalysis.ChoiceBehavior_Whitehall([self.hdf_file]) #method needs hdf filenames in a list
		
		# Find times of successful trials
# 		ind_hold_center = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
# 		ind_mvmt_period = cb.ind_check_reward_states - 3 #times corresponding to target prompt
# 		ind_reward_period = cb.ind_check_reward_states #times corresponding to reward period onset
		ind_target_hold = cb.ind_check_reward_states - 2 # times corresponding to target hold onset
		
		# align spike tdt times with hold center hdf indices using syncHDF files
		target_hold_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind_target_hold,cb.state_time,self.syncHDF_file)

		# Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
		assert len(target_hold_TDT_ind) == len(ind_target_hold), f'Repeat hold times! Session: {self.session}'
		assert len(target_hold_TDT_ind) == self.num_trials

		self.times_align = target_hold_TDT_ind
		
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

		assert self.has_nev, FileNotFoundError(f'.nev file not found! Session: {self.session}')

		## Get Spike Times
		print('Loading spike times..')
		self.nevfile = NSFile(self.file_prefix + '.nev')
		spike_entities = [e for e in self.nevfile.get_entities() if e.entity_type==3]
		headers = np.array([s.get_extended_headers() for s in spike_entities]) #get info for each ch
		unit_idxs = np.nonzero([h[b'NEUEVWAV'].number_sorted_units for h in headers])[0] #get ch idxs where there is a sorted unit
		unit_idxs=unit_idxs[:3] #to make runtime shorter for testing
		self.num_units = len(unit_idxs)
		self.unit_labels = [h[b'NEUEVLBL'].label[:7] for h in headers[unit_idxs]] #get labels of all sorted units
		num_units_per_idx = [h[b'NEUEVWAV'].number_sorted_units for h in headers[unit_idxs]]


		spike_times = [] #each element is a list spike times for a sorted unit
		spike_waveforms = [] #each element is a list of waveforms for a sorted unit
		for i,unit_idx in enumerate(unit_idxs): #loop thru sorted unit
			unit = spike_entities[unit_idx]
			spike_times.append([]) #initiate list of spike times for this unit
			
			for spike_idx in range(unit.item_count):
				spike_times[i].append(unit.get_segment_data(spike_idx)[0])
# 				spike_waveforms.append(unit.get_segment_data(spike_idx)[1])

			print(f'{self.unit_labels[i]}: {unit.item_count} spikes. ({i+1}/{len(unit_idxs)})')
		print('All spike times loaded!')
		
		## Get Spike Counts
		print('Counting spikes..')
		firing_rates = np.zeros((self.num_trials,self.num_units))
		for trial in range(self.num_trials):
			
			win_begin = self.times_align[trial] - self.t_before
			win_end = self.times_align[trial] + self.t_after

			for i in range(self.num_units):
				
				unit_spikes = np.array(spike_times[i])
				num_spikes = sum( (unit_spikes>win_begin) & (unit_spikes<win_end) )
				
				firing_rates[trial,i] = num_spikes / (self.t_before + self.t_after)
				
		print('Done counting spikes!')

		self.firing_rate_df = pd.DataFrame(firing_rates,columns=self.unit_labels)
		self.firing_rate_df['Trial'] = np.arange(self.num_trials)+1
		self.firing_rate_df['Unit_labels'] = [self.unit_labels for i in range(self.num_trials)] 
				
		return 
			
		
		
		
		
		
def merge_sessions_processed_data_df(): #or this should probly be a dict of dfs huh since each will have diff num of trials
	return		#uses sessions to find dfs 
		
		
		
class Regressions:
	
	def __init__(self,processed_data_df):
		
		self.df = processed_data_df
		
		self.alpha_threshold = 0.05

		self.neuron_type_regression()
		
		return


	def _split_rpe(rpe):
		#split rpe into positive and negative rpe signals
		pos_rpe = copy.deepcopy(rpe) #use deep copy so I can change one without changing the other
		neg_rpe = copy.deepcopy(rpe)
		pos_rpe[pos_rpe<0] = 0 #for positive rpe, set all negative rpes to zero
		neg_rpe[neg_rpe>0] = 0 #for negative rpe, set all positive rpes to zero		
		
		return pos_rpe, neg_rpe
	
		
	def neuron_type_regression(self):
		
		print('Performing neuron type regression..')

		# Make array of behavioral data
		regressor_labels = ['Choice1','Side','Q1','Q2','Time']
		regressor_matrix = self.df[regressor_labels].to_numpy()
		
		encodings = np.full(len(self.df['Unit_labels'][0]),None)
		
		
		# Regress FR of each unit against behavior
		for i,unit in enumerate(self.df['Unit_labels'][0]): #loop thru all units of session
		
			FR = self.df[unit].to_numpy() #vector of FRs for each trial

			model = sm.OLS(FR, sm.add_constant(regressor_matrix),hasconst=True)
			res = model.fit()
			
			if res.f_pvalue < self.alpha_threshold: #if regression is statistically significant
				signif_regressors_bools = res.pvalues<self.alpha_threshold #get which regressors were statistically signficant
				signif_regressors = [reg for indx,reg in enumerate(regressor_labels) if signif_regressors_bools[indx]] #get names of those regressors
				encodings[i] = signif_regressors 
				
# 				signif_regressors_posneg = res.params[signif_regressors_bools] #get the sign of the coefficient of the significant regressors
# 				encoding_posneg[chann][unit] = signif_regressors_posneg
# 				#within nested dict, unit number is the key and the significant regressors are the values
			
			else: #if regression isn't statistically significant, there's no encoding
				encodings[i] = 'Non-encoding'
				
				
			print(f'{unit} done. Neuron type: {encodings[i]}. ({i+1}/{len(self.df["Unit_labels"][0])})')
		print('Neuron type regressions done!')		
		
		return
		
		
		
		
		
	def learning_rate_regression():
		return
		
		
		
		def single_learning_rate_regression(learning_rate):
			return
			
			
			
			
		def get_best_learning_rate(regression_weights):
			return
			
			
	
	
	
	
	
	
class Plotting:
	
	
	def __init__(self,neuron_type_df,neuron_learning_rate_df,processed_data_df):
		small_fontsize = 8
		med_fontsize = 12
		large_fontsize = 18
		
		
		
	def piecharts():
		return
		
	def learning_rate_hist():
		return
		
		
		
	def summary_behavior():
		return
		
		
	def session_behavior(session):
		return




# class SanityChecks:
# 	def __init__(self):
# 		sdkfj
		