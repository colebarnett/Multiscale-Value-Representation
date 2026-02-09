# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:07:55 2023

@author: coleb
"""
import tables
import os
# import sys
# import time
# import warnings
import matplotlib.pyplot as plt
import numpy as np
# import tdt
# import pandas as pd
# from matplotlib.patches import Ellipse
import scipy as sp
# spect = sp.signal.spectrogram
# zscr = sp.stats.zscore
# from scipy.interpolate import make_interp_spline as spline
# from scipy import signal
# from scipy.ndimage import filters
from scipy import optimize as op
# import statsmodels.api as sm
# from logLikelihoodRLPerformance import logLikelihoodRLPerformance, RLPerformance
# from statsmodels.multivariate.manova import MANOVA


def trial_sliding_avg(trial_array, num_trials_slide):
    '''
    This method performs a simple sliding average of trial_array using a window
    of length num_trials_slide.
    '''
    num_trials = len(trial_array)
    slide_avg = np.zeros(num_trials)

    for i in range(num_trials):
        if i < num_trials_slide:
            slide_avg[i] = np.sum(trial_array[:i+1])/float(i+1)
        else:
            slide_avg[i] = np.sum(trial_array[i-num_trials_slide+1:i+1])/float(num_trials_slide)

    return slide_avg
    
    

def get_HDFstate_TDT_LFPsamples(ind_state,state_time,syncHDF_file):
    '''
    This method finds the TDT sample numbers that correspond to indicated task state using the syncHDF.mat file.

    Inputs:
        - ind_state: array with state numbers corresponding to which state we're interested in finding TDT sample numbers for, e.g. self.ind_hold_center_states
        - state_time: array of state times taken from corresponding hdf file
        - syncHDF_file: syncHDF.mat file path, e.g. '/home/srsummerson/storage/syncHDF/Mario20161104_b1_syncHDF.mat'
    Output:
        - lfp_state_row_ind: array of ripple sample numbers that correspond the the task state events in ind_state array
    '''
    # Load syncing data
    hdf_times = dict()
    sp.io.loadmat(syncHDF_file, hdf_times)
    #print(syncHDF_file)
    hdf_rows = np.ravel(hdf_times['row_number'])
    hdf_rows = [val for val in hdf_rows]
    #print(hdf_times['tdt_dio_samplerate'])
    dio_tdt_sample = np.ravel(hdf_times['ripple_samplenumber'])
    dio_freq = np.ravel(hdf_times['ripple_dio_samplerate'])

    lfp_dio_sample_num = dio_tdt_sample  # assumes DIOx and LFPx are saved using the same sampling rate

    state_row_ind = state_time[ind_state]        # gives the hdf row number sampled at 60 Hz
    lfp_state_row_ind = np.zeros(state_row_ind.size)

    for i in range(len(state_row_ind)):
        hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind[i])) #find index of row we are on
        if np.abs(hdf_rows[hdf_index] - state_row_ind[i])==0: #if time of state matches hdf row timestamp
            lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
        elif hdf_rows[hdf_index] > state_row_ind[i]: #if times don't match up, do linear interp with the prev timestamp
            hdf_row_diff = hdf_rows[hdf_index] - hdf_rows[hdf_index -1]  # distance of the interval of the two closest hdf_row_numbers
            m = (lfp_dio_sample_num[hdf_index]-lfp_dio_sample_num[hdf_index - 1])/hdf_row_diff
            b = lfp_dio_sample_num[hdf_index-1] - m*hdf_rows[hdf_index-1]
            lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
        elif (hdf_rows[hdf_index] < state_row_ind[i])&(hdf_index + 1 < len(hdf_rows)): #if times don't match up, do linear interp with the next timestamp
            hdf_row_diff = hdf_rows[hdf_index + 1] - hdf_rows[hdf_index]
            if (hdf_row_diff > 0):
                m = (lfp_dio_sample_num[hdf_index + 1] - lfp_dio_sample_num[hdf_index])/hdf_row_diff
                b = lfp_dio_sample_num[hdf_index] - m*hdf_rows[hdf_index]
                lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
            else:
                lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
        else:
            lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]

    return lfp_state_row_ind, dio_freq
    
    
class ChoiceBehavior_Whitehall():
    '''
    Class for behavior taken from Whitehall dynamic vs stable blocks task, where 
    probabilities for each target change throughout the course of the task. 
    Can pass in a list of hdf files when initially instantiated in the case that behavioral data
    is split across multiple hdf files. In this case, the files should be listed in the order in which they were saved.
    '''

    def __init__(self, hdf_files): #, num_trials_A, num_trials_B):
        for i, hdf_file in enumerate(hdf_files): 
            self.filename =  hdf_file
            table = tables.open_file(self.filename)
            if i == 0:
                self.state = table.root.task_msgs[:]['msg']
                self.state_time = table.root.task_msgs[:]['time']
#                 self.trial_type = table.root.task[:]['target_index']
                self.targetL = table.root.task[:]['targetL']
                self.targetH = table.root.task[:]['targetH']
#                 self.hdfs = [table]
                
                #get info on block type if available
                try:
                    self.is_stable_block = table.root.task[:]['is_stable_block']
                    self.is_volatile_block = table.root.task[:]['is_volatile_block']
                except:
                    self.is_stable_block = None
                    self.is_volatile_block = None

            else:
                self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
                self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])
                # self.trial_type = np.append(self.trial_type, table.root.task[:]['target_index'])
                self.targetL = np.vstack([self.targetL, table.root.task[:]['targetL']])
                self.targetH = np.vstack([self.targetH, table.root.task[:]['targetH']])
#                 self.hdfs.append(table)
                
        self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
        self.ind_center_states = np.ravel(np.nonzero(self.state == b'center'))   # total number of totals (includes repeats if trial was incomplete)
        self.ind_hold_center_states = np.ravel(np.nonzero(self.state == b'hold_center'))
        self.ind_hold_center_stimulate_states = np.ravel(np.nonzero(self.state == b'hold_center_and_stimulate'))
        self.ind_target_states = np.ravel(np.nonzero(self.state == b'target'))
        self.ind_hold_targetL_states = np.ravel(np.nonzero(self.state == b'hold_targetL'))
        self.ind_hold_targetH_states = np.ravel(np.nonzero(self.state == b'hold_targetH'))
        self.ind_check_reward_states = np.ravel(np.nonzero(self.state == b'check_reward'))
        
        self.fs_hdf = 60 #Hz
        
        self.num_trials = self.ind_center_states.size
        self.num_successful_trials = self.ind_check_reward_states.size
#         self.num_trials_A = num_trials_A
#         self.num_trials_B = num_trials_B
        #self.table=table


    def GetChoicesAndRewards(self):
        '''
        This method extracts which target was chosen for each trial, whether or not a reward was given, and if
        the trial was instructed or free-choice.

        Output:
        - chosen_target: array of length N, where N is the number of trials (instructed + freechoice), which contains 
                        contains values indicating a low-value target choice was made (=1) or if a high-value target
                        choice was made (=2)
        - rewards: array of length N, which contains 0s or 1s to indicate whether a reward was received at the end
                            of the ith trial. 
        - instructed_or_freechoice: array of length N, which indicates whether a trial was instructed (=1) or free-choice (=2)

        '''

        ind_holds = self.ind_check_reward_states - 2
        ind_rewards = self.ind_check_reward_states + 1
        rewards = np.array([float(st==b'reward') for st in self.state[ind_rewards]])
#         instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
        chosen_target = np.array([(int(self.state[ind]==b'hold_targetH') + 1) for ind in ind_holds])

        
        return chosen_target, rewards #, instructed_or_freechoice


#     def ChoicesAfterStimulation(self):
#         '''
#         Method to extract information about the stimulation trials and the trial immediately following
#         stimulation. 

#         Output: 
#         - choice: length n array with values indicating which target was selecting in the trial following
#             stimulation (= 1 if LV, = 2 if HV)
#         - stim_reward: length n array with Boolean values indicating whether reward was given (True) during
#             the stimulation trial or not (False)
#         - target_reward: length n array with Boolean values indicating whether reward was given (True) in
#             the trial following the stimulation trial or not (False)
#         - stim side: length n array with values indicating what side the MV target was on during the 
#             stimulation trial (= 1 for left, -1 for right)
#         - stim_trial_ind: length n array containing trial numbers during which stimulation was performed. 
#             Since stimulation is during Block A' only, the minimum value should be num_trials_A + num_trials_B
#             and the maximum value should be the total number of trials.
#         '''

#         # Get target selection information
#         chosen_target, rewards, instructed_or_freechoice, _ = self.GetChoicesAndRewards()
#         ind_targets = self.ind_check_reward_states - 3
#         targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
#         targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
#         instructed_trials_inds = np.ravel(np.nonzero(2 - instructed_or_freechoice))
#         stim_trial_inds = instructed_trials_inds[np.ravel(np.nonzero(np.greater(instructed_trials_inds, self.num_trials_A + self.num_trials_B)))]
#         stim_trial_inds = np.array([ind for ind in stim_trial_inds if ((ind+1) not in stim_trial_inds)&(ind < (self.num_successful_trials-1))])
#         fc_trial_inds = stim_trial_inds + 1

#         choice = chosen_target[fc_trial_inds]        # what target was selected in trial following stim trial
#         stim_reward = rewards[stim_trial_inds]        # was the stim trial rewarded
#         target_reward = rewards[fc_trial_inds]        # was the target selected in trial after stimulation rewarded
#         stim_side = targetL_side[stim_trial_inds]         # side the MV target was on during stimulation
#         choice_side = np.array([(targetH_side[ind]*(choice[i]==2) + targetL_side[ind]*(choice[i]==1)) for i,ind in enumerate(fc_trial_inds)])        # what side the selected target was on following the stim trial
#         
#         return choice, choice_side, stim_reward, target_reward, stim_side, stim_trial_inds

    def GetTargetSideSelection(self):
        '''
        Method to determine which side (left or right) the chosen target was on 
        for all successful reach trials.

        Output:
            - choice_side: 1 x num_trials vector
            Vector containing the side (left or right) chosen for each 
            successful reach trial. 0 or 1.

        '''
        choices, _  = self.GetChoicesAndRewards()

        # Get target side information
        ind_targets = self.ind_check_reward_states - 3
        targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
        targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
        
        # Get the side information for the chosen target
        choice_side = np.array([(targetH_side[i]*(choice==2) + targetL_side[i]*(choice==1)) for i,choice in enumerate(choices)])

        return choice_side
    
    
    
    def CalcValue_2Targs(self,choices,rewards,win_sz,smooth):
        '''
        Calculates values of the high value and low value target for each trial throughout a session by using a sliding window.
        This defines value as a number of rewards dispensed in past n choices of a particular target divided by past n choices to that target.
        Essentially empirical probability of reward as calculated by a sliding window of size n.
        
        Parameters
        ----------
        choices : 1 x num_trials vector
            Vector containing the NHP's choice for each trial which had a successful reach. 
            1 = low value target, 2 = high value target.
        rewards : 1 x num_trials vector
            Vector containing if a reward was dispensed for each trial which had a successful reach. 
            0 = no reward, 1 = reward dispensed.
        win_sz : int
            Defines window size for sliding average
        smooth : bool
            Whether to perform a sliding average over the final result
            
        Returns
        -------
        values : num_targets x num_trials
            Matrix containing the value of each target during each trial which had a successful reach.
            Row 0  = low value target
            Row 1 = high value target
        win_sz : int
            Defines window size for sliding average.
        '''
    
        val_lv = np.full([len(choices)],0.5) #0.5 is default value
        val_hv = np.full([len(choices)],0.5)
        
        # Calculate empirical probability of reward using sliding window
        for i in range(len(choices)):
            
            if i !=0 : #skip first trial
                
                #LV
                counter=0 #counts number of trials to target
                j=0 #variable to increment to go backwards in time
                lv_rewards=[] #list to accumulate rewards of lv choice trials
                
                while counter < win_sz: #go backwards in time until you get win_sz number of trials
                    
                    if i-j == 0: #if we are at beginning, then exit while loop
                        break
                    j+=1 
                    
                    if choices[i-j] == 1: #lv choices
                        lv_rewards.append(rewards[i-j]) #append rewards of lv choices
                        counter+=1 #update counter of how many trials we have
                        
                # calculate value
                if counter>0:
                    val_lv[i] = np.sum(lv_rewards) / counter #sum rewards and divide by number of trials
                    
                
                #HV
                counter=0 #counts number of trials to target
                j=0 #variable to increment to go backwards in time
                hv_rewards=[] #list to accumulate rewards of hv choice trials
                
                while counter < win_sz: #go backwards in time until you get win_sz number of trials
                    
                    if i-j == 0: #if we are at beginning, then exit while loop
                        break
                    j+=1 
                    
                    if choices[i-j] == 2: #hv choices
                        hv_rewards.append(rewards[i-j]) #append rewards of hv choices
                        counter+=1 #update counter of how many trials we have
                        
                # calculate value
                if counter>0:
                    val_hv[i] = np.sum(hv_rewards) / counter #sum rewards and divide by number of trials
            
#         print(lv_rewards)

        if smooth:
            #val_lv = trial_sliding_avg(val_lv,num_trials_slide=5)
            #val_hv = trial_sliding_avg(val_hv,num_trials_slide=5)
            val_hv = sp.ndimage.gaussian_filter(val_hv,sigma=1)
            val_lv = sp.ndimage.gaussian_filter(val_lv,sigma=1)

        # Combine high and low values into a single matrix
        values = [np.transpose(val_lv),np.transpose(val_hv)]
        return values, win_sz

    
    def GetRxnTime_ToggleSwitch(self):
        '''
        Plots a histogram of rxn times for successful trials.
        
        Gets monkey reaction time based on the time between the hold_center prompt
        and the target hold. This assumes instantaneous movement from center target
        to peripheral target which is pretty valid if a toggle switch is being used.
        '''
        target_prompt_inds = self.ind_check_reward_states - 3
        target_hold_inds = self.ind_check_reward_states - 1
        target_prompt_times = self.state_time[target_prompt_inds]
        target_hold_times = self.state_time[target_hold_inds]
        
        rxn_times = (target_hold_times - target_prompt_times) / self.fs_hdf
        
        fig,ax=plt.subplots()
        ax.hist(rxn_times,bins=np.linspace(0,5,50))
        ax.vlines(np.median(rxn_times),ax.get_ylim()[0],ax.get_ylim()[1],'k',linestyle='--',label=f'median = {np.median(rxn_times)}')
        ax.set_ylabel('Number of trials')
        ax.set_xlabel('Reaction Time (sec)')
        ax.legend()
        fig.suptitle(os.path.basename(self.filename))
        fig.tight_layout()
        
        



#     def GetFreeChoicesRewardsAndValues(choices,rewards,values,instructed_or_freechoice):
#         '''
#         Takes the choices, rewards, and values and filters them by instructed_or_freechoice
#         so that only the free choices remain.
#     
#         Parameters
#         ----------
#         choices : 1 x num_trials vector
#             Vector containing the NHP's choice for each trial which had a successful reach. 
#             0 = low value target, 2 = high value target.
#         rewards : 1 x num_trials vector
#             Vector containing if a reward was dispensed for each trial which had a successful reach. 
#             0 = no reward, 1 = reward dispensed.
#         instructed_or_freechoice: array of length num_trials
#             Indicates whether a trial was instructed (=1) or free-choice (=2)
#         values : num_targets x num_trials
#             Matrix containing the value of each target during each trial which had a successful reach.
#             Row 0  = low value target
#             Row 1 = high value target
#     
#         Returns
#         -------
#         free_choices : 1 x num_free_choices vector
#             Vector containing the NHP's choice for free choice trials which had a successful reach. 
#             1 = low value target, 2 = high value target.
#         free_choice_rewards : 1 x num_free_choices vector
#             Vector containing if a reward was dispensed for free choice trials which had a successful reach.
#         free_choice_values: num_targets x num_free_choices
#             Matrix containing the value of each target during free choice trials which had a successful reach.
#             Row 0  = low value target
#             Row 1 = high value target
#     
#         '''
#         
#         #Filter original vectors by instructed_or_freechoice to get only free choice trials
#         free_choices = choices[instructed_or_freechoice==2]
#         free_choice_rewards = rewards[instructed_or_freechoice==2]
#         free_choice_low_values = values[0][instructed_or_freechoice==2]
#         free_choice_high_values = values[1][instructed_or_freechoice==2]
#         free_choice_values = [free_choice_low_values,free_choice_high_values]
#         
#         return free_choices, free_choice_rewards, free_choice_values
    
    
    
#     def CalcMovtTime(self):
#         
#         #combine hold_targetL and hold_targetH
#         ind_hold_target_states = np.sort(np.concatenate((self.ind_hold_targetH_states,self.ind_hold_targetL_states)))
#                 
#         mts = []
#         #we only want the successful reaches (where the hold_center precedes a check_reward) 
#         for ind in ind_hold_target_states:
#             if self.state[ind+2] == b'check_reward': #if the reach was successful
#                 mt = (self.state_time[ind] - self.state_time[ind-1]) / self.fs_hdf
#                  #print(mt)
#                 mts.append(mt)
#             
#         return np.array(mts)

def CalcValue_2Targs_QLearning(chosen_target,rewards,alpha):
    # NOTE: We are only solving for beta here since alpha (learning rate) is specified.
    # The functions used have been changed accordingly.
    
    Q_initial = 0.5*np.ones(2)
    nll = lambda *args: -LL_Qlearning(*args)
    instructed_or_freechoice = np.full_like(chosen_target,2) #all trials are freechoice in Whitehall task
    result = op.minimize(nll, x0=1, args=(alpha, Q_initial, chosen_target, rewards, instructed_or_freechoice), bounds=[(0,None)])
    beta_ml = result["x"]
    
    if beta_ml > 100:
        print('Warning! Large beta value yielded.')
        print(f'alpha: {alpha}, beta: {beta_ml}')
    
    # RL model fit for Q values
    Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy,log_likelihood = Calculate_Qlearning(beta_ml, alpha, Q_initial, chosen_target, rewards, instructed_or_freechoice)

    return (Q_low, Q_high)


def Calculate_Qlearning(beta, alpha, Q_initial, chosen_target, rewards, instructed_or_freechoice):


    # Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
    Q_low = np.zeros(len(chosen_target))
    Q_high = np.zeros(len(chosen_target))

    # Set values for first trial (indexed as trial 0)
    Q_low[0] = Q_initial[0]
    Q_high[0] = Q_initial[1]

    # Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
    prob_choice_low = np.zeros(len(chosen_target))
    prob_choice_high = np.zeros(len(chosen_target))

    # Set values for first trial (indexed as trial 0)
    prob_choice_low[0] = 0.5
    prob_choice_high[0] = 0.5

    log_prob_total = 0.
    accuracy = np.array([])

    for i in range(len(chosen_target)-1):
        # Update Q values with temporal difference error
        delta_low = float(rewards[i]) - Q_low[i]
        delta_high = float(rewards[i]) - Q_high[i]
        Q_low[i+1] = Q_low[i] + alpha*delta_low*float(chosen_target[i]==1)
        Q_high[i+1] = Q_high[i] + alpha*delta_high*float(chosen_target[i]==2)

        # Update probabilities with new Q-values
        if instructed_or_freechoice[i+1] == 2:
            # Q_opt = Q_high[i+1]
            # Q_nonopt = Q_low[i+1]

            prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_low[i+1])))
            prob_choice_high[i+1] = 1. - prob_choice_low[i+1]

            prob_choice_opt = prob_choice_high[i+1]
            prob_choice_nonopt = prob_choice_low[i+1]

            #     # The choice on trial i+1 as either optimal (choice = 1.5) or nonoptimal (choice = 1)
            # choice = 0.5*chosen_target[i+1]+1
                # Does the predicted choice for trial i+1 match the actual choice
            accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))

            log_prob_total += np.log(prob_choice_nonopt*(chosen_target[i+1]==1) + prob_choice_opt*(chosen_target[i+1]==2))

        else:
            prob_choice_low[i+1] = prob_choice_low[i]
            prob_choice_high[i+1] = prob_choice_high[i]

    return Q_low,  Q_high, prob_choice_low, prob_choice_high,accuracy, log_prob_total

def LL_Qlearning(beta, alpha, Q_initial, chosen_target, rewards, instructed_or_freechoice):  
    Q_low, Q_high, prob_choice_low, prob_choice_high,accuracy, log_prob_total = Calculate_Qlearning(beta, alpha, Q_initial, chosen_target, rewards, instructed_or_freechoice)
    
    return log_prob_total