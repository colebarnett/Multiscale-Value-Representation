# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:30:10 2026

@author: coleb
"""
#### Libraries, Paths, and Imports
import os
import sys
import gif
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
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
# from matplotlib.colors import BoundaryNorm
# from numpy.polynomial.polynomial import polyfit

import Whitehall_utils as utils
from file_info import get_block_info, get_area_info
import DecisionMakingBehavior_Whitehall as BehaviorAnalysis
from Whitehall_Processing import ProcessSpikes,ProcessBehavior,load_data
from Whitehall_Globals import (SUBSESSIONS,SESSIONS,
PROJ_FOLDER,BMI_FOLDER,DATA_FOLDER,NS_FOLDER,NSX_FOLDER,
REGRESSOR_SUPERSET_DICT,ALL_REGRESSORS,FREQ_BANDS,AREAS,REW_PROBS,
VALUE_MODEL,ALIGNMENT,HOLD_DURATION,ENCODING_WINDOW,
T_BEFORE,T_AFTER,T_RES,T_VECTOR,MIN_FR_THRESH,MIN_NUM_TRIALS,
LEARNING_RATES,ALPHA_THRESHOLD,
COLORS,LINESTYLES,MARKERSTYLES,HATCHSTYLES)


#%% Load Data
os.chdir(PROJ_FOLDER)
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


#%% Main
        

    
    


class Behavior_Analysis:
    def __init__(self):
        return
    
    
    @staticmethod
    def choice_behav():
        df_list = []
        for s, session in enumerate(SESSIONS):
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s, session)

            #for plotting single sess results
            # Behav.plot_values(save_flag=False)
            # Behav.plot_value_dist(save_flag=False) #good for exemplar plots

            df_list.append(Behav.behavior_df)

        print(f'len df_list: {len(df_list)}')
        Behavior_Analysis.plot_choice_behav(df_list)

        return
    
    @staticmethod
    def plot_choice_behav(behavior_df_list):
        ##get fraction of choices towards a target during a given subsession and rew prob
        combined_df = utils.merge_sessions_df(behavior_df_list)

        def get_choices_rewprob_volatility(df, stable_or_volatile, rew_prob1):
            
            trials = utils.get_trials_rewprob_volatility(df, stable_or_volatile, rew_prob1)
            choice1 = df['Choice1'].iloc[trials]
            
            trials = utils.get_trials_rewprob_volatility(df, stable_or_volatile,1 - rew_prob1)  # do 1 minus rew prob since this is for targ 2
            choice2 = df['Choice2'].iloc[trials]
            
            return np.mean(pd.concat([choice1,choice2]))

        

        ## Variance Bar plots
        # vars_df = {'RewProb': [], 'StableOrVolatile': [], 'Variance': []}
        ylims = 0, 1
        colors = ['darkgray', 'dimgray']
        
        def plot_choice_bar(rewprobs,rewprob_labels):
            fig, ax = plt.subplots(layout='constrained')
            handles = []
            for i, stable_or_volatile in enumerate(SUBSESSIONS[1:]):
                for ii, rew_prob in enumerate(rewprobs):
                    choices = []
                    for behavior_df in behavior_df_list:  # loop thru all sessions individually
                        pct = get_choices_rewprob_volatility(behavior_df, stable_or_volatile, rew_prob)
                        choices.append(pct)

                    avg, sem = utils.get_avg_sem(choices, axis=0)
                    ax.bar(i + (ii * 2.5), avg, label=stable_or_volatile, color=colors[i])
                    ax.errorbar(i + (ii * 2.5), avg, sem, ecolor='k', color='k')
                patch = mpatches.Patch(color=colors[i], label=stable_or_volatile)
                handles.append(patch)

            ax.legend(handles=handles)
            ax.set_ylabel('Choice Fraction')
            ax.set_xticks([0.5, 3.0], [f'Rew Prob: {rewprobs[0]}', f'Rew Prob: {rewprobs[1]}'])
            ax.set_title(f'Target Reward Probabilities: {rewprob_labels}\nn={len(choices)}')
            ax.set_ylim(ylims)
            return

        # rewprob = 0/1 plot
        plot_choice_bar([0.,1.],'0/1') # rewprob = 0/1 plot
        plot_choice_bar([0.2,0.8],'0.2/0.8') # rewprob = 0.2/0.8 plot
        plot_choice_bar([0.4,0.6],'0.4/0.6') # rewprob = 0.4/0.6 plot
        
        
        return





    @staticmethod
    def value_dists(chosen_value_only=False):
        df_list = []
        for s, session in enumerate(SESSIONS):
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s, session)

            #for plotting single sess results
            Behavior_Analysis.plot_value_dist([Behav.behavior_df],chosen_value_only,Behav.session) #good for exemplar plots

            df_list.append(Behav.behavior_df)

        print(f'len df_list: {len(df_list)}')
        Behavior_Analysis.plot_value_dist(df_list,chosen_value_only,'All sess') #hists
        Behavior_Analysis.plot_value_dist_var(df_list,chosen_value_only) #bars

        return
    
    @staticmethod
    def _get_values_rewprob_volatility(df, stable_or_volatile, rew_prob1, chosen_value_only):
        if 'Q1' in df.columns:
            trials = utils.get_trials_rewprob_volatility(df, stable_or_volatile, rew_prob1)
            Q1 = df['Q1'].iloc[trials]
        else:
            Q1 = np.full(len(df), np.nan)

        if 'Q2' in df.columns:
            trials = utils.get_trials_rewprob_volatility(df, stable_or_volatile,
                                                         1 - rew_prob1)  # do 1 minus rew prob1 since this is for targ 2
            Q2 = df['Q2'].iloc[trials]
        else:
            Q2 = np.full(len(df), np.nan)

        if 'Qchosen' in df.columns:
            trials = utils.get_trials_rewprob_volatility_Qchosen(df, stable_or_volatile,rew_prob1)
            Qch = df['Qchosen'].iloc[trials]
        else:
            Qch = np.full(len(df), np.nan)
        
        if chosen_value_only:
            return Qch
        else:
            return np.concatenate([Q1, Q2])
        
    @staticmethod    
    def plot_value_dist(behavior_df_list,chosen_value_only,title):

        combined_df = utils.merge_sessions_df(behavior_df_list)
        
        ## Histogram plots
        bins = np.linspace(0.0, 1.0, num=20)
        
        def plot_valuedist_hist(rewprobs,rewprob_labels,chosen_value_only):
            
            fig, ax = plt.subplots(layout='constrained')
            for i, stable_or_volatile in enumerate(SUBSESSIONS[1:]):
                for ii, rew_prob in enumerate(rewprobs):
                    values = Behavior_Analysis._get_values_rewprob_volatility(combined_df, stable_or_volatile, rew_prob, chosen_value_only)
                    ax.hist(values, density=True, bins=bins, histtype='step',
                            label=stable_or_volatile, linestyle=LINESTYLES[i], color=COLORS[ii])
            ax.legend()
            ax.set_ylabel('Density')
            ax.set_xlabel('Subjective Value')
            ax.set_title(f'Target Reward Probabilities: {rewprob_labels}')
            fig.suptitle(title)
            return
        
        plot_valuedist_hist([0.,1.],'0/1',chosen_value_only) # rewprob = 0/1 plot
        plot_valuedist_hist([0.2,0.8],'0.2/0.8',chosen_value_only) # rewprob = 0.2/0.8 plot
        plot_valuedist_hist([0.4,0.6],'0.4/0.6',chosen_value_only) # rewprob = 0.4/0.6 plot
        
        return
    
    @staticmethod    
    def plot_value_dist_var(behavior_df_list,chosen_value_only):

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
                        values = Behavior_Analysis._get_values_rewprob_volatility(behavior_df, stable_or_volatile, rew_prob, chosen_value_only)
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
    
    
    @staticmethod
    def plot_behavior_learningrates_dual():

        assert 'Dual' in VALUE_MODEL
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

        assert 'Dual' not in VALUE_MODEL
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


class BehaviorPlots:
    def __init__(self, session: str, subsession: str):
        
        self.session=session
        self.subsession=subsession
        session_idx = SESSIONS.index(session)
        self.behavior_df = BEHAV_LIST[session_idx].behavior_df
        
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
     
  
    
    
    def plot_values_vs_choices_vs_rewprob(self,num_targs=1,trials_to_plot=[0,60],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        
        fig,ax=plt.subplots(layout='constrained')
        
        #actual rew prob
        ax.plot(self.behavior_df['Rew_prob_1'].iloc[start:end],color='k',linestyle='--',label='Reward Probability 1')
        if num_targs==2:
            ax.plot(self.behavior_df['Rew_prob_2'].iloc[start:end],color='gray',linestyle='--',label='Reward Probability 2')
        
        #smoothed choice prob
        window_length=10
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)
        ax.plot(np.arange(start,end),P_LV[start:end],color='red',label='Choice 1')
        if num_targs==2:
            ax.plot(np.arange(start,end),P_HV[start:end],color='green',label='Choice 2')
            
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[start:end],color='orange',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[start:end],color='turquoise',label='Q2')
            
        #plot labels
        ax.set_title('Modeled Values vs Actual Choices vs Actual Rew Prob')
        ax.legend()
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsChoicesVsRewProb',PROJ_FOLDER)
        
        return fig
    
    
    def plot_choices_vs_rewprob(self,num_targs=1,trials_to_plot=[0,200],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        fig,ax=plt.subplots(layout='constrained')
        
        #actual rew prob
        ax.plot(self.behavior_df['Rew_prob_1'].iloc[start:end],color='k',linestyle='--',label='Reward Probability 1')
        if num_targs==2:
            ax.plot(self.behavior_df['Rew_prob_2'].iloc[start:end],color='gray',linestyle='--',label='Reward Probability 2')
        
        #smoothed choice prob
        window_length=10
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)
        ax.plot(np.arange(start,end),P_LV[start:end],color='red',label='Choice 1')
        if num_targs==2:
            ax.plot(np.arange(start,end),P_HV[start:end],color='green',label='Choice 2')
            
        #shade every other block
        edges = np.nonzero(np.diff(self.behavior_df['Rew_prob_1'].iloc[start:end]))[0]
        y=[-0.05,1.05]
        for edge_num in range(0,len(edges),2):
            ax.fill_betweenx(y,edges[edge_num],edges[edge_num+1],color='k',alpha=0.15)
            
        #plot labels
        ax.set_title('Actual Choices vs Actual Rew Prob')
        ax.legend()
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'ChoicesVsRewProb',PROJ_FOLDER)
        
        return fig
    
    def plot_choices(self,num_targs=1,trials_to_plot=[0,200],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        fig,ax=plt.subplots(layout='constrained')
        
        #smoothed choice prob
        window_length=10
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)
        ax.plot(np.arange(start,end),P_LV[start:end],color='red',label='Choice 1')
        if num_targs==2:
            ax.plot(np.arange(start,end),P_HV[start:end],color='green',label='Choice 2')
        
        #shade every other block
        edges = np.nonzero(np.diff(self.behavior_df['Rew_prob_1'].iloc[start:end]))[0]
        y=[-0.05,1.05]
        for edge_num in range(0,len(edges),2):
            ax.fill_betweenx(y,edges[edge_num],edges[edge_num+1],color='k',alpha=0.15)
            
        #plot labels
        ax.set_title('Actual Choices')
        ax.legend()
        ax.set_xlabel('Trials')
        ax.set_ylim(y)
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'Choices',PROJ_FOLDER)
        
        return fig    
    
    

    def plot_rewprob(self,num_targs=1,trials_to_plot=[0,200],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        fig,ax=plt.subplots(layout='constrained')
        
        #actual rew prob
        ax.plot(self.behavior_df['Rew_prob_1'].iloc[start:end],color='k',linestyle='--',label='Reward Probability 1')
        if num_targs==2:
            ax.plot(self.behavior_df['Rew_prob_2'].iloc[start:end],color='gray',linestyle='--',label='Reward Probability 2')

            
        #plot labels
        ax.set_title('Actual Choices vs Actual Rew Prob')
        ax.legend()
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'RewProb',PROJ_FOLDER)
        
        return fig

    
    def plot_values_vs_choices(self,num_targs=1,trials_to_plot=[0,200],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        fig,ax=plt.subplots(layout='constrained')
        
        #smoothed choice prob
        window_length=10
        P_LV = BehaviorAnalysis.trial_sliding_avg(self.behavior_df['Choice1'],window_length)
        P_HV = BehaviorAnalysis.trial_sliding_avg(1 - self.behavior_df['Choice1'],window_length)
        ax.plot(np.arange(start,end),P_LV[start:end],color='red',label='Choice 1')
        if num_targs==2:
            ax.plot(np.arange(start,end),P_HV[start:end],color='green',label='Choice 2')
            
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[start:end],color='orange',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[start:end],color='turquoise',label='Q2')
            
        #plot labels
        ax.set_title('Modeled Values vs Actual Choices')
        ax.legend()
        ax.set_xlabel('Trials')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsChoices',PROJ_FOLDER)
        
        return fig
    
    
    def plot_values_vs_rewards(self,num_targs=2,trials_to_plot=[0,200],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        fig,ax=plt.subplots(layout='constrained')
            
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[start:end],color='orange',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[start:end],color='turquoise',label='Q2')
            
        #Rewards
        for trial in np.arange(start,end):
            if self.behavior_df['Choice1'].iloc[trial] == 1:
                if self.behavior_df['Reward'].iloc[trial] == 1:
                    ax.plot(trial,self.behavior_df['Q1'].iloc[trial],color='orange',marker='^') #rewarded choice1s
                else:
                    ax.plot(trial,self.behavior_df['Q1'].iloc[trial],color='orange',marker='s') #unrewarded choice1s
            
            if num_targs==2:
                if self.behavior_df['Choice2'].iloc[trial] == 1:
                    if self.behavior_df['Reward'].iloc[trial] == 1:
                        ax.plot(trial,self.behavior_df['Q2'].iloc[trial],color='turquoise',marker='^') #rewarded choice2s
                    else:
                        ax.plot(trial,self.behavior_df['Q2'].iloc[trial],color='turquoise',marker='s') #unrewarded choice2s
        
            
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
        
        return fig,ax
    
    def plot_values(self,num_targs=2,trials_to_plot=[0,200],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        fig,ax=plt.subplots(layout='constrained')
        
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[start:end],color='red',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[start:end],color='green',label='Q2')
        
        #shade every other block
        edges = np.nonzero(np.diff(self.behavior_df['Rew_prob_1'].iloc[start:end]))[0]
        y=[-0.05,1.05]
        for edge_num in range(0,len(edges),2):
            ax.fill_betweenx(y,edges[edge_num],edges[edge_num+1],color='k',alpha=0.15)
        
        #plot labels
        ax.set_title('Modeled Values vs Actual Reward Probability')
        ax.legend()
        ax.set_xlabel('Trials')
        ax.set_ylabel('Value')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')   
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsRewProb',PROJ_FOLDER)
        
        return fig,ax

    
    def plot_values_vs_rewprob(self,num_targs=2,trials_to_plot=[0,200],save_flag=False):
        start=trials_to_plot[0]
        end=trials_to_plot[1]
        fig,ax=plt.subplots(layout='constrained')
        
        #Q-learning values
        ax.plot(self.behavior_df['Q1'].iloc[start:end],color='red',label='Q1')
        if num_targs==2:
            ax.plot(self.behavior_df['Q2'].iloc[start:end],color='green',label='Q2')
        
        #actual rew prob
        ax.plot(self.behavior_df['Rew_prob_1'].iloc[start:end],color='red',linestyle='--',label='Reward Probability 1')
        if num_targs==2:
            ax.plot(self.behavior_df['Rew_prob_2'].iloc[start:end],color='green',linestyle='--',label='Reward Probability 2')
        
        #plot labels
        ax.set_title('Modeled Values vs Actual Reward Probability')
        ax.legend()
        ax.set_xlabel('Trials')
        ax.set_ylabel('Value')
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')   
        
        if save_flag:
            utils.save_svg(fig,self.session,'ValueVsRewProb',PROJ_FOLDER)
        
        return fig,ax
    
    def plot_specific_value_vs_rewprob(self,value_type='Q1',num_trials_to_plot=200,save_flag=False):
        
        fig,ax=plt.subplots(layout='constrained')
        
        color = 'red'
        
        #Q-learning values
        ax.plot(self.behavior_df[value_type].iloc[:num_trials_to_plot],color=color,label=value_type)
        
        #actual rew prob (only for q1 and q2)
        if value_type == 'Q1':
            ax.plot(self.behavior_df['Rew_prob_1'].iloc[:num_trials_to_plot],color=color,linestyle='--',label='Reward Probability 1')
        elif value_type == 'Q2':
            ax.plot(self.behavior_df['Rew_prob_2'].iloc[:num_trials_to_plot],color=color,linestyle='--',label='Reward Probability 2')
        
        #plot labels
        ax.set_title(f'Modeled {value_type} Values')
        ax.legend()
        ax.set_xlabel('Trials')
        ax.set_ylabel('Value',color=color)
        ax.tick_params(axis='y', labelcolor=color)
        
        fig.suptitle(f'{self.session}\n{VALUE_MODEL}')   
        
        if save_flag:
            utils.save_svg(fig,self.session,'SpecificValue',PROJ_FOLDER)
        
        
        return fig,ax
    
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
    

    
    
    def get_behavior_var_corr(self,list_of_vars):
        
        num_vars = len(list_of_vars)
        ind_vars = np.zeros((num_vars,self.num_trials))
        corr_matrix = np.zeros((num_vars,num_vars))
        print('asdklfj')
        
        #gather behavioral vars from behavior df
        for i,var in enumerate(list_of_vars): 
            
            if ' x ' not in var: #for single (non-interaction) regressors
                ind_vars[i,:] = sp.stats.zscore(self.behavior_df[var].to_numpy())
            if ' x ' in var: #for interaction terms
                reg1,reg2 = var.split(sep=' x ')
                interaction = self.behavior_df[reg1] * self.behavior_df[reg2]
                ind_vars[i,:] = sp.stats.zscore(interaction.to_numpy())
      
        # calculate pairwise corr between all combos of vairables
        for i in range(num_vars):
            for j in range(num_vars):
                res = sp.stats.pearsonr(ind_vars[i,:],ind_vars[j,:])
                corr_matrix[i,j] = res.statistic
                
        corr_matrix = np.multiply(corr_matrix,corr_matrix) #make it r squared
        
        return corr_matrix
    
    @staticmethod
    def plot_var_corr(corr_matrix,list_of_vars,title):
        num_vars = len(list_of_vars)
        fig,ax=plt.subplots(layout='constrained')
        im = ax.imshow(corr_matrix,vmin=0.2,vmax=0.9,cmap='Reds')
        ax.set_title('Behavior Variable Correlation (R^2)')
        ax.set_aspect(0.9)#to make labels readable
        c=plt.colorbar(im)
        plt.xticks(range(num_vars),list_of_vars)
        ax.tick_params(axis='x', labelrotation=90)
        # for tick in ax.xaxis.get_major_ticks()[1::2]:
        #     tick.set_pad(15) #
        plt.yticks(range(num_vars),list_of_vars)
        fig.suptitle(title)
        
        return fig    
    
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

#### Behavior example plots
def plot_choice_vs_rewprob_vol_exemplar():
    
    
    session = "braz20260824_03_te2584"
    
    bp=BehaviorPlots(session,'all trials')
    # bp.plot_choices_vs_rewprob(num_targs=1,trials_to_plot=[0,300],save_flag=False)
    # bp.plot_choices_vs_rewprob(num_targs=1,trials_to_plot=[480,780],save_flag=False)
    # # bp.plot_choices_vs_rewprob(num_targs=1,trials_to_plot=[0,1080],save_flag=False)
    # # bp.plot_rewprob(num_targs=2,trials_to_plot=[0,1080],save_flag=False)
    # bp.plot_choices(num_targs=1,trials_to_plot=[0,1080],save_flag=False)
    
    bp.plot_values_vs_choices_vs_rewprob(num_targs=2,trials_to_plot=[35,75],save_flag=False)
    # bp.plot_values_vs_rewards(num_targs=1,trials_to_plot=[0,100],save_flag=False)
    # bp.plot_values(num_targs=1,trials_to_plot=[0,1080],save_flag=False)

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
            # Behav.plot_values_vs_choices_vs_rewprob(num_targs=1,num_trials_to_plot=60,save_flag=False)
            # Behav.plot_values_vs_rewards(num_targs=1,num_trials_to_plot=60,save_flag=False)
            # Behav.plot_values_vs_choices_vs_rewprob(num_targs=2,num_trials_to_plot=60,save_flag=False)
            # Behav.plot_values_vs_rewards(num_targs=2,num_trials_to_plot=60,save_flag=False)
            Behav.plot_choices_vs_rewprob(num_targs=1,num_trials_to_plot=300,save_flag=False)
            
    return 