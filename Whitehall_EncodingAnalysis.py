# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:36:20 2026

@author: coleb
"""
#### Libraries, Paths, and Imports
import os
# import sys
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
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
# from matplotlib.colors import BoundaryNorm
# from numpy.polynomial.polynomial import polyfit

import Whitehall_utils as utils
from file_info import get_block_info, get_area_info
import DecisionMakingBehavior_Whitehall as BehaviorAnalysis
from Whitehall_Processing import (ProcessSpikes,ProcessBehavior,get_Spikes_Behav_LFP,
SPIKES_LIST,BEHAV_LIST,LFP_LIST) #data is loaded here
from Whitehall_Globals import (SUBSESSIONS,SESSIONS,
PROJ_FOLDER,BMI_FOLDER,DATA_FOLDER,NS_FOLDER,NSX_FOLDER,
REGRESSOR_SUPERSET_DICT,ALL_REGRESSORS,FREQ_BANDS,AREAS,REW_PROBS,
VALUE_MODEL,ALIGNMENT,HOLD_DURATION,ENCODING_WINDOW,
T_BEFORE,T_AFTER,T_BOXCAR_SIZE,T_BOXCAR_STEP,T_VECTOR,N_TIMEBINS,MIN_FR_THRESH,MIN_NUM_TRIALS,
LEARNING_RATES,ALPHA_THRESHOLD,
COLORS,LINESTYLES,MARKERSTYLES,HATCHSTYLES)

os.chdir(PROJ_FOLDER)

#%% Main


class Regressor_Analysis:
    
    def __init__(self):
        
        # set of all regressors, no interactions
        # self.all_regs = [
        #               'Q1',
        #               'Q2',
        #               'Qleft',
        #               'Qright',
        #               'Qhigh',
        #               'Qlow',
        #               'Qchosen',
        #               'Qdiff_',
        #               'absQdiff',
        #               'ChoiceTarg',
        #               'ChoiceExploit',
        #               'ChoiceSide',
        #               'Time',
        #               # 'Time x Time',
        #               # 'Log_Time',
        #               'State',
        #               'RPE',
        #               'absRPE',
        #               'Prev_Reward',
        #               'Prev_Rew ⊥ Qch',
        #               'Qch ⊥ Prev_Rew',
        #               ]
        
        
        # set of all regressors EXCEPT TIME, no interactions
        # self.all_regs = [
        #               'Q1',
        #               'Q2',
        #               'Qleft',
        #               'Qright',
        #               'Qhigh',
        #               'Qlow',
        #               'Qchosen',
        #               'Qdiff_',
        #               'absQdiff',
        #               'ChoiceTarg',
        #               'ChoiceExploit',
        #               'ChoiceSide',
        #               # 'Time',
        #               # 'Log_Time',
        #               # 'State',
        #               'RPE',
        #               ]
    
        
        # value regressors with state (stable or vol) interactions
        # self.all_regs = [
        #               'Q1 x State',
        #               'Q2 x State',
        #               'Qleft x State',
        #               'Qright x State',
        #               'Qhigh x State',
        #               'Qlow x State',
        #               'Qchosen x State',
        #               'Qdiff_ x State',
        #               'absQdiff x State',
        #               'RPE x State',
        #               ]
        
        # value regressors with time interactions
        # self.all_regs = [
        #               'Q1 x Time',
        #               'Q2 x Time',
        #               'Qleft x Time',
        #               'Qright x Time',
        #               'Qhigh x Time',
        #               'Qlow x Time',
        #               'Qchosen x Time',
        #               'Qdiff_ x Time',
        #               'absQdiff x Time',
        #               'RPE x Time',
        #               ]
        
        # value regressors with ChoiceSide interactions
        # self.all_regs = [
        #               'Q1 x ChoiceSide',
        #               'Q2 x ChoiceSide',
        #               'Qleft x ChoiceSide',
        #               'Qright x ChoiceSide',
        #               'Qhigh x ChoiceSide',
        #               'Qlow x ChoiceSide',
        #               'Qchosen x ChoiceSide',
        #               'Qdiff_ x ChoiceSide',
        #               'absQdiff x ChoiceSide',
        #               'RPE x ChoiceSide',
        #               ]
    
    
        # value regressors with exploitation/exploration interactions
        # self.all_regs = [
        #               'Q1 x ChoiceExploit',
        #               'Q2 x ChoiceExploit',
        #               'Qleft x ChoiceExploit',
        #               'Qright x ChoiceExploit',
        #               'Qhigh x ChoiceExploit',
        #               'Qlow x ChoiceExploit',
        #               'Qchosen x ChoiceExploit',
        #               'Qdiff_ x ChoiceExploit',
        #               'absQdiff x ChoiceExploit',
        #               'RPE x ChoiceExploit',
        #               ]
    
        # chosen value regressors with all interactions
        # self.all_regs = [
        #               'Qchosen',
        #               'Qchosen x ChoiceTarg',
        #               'Qchosen x ChoiceExploit',
        #               'Qchosen x Time',
        #               'Qchosen x State',
        #               'Qchosen x ChoiceSide',
        #               ]

        # chosen value regressors with interactions and multiple terms
        # self.all_regs = [
        #               'Qchosen',
        #               'Prev_Reward',
        #               # ['Qchosen','State'],
        #               # 'Qchosen x State',
        #               'State',
        #               # 'Qchosen x Time',
        #               # ['Qchosen','Time'],
        #               'Time',
        #               ['Qchosen','Time','State'],
        #               ['Prev_Reward','Time','State'],
        #               # ['Qchosen','Prev_Reward','Time','State'],
        #               ['Qch ⊥ Prev_Rew','Time','State'],
        #               ['Qch ⊥ Prev_Rew','Prev_Reward','Time','State'],
        #               ]
        
        
        
        # chosen value regressors with interactions and multiple terms
        self.all_regs = [
                      'Qchosen',
                      'Prev_Reward',
                      'State',
                      'Time',
                      ['Time','State'],
                      ['Qchosen','Time','State'],
                      ['Prev_Reward','Time','State'],
                      ['Prev_Rew ⊥ Qch','Time','State'],
                      ['Qch ⊥ Prev_Rew','Time','State'],
                      ['Qch ⊥ Prev_Rew','Prev_Reward','Time','State'],
                      ]
        
    
        # residualized prev reward regressor with all interactions
        # self.all_regs = [
        #               'Prev_Rew ⊥ Qch',
        #               'Prev_Rew ⊥ Qch x ChoiceTarg',
        #               'Prev_Rew ⊥ Qch x ChoiceExploit',
        #               'Prev_Rew ⊥ Qch x Time',
        #               'Prev_Rew ⊥ Qch x State',
        #               'Prev_Rew ⊥ Qch x ChoiceSide',
        #               ]
        
        # prev reward regressor with all interactions
        # self.all_regs = [
        #               'Prev_Reward',
        #               'Prev_Reward x ChoiceTarg',
        #               'Prev_Reward x ChoiceExploit',
        #               'Prev_Reward x Time',
        #               'Prev_Reward x State',
        #               'Prev_Reward x ChoiceSide',
        #               ]
        
        # Q1 regressors with all interactions
        # self.all_regs = [
        #               'Q1',
        #               'Q1 x ChoiceTarg',
        #               'Q1 x ChoiceExploit',
        #               'Q1 x Time',
        #               'Q1 x State',
        #               'Q1 x ChoiceSide',
        #               ]
        
        # reward regressors with all interactions
        # self.all_regs = [
        #               'Q2',
        #               'Q2 x ChoiceTarg',
        #               'Q2 x ChoiceExploit',
        #               'Q2 x Time',
        #               'Q2 x State',
        #               'Q2 x ChoiceSide',
        #               ]
        
        return
    
    def get_reg_corr_matrix(self):
        
        reg_corr_matrices=[]
        for s,session in enumerate(SESSIONS):
            
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            reg_corr_matrices.append(Behav.get_behavior_var_corr(self.all_regs))
            
            
        # reg corr plot
        _=ProcessBehavior.plot_var_corr(np.mean(reg_corr_matrices,axis=0),self.all_regs,'Across All Sessions Average')
        
        
        return
        
    def calc_reg_metrics(self, metrics=['pval','rsqr','coef']):
    
        def get_interaction(compound_reg):
            reg1,reg2 = compound_reg.split(sep=' x ')
            interaction = behavior_df[reg1] * behavior_df[reg2]
            interaction.name = compound_reg
            return interaction
        
        # metrics = ['pval','rsqr','coef']
        # metrics = ['rsqr']
        # bounds = {'pval':[0.00,0.15], 'rsqr':[0.00,0.05], 'coef':[0.00,0.75]}
        # thresh = {'pval':0.05, 'rsqr':0.05, 'coef':0.5}
        colors = {AREAS[0]:'tab:green', AREAS[1]:'tab:blue'}
        
        PCA = PCA_Analysis()
        
        for PCA_or_singleunit in ['PCs','single units']:
            print('*'*20)
            print(PCA_or_singleunit)
            
            for area in AREAS:
                print(area)
                color=colors[area]
            
                metric_out_dict = dict()
    
                for metric in metrics:
                    print(metric)
                    start = time.time()
                    for reg in self.all_regs:
                        metric_out_dict[str(reg)] = []
                        
                        for s,session in enumerate(SESSIONS):
                            
                            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
                            behavior_df = Behav.behavior_df
                            
                            if PCA_or_singleunit == 'PCs':
                                try:
                                    area_fr_df = PCA.pcs_dfs_dict[session][area]
                                except:
                                    continue #no pca for this session/area
                            elif PCA_or_singleunit == 'single units':
                                area_fr_df = utils.area_parser(Spikes.fr_df,area)
                                if len(area_fr_df)<1:
                                    continue #no units for this session/area
                                
                            units = list(area_fr_df.columns)
    
                            
                            for j,unit in enumerate(units):
                                fr = area_fr_df[unit]       
                            
                                # simple regression
                                if ' x ' not in reg: #for single (non-interaction) regressors
                                    res = utils.linear_encoding_regression(fr,behavior_df[reg])
                                if ' x ' in reg: #for interaction terms
                                    interaction = get_interaction(reg)
                                    res = utils.linear_encoding_regression(fr,interaction)
                                
                                # #multiple term regression
                                # if isinstance(reg,list):
                                    
                                    
                                
                                if metric=='pval':
                                    metric_out_dict[str(reg)].append(res.pvalues[reg])
                                elif metric=='rsqr':
                                    metric_out_dict[str(reg)].append(res.rsquared)
                                elif metric=='adj rsqr':
                                    metric_out_dict[str(reg)].append(res.rsquared_adj)
                                elif metric=='coef':
                                    metric_out_dict[str(reg)].append(abs(res.params[reg]))
                                else:
                                    raise ValueError
                                    
                    
                        
                    num_units = len(metric_out_dict[str(reg)])
                    
                    # # histograms
                    # fig,axs=plt.subplots(3,3,layout='constrained')
                    # fig.suptitle(f'{metric}, {area} {PCA_or_singleunit} (n={num_units})')
                    # row_counter=0
                    # col_counter=0
                    # for reg in all_regs:
                    #     ax=axs[row_counter,col_counter]
                    #     ax.hist(metric_out_dict[reg],range=bounds[metric],color=color)
                    #     # ax.vlines(thresh[metric],0,np.max(np.histogram(metric_out_dict[reg],range=bounds[metric])[0]),color='k',linestyle='--')
                    #     ax.set_title(reg)
                    #     if row_counter==2:
                    #         ax.set_xlabel(metric)
                    #     else:
                    #         ax.set_xlabel('')
                    #     if col_counter==0:
                    #         ax.set_ylabel('Num')
                    #     else:
                    #         ax.set_ylabel('')
                            
                    #     row_counter+=1
                    #     if row_counter>2:
                    #         row_counter=0
                    #         col_counter+=1
                    #     if col_counter>2:
                    #         fig,axs=plt.subplots(3,3,layout='constrained')
                    #         fig.suptitle(f'{metric}, {area} {PCA_or_singleunit} (n={num_units})')
                    #         row_counter=0
                    #         col_counter=0
                            
                    
                    
                    #get avg and sem for bar plot
                    avgs,sems = [],[]
                    for reg in self.all_regs:
                        avg,sem = utils.get_avg_sem(metric_out_dict[str(reg)],axis=0)
                        avgs.append(avg)
                        sems.append(sem)
                     
                    # avg + sem bar plot
                    fig,ax=plt.subplots()
                    fig.suptitle(f'{metric}, {area} {PCA_or_singleunit} (n={num_units})')
                    ax.bar([str(reg) for reg in self.all_regs],avgs,yerr=sems,color=color)
                    # ax.hlines(thresh[metric],0,len(all_regs),color='k',linestyle='--')
                    ax.tick_params(axis='x', labelrotation=90)
                    ax.set_ylabel(metric)
                    
                      
                    # Get quartiles for each reg
                    quarts={}
                    for quart in [0.25,0.5,0.75]:
                        q_=[]
                        for reg in self.all_regs:
                            quartiles = np.percentile(metric_out_dict[str(reg)],quart)
                            q_.append(quartiles)
                        quarts[str(quart)] = q_
                        
                    # quartile bar plot
                    fig,ax=plt.subplots()
                    fig.suptitle(f'{metric} Quartiles, {area} {PCA_or_singleunit} (n={num_units})')
                    my_plot = ax.grouped_bar(quarts, tick_labels=self.all_regs, group_spacing=1)
                    # for container in my_plot.bar_containers:
                    #     ax.bar_label(container, padding=3)
                    # ax.grouped_bar([str(reg) for reg in self.all_regs],quarts,color=color)
                    # ax.hlines(thresh[metric],0,len(all_regs),color='k',linestyle='--')
                    ax.tick_params(axis='x', labelrotation=90)
                    ax.set_ylabel(metric)                   
                    
                    #best fit bar graph: num of units for which each reg has the highest r sqr
                    if 'rsqr' in metric:
                        df = pd.DataFrame(metric_out_dict) # rows = neurons, columns = regressors
                        # df=df.drop(columns=df.filter(like="Time").columns) #exclude nuisance regs
                        # df=df.drop(columns=df.filter(like="Stable").columns) #exclude nuisance regs
                        counts = df.idxmax(axis=1).value_counts(normalize=True).reindex(df.columns, fill_value=0) *100
                        fig,ax=plt.subplots()
                        fig.suptitle(f'Percentage of units for which regressor was the best fit\n{area} {PCA_or_singleunit} (n={num_units})')  
                        ax.bar(df.columns,counts,color=color)
                        ax.tick_params(axis='x', labelrotation=90)
                        ax.set_ylabel('% units')
                        
                    
                    #explained bar graph: num of units for which each reg had a signif slope
                    if metric=='pval':
                        df = pd.DataFrame(metric_out_dict) # rows = neurons, columns = regressors
                        # df.drop(columns='Time') #exclude nuisance reg
                        threshold = 0.05
                        signif = (df < threshold).mean()*100
                        fig,ax=plt.subplots(layout='constrained')
                        fig.suptitle(f'Percentage of units for which regressor was significant\n{area} {PCA_or_singleunit} (n={num_units})')
                        ax.bar(df.columns,signif,color=color)
                        ax.tick_params(axis='x', labelrotation=90)
                        ax.set_ylabel('% units')
                        
                        
                                   
                    print(f'Time taken: {time.time()-start:.2f} secs')
            

                
    
        
        return

class PCA_Analysis:
    
    def __init__(self,overwrite_flag=False):#,pct_var_explained):
        # assert pct_var_explained<=1., 'Express percent as a decimal! (e.g. 50%=0.5)'
        # self.pct_var_explained = pct_var_explained
        
        self.min_num_units = 6
        
        self.overwrite_flag=overwrite_flag
    
        self.n_dims_dict = self.get_n_dims_scree()
        
        self.pcs_dfs_dict = self.get_PCs(self.n_dims_dict) #dict of dfs organized by session and area
        
        self.get_PCs_encoding_type()
        
        # self.get_PC_learning_rate()
        
        return
    
    def get_n_dims_scree(self):
        
        n_dims_dict=dict()
        for s,session in enumerate(SESSIONS):
            n_dims_dict[session]=dict()
            
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            for area in AREAS:
                               
                file_name = f'n_pca_dims_{area}'
                
                #load
                if (not self.overwrite_flag) and utils.does_pkl_exist(file_name,session,DATA_FOLDER):
                   n_dims_dict[session][area] = utils.load_pkl(file_name,session,DATA_FOLDER) #int
               
                #or compute and save
                else:
                    area_fr_df = utils.area_parser(Spikes.fr_df,area)
                    n_units = len(area_fr_df.keys())
                    if n_units < self.min_num_units:
                        continue #skip if not enough units for this session and area
                        
                    # _ = self.get_PCs(area_fr_df,session)
                    
                    pca = sklearn.decomposition.PCA()
                    pca = pca.fit(area_fr_df.values) #n_obs x n_components = n_trials x n_units
                
                
                    ev = pca.explained_variance_ # n_components x 1
                    nv = pca.noise_variance_ #float
                    
                    # Scree plot
                    fig,ax=plt.subplots(layout='constrained')
                    ax.plot(np.arange(len(ev))+1,ev*100,'o-',label='Components')
                    ax.hlines(nv,1,len(ev),color='k',linestyle='--',label='Noise Variance')
                    ax.set_xlabel('Principal Component #')
                    ax.set_ylabel('Variance Explained')
                    ax.set_title('Scree Plot')
                    fig.suptitle(f'{session}\n{area}, {n_units} units')
                    
                    plt.show()
                    
                    n_dims = int(input("How many components to keep? "))
                    n_dims_dict[session][area] = n_dims
                    
                    utils.save_pkl(n_dims,file_name,session,DATA_FOLDER)
                    
                    plt.close()
            print(f'{s+1}/{len(SESSIONS)} sessions done.')
                    
                
        return n_dims_dict
    
    def get_PCs(self,n_dims_dict):
        
        pcs_dfs_dict=dict()
        for s,session in enumerate(SESSIONS):
            pcs_dfs_dict[session]=dict()
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            for area in AREAS:
                
                area_fr_df = utils.area_parser(Spikes.fr_df,area)
                n_units = len(area_fr_df.keys())
                if n_units < self.min_num_units:
                    continue #skip if no units for this session and area
       
        
                # Fit PCA model on all trials. 
                # This defines the population manifold for the particular area (over the entire session)
                pca = sklearn.decomposition.PCA()
                pcs_arr = pca.fit_transform(area_fr_df.values) #n_obs x n_components = n_trials x n_units
                num_pcs_to_keep = n_dims_dict[session][area]
                comp_labels = [f'PC#{i+1} {session}' for i in range(num_pcs_to_keep)]
                pcs_df = pd.DataFrame(pcs_arr[:,:num_pcs_to_keep],columns=comp_labels)
                
                pcs_dfs_dict[session][area] = pcs_df
        
        print('PCA done for all sessions!')
        
        return pcs_dfs_dict
    
    @staticmethod
    def get_n_dims_pctvar_hist():
        pct_vars = [0.5,0.6,0.7,0.8,0.9] 
        min_num_units = 3
       
        for pct_var_explained in pct_vars:
            n_dims_list = []
            for s,session in enumerate(SESSIONS):
                
                Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
                
                for area in AREAS:
    
                    area_fr_df = utils.area_parser(Spikes.fr_df,area)
                    n_units = len(area_fr_df.keys())
                    if n_units < min_num_units:
                        continue #skip if not enough units for this session and area
                        
                    # _ = self.get_PCs(area_fr_df,session)
                    
                    pca = sklearn.decomposition.PCA()
                    pca = pca.fit(area_fr_df.values) #n_obs x n_components = n_trials x n_units
                    
                    assert np.all( pca.explained_variance_ratio_ == np.sort(pca.explained_variance_ratio_,descending=True) ) #make sure sorted big to small
                    cumul_var_explained = np.cumsum(pca.explained_variance_ratio_)
                    num_pcs_to_keep = sum(cumul_var_explained < pct_var_explained) + 1 #get all PCs needed to explain specified pct_var_explained
                    n_dims_list.append(num_pcs_to_keep)
                    
                    
            fig,ax=plt.subplots(layout='constrained')
            ax.hist(n_dims_list)
            ax.vlines(np.median(n_dims_list),0,np.max(np.histogram(n_dims_list)[0]),'k',linestyle='--',label=f'median={np.median(n_dims_list)}')
            ax.legend()
            ax.set_ylabel('num populations')
            ax.set_xlabel('num dims')
            ax.set_title(f'Num Dims needed to explain {pct_var_explained*100}% of Variance')
        
                
        return 
    
    def get_PCs_encoding_type(self):
        
        for s,session in enumerate(SESSIONS):
                
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            for area in AREAS:
                
                try:
                    pc_df = self.pcs_dfs_dict[session][area]
                except KeyError: #skip if no neurons for area
                    continue
                
                if not utils.does_csv_exist(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER):
                    pcs_encodingtype_df,_ = NeuronType_Analysis.get_neuron_types(pc_df,Behav.behavior_df)
                    utils.save_csv(pcs_encodingtype_df,f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                    
        return
    
    def _make_3level_counter_dict(self,list1,list2,list3):
        # I tried a slicker way of doing this but it didn't work cuz they used the same pointers
        
        counter_dict = {}
        for item1 in list1:
            counter_dict[item1]={}
            for item2 in list2:
                counter_dict[item1][item2]={}
                for item3 in list3:
                    counter_dict[item1][item2][item3] = 0
                    
        return counter_dict 
    
    def _count_value_vs_other(self,df,encoding_str):
        if encoding_str == 'Other': #i.e. not value
           return sum(~df['Neuron Type'].str.contains('Q')) 
        else: #value or any specific regressor
            return sum(df['Neuron Type'].str.contains(encoding_str))
            
    
    def count_eachPC_encoding_type(self):
        
        regressors = ['Qchosen','Q1','Q2','Other']
        colors = ['aquamarine','mediumspringgreen','cyan','grey']       
        PC_list = ['PC#1','PC#2','PC#3']
        #initialize counters. 
        counters_dict = self._make_3level_counter_dict(AREAS,PC_list,regressors)
        
        for s,session in enumerate(SESSIONS):
            for area in AREAS:
                
                try:
                    pcs_encodingtype_df = utils.load_csv(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                except:
                    continue
               
                for PC in PC_list:
                    
                    pc_df = pcs_encodingtype_df[pcs_encodingtype_df['Unit'].str.contains(PC)]
                    if len(pc_df)<1:
                        continue
                   
                    for regressor in regressors:
                       
                        n_type = self._count_value_vs_other(pc_df,regressor)
                        counters_dict[area][PC][regressor] += n_type
        self.plot_encoding_type(counters_dict,colors,suptitle='Encoding Types by PC')           
        return counters_dict
    
    def count_allPCs_encoding_type(self):
        
        regressors = ['Qchosen','Q1','Q2','Other']
        colors = ['aquamarine','mediumspringgreen','cyan','grey']

        #initialize counters. 
        counters_dict = self._make_3level_counter_dict(AREAS,SUBSESSIONS,regressors)
        
        for s,session in enumerate(SESSIONS):
            for area in AREAS:
                
                try:
                    pcs_encodingtype_df = utils.load_csv(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                except:
                    continue
               
                for subsession in SUBSESSIONS:
                    
                    subsession_df = pcs_encodingtype_df[pcs_encodingtype_df['Unit'].str.contains(subsession)]
                   
                    for regressor in regressors:
                       
                        n_type = self._count_value_vs_other(subsession_df,regressor)
                        counters_dict[area][subsession][regressor] += n_type
        self.plot_encoding_type(counters_dict,colors,suptitle='All PCs pooled: Encoding Types')           
        return counters_dict
    
    def plot_encoding_type(self,counters_dict,colors,suptitle):
        
        for key1 in counters_dict.keys():
            
            fig,axs=plt.subplots(1,3,layout='constrained')
            
            for ss,key2 in enumerate(counters_dict[key1].keys()):
                types = list(counters_dict[key1][key2].keys())
                counts = list(counters_dict[key1][key2].values())
                
                if sum(counts)>0:
                    ax=axs[ss]
                    ax.pie(counts, labels=types,colors=colors, autopct='%1.0f%%')
                    ax.set_title(f'{key2}')
                
            fig.suptitle(suptitle + f'\n{key1}')
        return
                
    def get_PC_learning_rate(self):
        
            
        for s,session in enumerate(SESSIONS):
            
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            for area in AREAS:
                
                try:
                    pc_df = self.pcs_dfs_dict[session][area]
                except KeyError: #skip if no neurons for area
                    continue
                
                assert utils.does_csv_exist(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                pcs_encodingtype_df = utils.load_csv(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                
                if not utils.does_csv_exist(f'PCA_LearningRates_{area}',session,DATA_FOLDER):
                    learning_rate_df = LearningRate_Analysis.get_learning_rates(pc_df,Behav,pcs_encodingtype_df,do_even_num_trials=True)
                    utils.save_csv(learning_rate_df,f'PCA_LearningRates_{area}',session,DATA_FOLDER)
        
        return
    
    
    
    @staticmethod
    def plot_PCA_trajs(list_of_sessions,num_trials_to_plot):
        
        color = 'blue'
    
        PCA = PCA_Analysis(overwrite_flag = False)
        
        for s,session in enumerate(SESSIONS):        
            
            if session not in list_of_sessions:
                continue
            
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            for area in AREAS:
                try:
                    pcs_encodingtype_df = utils.load_csv(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                    pcs_df = PCA.pcs_dfs_dict[session][area]
                except:
                    print(f'Skipping {session}, {area}!!')
                    continue
                
                for pc in pcs_df.columns:
                    
                    encoding_type = str(utils.get_unit_data(pcs_encodingtype_df, f'{pc} - all trials', 'Neuron Type'))
                    if "Q" not in encoding_type: #only do pcs that encode value
                        continue
                    
                    label = f'{pc[:4]}, {encoding_type}' 
                    assert session in pc
                
                    fig,ax = Behav.plot_specific_value_vs_rewprob(encoding_type,num_trials_to_plot=num_trials_to_plot,save_flag=False)
                    ax2 = ax.twinx()
                    ax2.plot(pcs_df[pc].iloc[:num_trials_to_plot],color=color,label=label)
                    ax2.legend()
                    ax2.set_ylabel('Z-scored pop activity',color=color)
                    ax2.tick_params(axis='y', labelcolor=color)
            
        print('Plotting..')
        return
     
    def get_rsqrs(self):

        rsqrs_dict = {stable_or_volatile: [] for stable_or_volatile in SUBSESSIONS}
        for s,session in enumerate(SESSIONS):
            start=time.time()
            
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            
            for area in AREAS:
                
                try:
                    pc_df = self.pcs_dfs_dict[session][area]
                except KeyError: #skip if no neurons for area
                    continue
                
                
                assert utils.does_csv_exist(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                encoding_type_df = utils.load_csv(f'PCA_EncodingTypes_{area}',session,DATA_FOLDER)
                assert utils.does_csv_exist(f'PCA_LearningRates_{area}',session,DATA_FOLDER)
                learning_rate_df = utils.load_csv(f'PCA_LearningRates_{area}',session,DATA_FOLDER)
            
                for stable_or_volatile in SUBSESSIONS:
                
                    rsqrs_sess = utils.get_rsqrs_from_LRs(stable_or_volatile, pc_df, Behav, encoding_type_df, learning_rate_df, do_even_num_trials=True, REGRESSORS=REGRESSORS)
                    rsqrs_dict[stable_or_volatile] = rsqrs_dict[stable_or_volatile] + rsqrs_sess
                
            utils.get_time_left(start,s,len(SESSIONS)) 
        
        #get avgs and sem
        avgs,sems=[],[]
        for stable_or_volatile in SUBSESSIONS:
            avg,sem = utils.get_avg_sem(rsqrs_dict[stable_or_volatile],axis=0)
            avgs.append(avg)
            sems.append(sem)
            
        fig,ax=plt.subplots(layout='constrained')
        ax.bar(SUBSESSIONS,avgs,yerr=sems)
        ax.set_ylabel('R-Squared')
        ax.set_title('R-Squareds of neurons with fitted learning rates')
        
        return






def find_sig_bins(pvals, f_pvals, alpha=0.05, n_consec=3):
    """Significant if at least `n_consec` consecutive bins have both
    regressor p <= alpha and whole-regression F-test p <= alpha.

    pvals   : (n_timebins,) p-value of the regressor's beta in each bin
    f_pvals : (n_timebins,) F-test p-value of the full regression in each bin

    Returns (is_sig, sig_bins), where sig_bins are the indices of all bins
    that belong to a run of >= n_consec significant bins.
    """
    sig = ((pvals <= alpha) & (f_pvals <= alpha)).astype(int)
    run_starts = np.flatnonzero(np.convolve(sig, np.ones(n_consec, dtype=int), 'valid') == n_consec)
    sig_bins = np.unique(run_starts[:, None] + np.arange(n_consec))
    return sig_bins.size > 0, sig_bins


def plot_temporal_encoding(psth_dict, unit, behavior_df, regressor, bin_times, n_bins=5,
                           alpha=0.05, n_consec=3):
    """Rainbow plot for one unit.

    Top: trial-averaged PSTHs, split into bins by the value of `regressor`.
    Bottom: OLS beta for `regressor` at each time bin (all regressors fit jointly),
            red = p<=0.05, blue = p<0.1, grey = n.s.

    psth_dict   : dict, unit label -> (n_trials, n_timebins) array
    unit        : key of psth_dict to plot
    behavior_df : DataFrame, one row per trial, one column per regressor
    regressor   : column of behavior_df to plot
    bin_times   : (n_timebins,) time of each PSTH bin relative to the alignment event
    """
    psths = psth_dict[unit]
    assert psths.shape == (len(behavior_df), len(bin_times))
    title = unit

    # Regress each time bin on all regressors; keep the chosen regressor's beta and p-value
    X = sm.add_constant(behavior_df)
    fits = [sm.OLS(psths[:, k], X).fit() for k in range(psths.shape[1])]
    betas = np.array([f.params[regressor] for f in fits])
    pvals = np.array([f.pvalues[regressor] for f in fits])
    f_pvals = np.array([f.f_pvalue for f in fits])
    is_sig, sig_bins = find_sig_bins(pvals, f_pvals, alpha, n_consec)

    # Bin trials by regressor value (equal-width, non-overlapping bins)
    values = behavior_df[regressor].to_numpy()
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    bin_idx = np.digitize(values, edges[1:-1])  # 0 .. n_bins-1
    centers = (edges[:-1] + edges[1:]) / 2
    cmap = plt.cm.jet
    norm = Normalize(edges[0], edges[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[2, 1], layout="constrained")

    # Top: PSTH per value bin
    for j in range(n_bins):
        in_bin = bin_idx == j
        if not in_bin.any():
            continue
        avg, sem = utils.get_avg_sem(psths[in_bin], axis=0)
        color = cmap(norm(centers[j]))
        ax1.plot(bin_times, avg, color=color)
        ax1.fill_between(bin_times, avg - sem, avg + sem, color=color, alpha=0.1)
    ax1.set_ylabel('FR')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax1, location='top', label=regressor)

    # Bottom: beta over time, colored by significance
    colors = np.where(pvals <= alpha, 'red', np.where(pvals < 0.1, 'blue', 'grey'))
    ax2.scatter(bin_times, betas, c=colors)
    ax2.axhline(0, color='k')
    ax2.set_ylabel('Beta coeff')
    ax2.set_xlabel('Time (s)')

    fig.suptitle(title)
    plt.show()
    return fig, is_sig, sig_bins

def do_rainbow_plots():
    
    for s,session in enumerate(SESSIONS):
        
        Spikes,Behav,LFP = get_Spikes_Behav_LFP(s,session)
        # NEED TO DO FOR EACH NEURON TYPE??
        for unit,psth_arr in Spikes.psth_dict.items():
        
            _ = plot_temporal_encoding(psth_arr, unit, Behav.behavior_df[QCHOSEN_REGRESSORS], 'Qchosen')
        xx
        
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
    
    @staticmethod
    def get_neuron_types(fr_df,behavior_df):
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
                signif_regressors = utils.linear_encoding_regression_feature_selection_new(
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