# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:31:09 2026

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
T_BEFORE,T_AFTER,T_BOXCAR_SIZE,T_BOXCAR_STEP,T_VECTOR,N_TIMEBINS,MIN_FR_THRESH,MIN_NUM_TRIALS,
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

class Process_LearningRates:
    
    def __init__(self,do_even_num_trials=True):
        
        self.fig_folder = 'LearningRates'
        
        self.do_even_num_trials=do_even_num_trials
       
        self.lr_dfs = []
        self.nt_dfs = []
        for s,session in enumerate(SESSIONS):
            
            start = time.time()

            try:
                lr_grid_behav_dict = utils.load_pkl('LR_Grid_behavior_dfs_dict',session,DATA_FOLDER)
            except:
                print('Need to run get_learning_rate_grid_behavior first!')
                return
            
            # Get learning rate for each unit
            Spikes, Behav, _ = get_Spikes_Behav_LFP(s,session)
            learning_rate_df,learning_rate_df_grid = self.get_learning_rates(Spikes.fr_df,Behav,lr_grid_behav_dict,self.do_even_num_trials)
            utils.save_csv(learning_rate_df,'LearningRates',session,DATA_FOLDER)
            utils.save_csv(learning_rate_df_grid,'LearningRatesGridSearch',session,DATA_FOLDER)
            print('*'*50)
            print('*'*50)
            utils.get_time_left(start,s,len(SESSIONS))
            print('*'*50)
            print('*'*50)
                
            self.lr_dfs.append(learning_rate_df)
       
        self.allsess_lr_df = utils.merge_sessions_df(self.lr_dfs)
        utils.save_csv(self.allsess_lr_df,'LearningRates','',DATA_FOLDER)
        
        return 
    
    @staticmethod
    def get_learning_rate_grid_behavior():
        for s,session in enumerate(SESSIONS):
            
            start = time.time()
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            Behav.get_learning_rate_grid_behavior(LEARNING_RATES)
            utils.get_time_left(start,s,len(SESSIONS))
            
        return
    
    @staticmethod   
    def get_rsqrs(do_even_num_trials):

        rsqrs_dict = {stable_or_volatile: [] for stable_or_volatile in SUBSESSIONS}
        for s,session in enumerate(SESSIONS):
            
            start = time.time()
            
            print(session)
            
            
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            neuron_type_df = utils.load_csv('NeuronTypes',session,DATA_FOLDER)
            learning_rate_df = utils.load_csv('LearningRatesSingle',session,DATA_FOLDER)
            
            for stable_or_volatile in SUBSESSIONS:
            
                rsqrs_sess = utils.get_rsqrs_from_LRs(stable_or_volatile, Spikes.fr_df, Behav, neuron_type_df, learning_rate_df, do_even_num_trials, REGRESSORS)
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


    @staticmethod    
    def get_learning_rates(psth_dict,Behavior,lr_grid_behav_dict,do_even_num_trials):
    
        units = psth_dict.keys
        print(f'{len(units)} neurons')
        
        learning_rate_grid = lr_grid_behav_dict['learning rate grid']
        
        df_rows_learning_rates = [] #to save out info in a df
        for stable_or_volatile in SUBSESSIONS:        
            
            if do_even_num_trials:
                trials = utils.get_trials_even_num(Behavior.behavior_df,stable_or_volatile)
            else:
                trials = utils.get_trials(Behavior.behavior_df,stable_or_volatile)
        
            if not len(trials) > 1: #if no trials for current subsession
                print(f'No trials for {stable_or_volatile}. Skipping!')    
                continue
            else:
                print(stable_or_volatile)
        
            # Get FR for each neuron
            for i,unit in enumerate(units):
                
                start = time.time()
                
                psths = psth_dict[unit] #shape = n_trials x n_timebins
                assert psths.shape == (len(Behavior.behavior_df), N_TIMEBINS)
                
                psth_subsess = psths[trials,:] #psths for all subsession trials
                
                if 'Dual' in VALUE_MODEL: #dual LR
                
                    ## Do regression for all learning rate combos
                    ## Keep regressor set and learning rate(s) that fit best
                    best_rsqr = 0.
                    best_learning_rates = ['None','None']
                    neuron_type = 'None'
                
                    ##Grid search
                    for pos_alpha in learning_rate_grid:
                        for neg_alpha in learning_rate_grid:
                            
                            pos_alpha, neg_alpha = np.round([pos_alpha,neg_alpha],decimals=2) # to help with floating point errors
                            behavior_df = lr_grid_behav_dict['data'][(pos_alpha, neg_alpha)]
                            
                            # get set of regressors 
                            for value_type,regressors in REGRESSOR_SUPERSET_DICT.items():
                                
                                behav_subsess = behavior_df[regressors].iloc[trials]       
                                
                                # regress fr against the behavior calculated for this learning rate and set of regressors
                                # do this for each timebin
                                fits = [utils.linear_encoding_regression(psth_subsess[:,k],behav_subsess) for k in range(N_TIMEBINS)]
                                
                                fit_pvals = np.array([f.f_pvalue for f in fits])
                                reg_pval = np.array([f.pvalues[value_type] for f in fits])
                                betas = np.array([f.params[value_type] for f in fits])
                                rsqrs = np.array([f.rsquared for f in fits])
                                
                                # only consider if regression and value regressor are both significant
                                is_signif,signif_bin_idxs = utils.find_sig_bins(reg_pval, fit_pvals, alpha=ALPHAGOHERE, n_consec=NCONSEC)
                                if is_signif: FIND WAY TO DEAL WITH OR FIND BEST IDXS OR WHEN TO USE ONLY BEST IDXS
                                    
                                    # take lr and value type if it yields the best rsqr so far, otherwise continue on
                                    avg_rsqr = utils.mean_of_top_n_of_arr(rsqrs,NCONSEC)
                                    if avg_rsqr > best_rsqr:
                                        best_rsqr = avg_rsqr
                                        best_learning_rates = [pos_alpha,neg_alpha]
                                        neuron_type = value_type
                                        avg_beta = 
                                        
                                        
                    #save out info for unit
                    lr_row_dict_1 = {'Unit':f'{unit} - {stable_or_volatile}',
                                   'Neuron Type': neuron_type,
                                   'Pos Learning Rate': best_learning_rates[0],
                                   'Neg Learning Rate': best_learning_rates[1],
                                   'R-Squared': best_rsqr}
                    df_rows_learning_rates_grid.append(lr_row_dict_1)
                
                    ## Refine grid search using optimizer
                    if neuron_type != 'None':
                        nll = lambda *args: -1*utils.lr_reg(*args).llf
                        x0 = best_learning_rates  # [pos_alpha, neg_alpha]
                        result_A = sp.optimize.minimize(nll, x0, 
                        		   args=(Behavior,fr,trials,REGRESSOR_SUPERSET_DICT[neuron_type]),
                        		   bounds=[(0.01,1), (0.01,1)],
                        		   method='Nelder-Mead',
                                   tol=1e-3)
                        best_learning_rates = result_A["x"]
                    
                        #save out info for unit
                        lr_row_dict = {'Unit':f'{unit} - {stable_or_volatile}',
                                       'Neuron Type': neuron_type,
                                       'Pos Learning Rate': best_learning_rates[0],
                                       'Neg Learning Rate': best_learning_rates[1],
                                       'R-Squared': utils.lr_reg(best_learning_rates,Behavior,fr,trials,REGRESSOR_SUPERSET_DICT[neuron_type]).rsquared}
                        df_rows_learning_rates.append(lr_row_dict)
                        
                        
                if 'Dual' not in VALUE_MODEL:
                    
                    ## Do regression for all learning rate combos
                    ## Keep regressor set and learning rate(s) that fit best
                    best_rsqr = 0.
                    best_learning_rate = 'None'
                    neuron_type = 'None'
                    
                    ##Grid search
                    for alpha in learning_rate_grid:
                            
                        alpha = np.round(alpha,decimals=2) # to help with floating point errors
                        behavior_df = lr_grid_behav_dict['data'][(alpha)]
                        
                        # get set of regressors 
                        for value_type,regressors in REGRESSOR_SUPERSET_DICT.items():
                                    
                            # regress fr against the behavior calculated for this learning rate and set of regressors
                            res = utils.linear_encoding_regression(fr,behavior_df[regressors].iloc[trials])
                            
                            # only consider if regression and value regressor are both significant
                            if res.f_pvalue < ALPHA_THRESHOLD and res.pvalues[value_type] < ALPHA_THRESHOLD:
                                
                                # take lr and value type if it yields the best rsqr so far, otherwise continue on
                                if res.rsquared > best_rsqr:
                                    best_rsqr = res.rsquared
                                    best_learning_rate = alpha
                                    neuron_type = value_type
                                        
                                        
                    #save out info for unit
                    lr_row_dict_1 = {'Unit':f'{unit} - {stable_or_volatile}',
                                   'Neuron Type': neuron_type,
                                   'Learning Rate': best_learning_rate,
                                   'R-Squared': best_rsqr}
                    df_rows_learning_rates_grid.append(lr_row_dict_1)
                
                    ## Refine grid search using optimizer
                    if neuron_type != 'None':
                        nll = lambda *args: -1*utils.lr_reg(*args).llf
                        x0 = best_learning_rate  # [pos_alpha, neg_alpha]
                        result_A = sp.optimize.minimize(nll, x0, 
                                   args=(Behavior,fr,trials,REGRESSOR_SUPERSET_DICT[neuron_type]),
                                   bounds=[(0.01,1)],
                                   method='Nelder-Mead',
                                   tol=1e-3)
                        best_learning_rate = result_A["x"]
                    
                        #save out info for unit
                        lr_row_dict = {'Unit':f'{unit} - {stable_or_volatile}',
                                       'Neuron Type': neuron_type,
                                       'Learning Rate': best_learning_rate,
                                       'R-Squared': utils.lr_reg(best_learning_rate,Behavior,fr,trials,REGRESSOR_SUPERSET_DICT[neuron_type]).rsquared}
                        df_rows_learning_rates.append(lr_row_dict)
                    
                utils.get_time_left(start,i,len(units))
                                
        learning_rate_df = pd.DataFrame(df_rows_learning_rates) #rows: units x subsessions, cols: lrs for each reg
        learning_rate_df_grid = pd.DataFrame(df_rows_learning_rates_grid) #rows: units x subsessions, cols: lrs for each reg
        # utils.save_csv(learning_rate_df,'LR',"debug",DATA_FOLDER)

        return learning_rate_df, learning_rate_df_grid


class LearningRate_Analysis:
    
    def __init__(self,do_stable_first_only=False,do_volatile_first_only=False):
        
        self.fig_folder = 'LearningRates'
       
        self.lr_dfs = []
        for s,session in enumerate(SESSIONS):
            
            #to test stab->vol vs vol->stab
            is_stable_block, is_volatile_block = get_block_info(session)
            if do_volatile_first_only:
                if is_stable_block[0]==1 and is_volatile_block[0]==0:
                    continue #skip stab first
            if do_stable_first_only:
                if is_stable_block[0]==0 and is_volatile_block[0]==1:
                    continue #skip vol first
            
            # Load learning rate for each sess
            # learning_rate_df = utils.load_csv('LearningRates',session,DATA_FOLDER)
            learning_rate_df = utils.load_csv('LearningRatesGridSearch',session,DATA_FOLDER)
            self.lr_dfs.append(learning_rate_df)
       
        self.allsess_lr_df = utils.merge_sessions_df(self.lr_dfs)
        # utils.save_csv(self.allsess_lr_df,'LearningRates','',DATA_FOLDER)
        utils.save_csv(self.allsess_lr_df,'LearningRatesGridSearch','',DATA_FOLDER)
        
        if do_stable_first_only:
            suptitle = 'Stable to Volatile Sessions'
        elif do_volatile_first_only:
            suptitle = 'Volatile to Stable Sessions'
        else:
            suptitle = 'All Sessions'
            
        # Plot
        # self.plot_learning_rate_dual_violins(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        # self.plot_learning_rate_dual_hists_new(self.allsess_lr_df,suptitle,self.fig_folder)
        # self.plot_learning_rate_single_hists_new(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        # self.plot_winbias_violins(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        # self.plot_learning_rate_scatter_1(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        # self.plot_learning_rate_scatter_2(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        # self.plot_learning_rate_scatter_all_combined(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        # self.plot_winbias_hist(self.allsess_lr_df,self.allsess_nt_df,suptitle,self.fig_folder)
        # self.new_scatter_hists(self.allsess_lr_df)
        
        self.plot_learning_rate_hist_neuron_type_single(self.allsess_lr_df,'Qchosen','Qchosen - all sess',self.fig_folder)
        self.plot_learning_rate_hist_neuron_type_single(self.allsess_lr_df,'Q1','Q1 - all sess',self.fig_folder)
        self.plot_learning_rate_hist_neuron_type_single(self.allsess_lr_df,'Q2','Q2 - all sess',self.fig_folder)
        
        # self.plot_lr_vs_rsqr(self.allsess_lr_df,suptitle)
        return
    
    
    
    
    
    def new_scatter_hists(self,learning_rate_df):
        
        condition = 'all trials'   # 'stable block', 'volatile block', or 'all trials'
        
        for area in AREAS:
            
            area_df = utils.area_parser(learning_rate_df,area)
            df = area_df.dropna(subset=['Neuron Type'])
            df = df[df['Unit'].str.endswith(condition)]
            neg, pos, r2 = df['Neg Learning Rate'], df['Pos Learning Rate'], df['R-Squared']
            
            fig = plt.figure(figsize=(7, 7))
            gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 0.2], height_ratios=[1, 4],
                                  hspace=0.05, wspace=0.1)
            ax = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
            cax = fig.add_subplot(gs[1, 2])
            
            sc = ax.scatter(neg, pos, c=r2, cmap='inferno', edgecolor='k', s=50)
            ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel='Negative learning rate', ylabel='Positive learning rate')
            
            bins = np.linspace(0, 1, 21)
            ax_top.hist(neg, bins=bins, color='gray', edgecolor='k')
            ax_right.hist(pos, bins=bins, color='gray', edgecolor='k', orientation='horizontal')
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            ax_top.set_ylabel('Num Neurons')
            ax_right.set_xlabel('Num Neurons')
            
            fig.colorbar(sc, cax=cax, label='R²')
            fig.suptitle(condition + '\n' + area)
            plt.show()
        
        return
    
    
    

    @staticmethod
    def plot_learning_rate_hist_neuron_type(learning_rate_df, neuron_type, suptitle, fig_folder):
        
        # areas = AREAS + ['all areas']
        areas = AREAS
        # cols = ['Learning Rate for Qchosen']
    
        for i, area in enumerate(areas):
    
            # area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            #get all neurons which encode specified neuron_type
            area_type_learning_rate_df = area_learning_rate_df[area_learning_rate_df['Neuron Type'].str.contains(neuron_type)]

            #all trials
            fig,axs=plt.subplots(1,2,layout='constrained')
            
            for i,pos_neg in enumerate(['Pos','Neg']):
                
                # Get stable and volatile lrs
                alltr_learning_rate_df = area_type_learning_rate_df[area_type_learning_rate_df['Unit'].str.contains('all trials')]
                alltrs = pd.to_numeric(alltr_learning_rate_df[f'{pos_neg} Learning Rate'].values)
                #hists
                ax=axs[i]
                ax.hist(alltrs,label=f'all trials, n={len(alltrs)}',bins=20,range=(0,1),alpha=0.5,color='tab:green')
                #mean markers
                y_max1 = np.max(np.histogram(alltrs,bins=20,range=(0,1))[0])
                y_max = np.max(y_max1)
                ax.plot(np.mean(alltrs),y_max,marker='v',color='tab:green')
                #plot details
                ax.set_title(f'{pos_neg} LR Hist')
                ax.set_ylabel('Number of neurons')
                ax.set_xlabel('Learning Rate')
                ax.legend()
                fig.suptitle(suptitle + f'\n{area}')
            
            
            #stable vs volatile overlay
            fig,axs=plt.subplots(1,2,layout='constrained')
            
            for i,pos_neg in enumerate(['Pos','Neg']):
                
                # Get stable and volatile lrs
                stab_learning_rate_df = area_type_learning_rate_df[area_type_learning_rate_df['Unit'].str.contains('stable')]
                vol_learning_rate_df = area_type_learning_rate_df[area_type_learning_rate_df['Unit'].str.contains('volatile')]
                stabs = pd.to_numeric(stab_learning_rate_df[f'{pos_neg} Learning Rate'].values)
                vols = pd.to_numeric(vol_learning_rate_df[f'{pos_neg} Learning Rate'].values)
                
                #stats
                res = sp.stats.ttest_ind(stabs, vols, equal_var=True)
                #hists
                ax=axs[i]
                ax.hist(vols,label=f'volatile, n={len(vols)}',bins=20,range=(0,1),alpha=0.5,color='tab:blue')
                ax.hist(stabs,label=f'stable, n={len(stabs)}',bins=20,range=(0,1),alpha=0.5,color='tab:orange')
                #mean markers
                y_max1 = np.max(np.histogram(vols,bins=20,range=(0,1))[0])
                y_max2 = np.max(np.histogram(stabs,bins=20,range=(0,1))[0])
                y_max = np.max([y_max1,y_max2])
                ax.plot(np.mean(vols),y_max,marker='v',color='tab:blue')
                ax.plot(np.mean(stabs),y_max,marker='v',color='tab:orange')
                #plot details
                ax.set_title(f'{pos_neg} LR Hist\np={res.pvalue:.4f}')
                ax.set_ylabel('Number of neurons')
                ax.set_xlabel('Learning Rate')
                ax.legend()
                fig.suptitle(suptitle + f'\n{area}')
                
        return  


    @staticmethod
    def plot_learning_rate_hist_neuron_type_single(learning_rate_df, neuron_type, suptitle, fig_folder):
        
        # areas = AREAS + ['all areas']
        areas = AREAS
        # cols = ['Learning Rate for Qchosen']
    
        for i, area in enumerate(areas):
    
            # area_neuron_type_df = utils.area_parser(neuron_type_df, area)
            area_learning_rate_df = utils.area_parser(learning_rate_df, area)
    
            print(area)
            if area_learning_rate_df.empty:
                print('skipped!')
                continue
    
            #get all neurons which encode specified neuron_type
            area_type_learning_rate_df = area_learning_rate_df[area_learning_rate_df['Neuron Type'].str.contains(neuron_type)]

            #all trials
            fig,ax=plt.subplots(layout='constrained')
                
            # Get stable and volatile lrs
            alltr_learning_rate_df = area_type_learning_rate_df[area_type_learning_rate_df['Unit'].str.contains('all trials')]
            alltrs = pd.to_numeric(alltr_learning_rate_df['Learning Rate'].values)
            # alltrs = pd.to_numeric(alltr_learning_rate_df['Learning Rate'].str.removeprefix('[').str.removesuffix(']').values)
            #hists
            ax.hist(alltrs,label=f'all trials, n={len(alltrs)}',bins=20,range=(0,1),alpha=0.5,color='tab:green')
            #mean markers
            y_max1 = np.max(np.histogram(alltrs,bins=20,range=(0,1))[0])
            y_max = np.max(y_max1)
            ax.plot(np.mean(alltrs),y_max,marker='v',color='tab:green')
            #plot details
            ax.set_title('LR Hist')
            ax.set_ylabel('Number of neurons')
            ax.set_xlabel('Learning Rate')
            ax.legend()
            fig.suptitle(suptitle + f'\n{area}')
            
            
            #stable vs volatile overlay
            fig,ax=plt.subplots(layout='constrained')
                
            # Get stable and volatile lrs
            stab_learning_rate_df = area_type_learning_rate_df[area_type_learning_rate_df['Unit'].str.contains('stable')]
            vol_learning_rate_df = area_type_learning_rate_df[area_type_learning_rate_df['Unit'].str.contains('volatile')]
            stabs = pd.to_numeric(stab_learning_rate_df['Learning Rate'].values)
            vols = pd.to_numeric(vol_learning_rate_df['Learning Rate'].values)
            # stabs = pd.to_numeric(stab_learning_rate_df['Learning Rate'].str.removeprefix('[').str.removesuffix(']').values)
            # vols = pd.to_numeric(vol_learning_rate_df['Learning Rate'].str.removeprefix('[').str.removesuffix(']').values)
            
            #stats
            res = sp.stats.ttest_ind(stabs, vols, equal_var=True)
            #hists
            ax.hist(vols,label=f'volatile, n={len(vols)}',bins=20,range=(0,1),alpha=0.5,color='tab:blue')
            ax.hist(stabs,label=f'stable, n={len(stabs)}',bins=20,range=(0,1),alpha=0.5,color='tab:orange')
            #mean markers
            y_max1 = np.max(np.histogram(vols,bins=20,range=(0,1))[0])
            y_max2 = np.max(np.histogram(stabs,bins=20,range=(0,1))[0])
            y_max = np.max([y_max1,y_max2])
            ax.plot(np.mean(vols),y_max,marker='v',color='tab:blue')
            ax.plot(np.mean(stabs),y_max,marker='v',color='tab:orange')
            #plot details
            ax.set_title(f'LR Hist\np={res.pvalue:.4f}')
            ax.set_ylabel('Number of neurons')
            ax.set_xlabel('Learning Rate')
            ax.legend()
            fig.suptitle(suptitle + f'\n{area}')
                
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
    
def plot_tuning_curves_value(fr_df,behavior_df,learning_rate_df):
    assert len(fr_df) == len(behavior_df)
    
    # units = list(fr_df.keys())
    # units = utils.get_unique_units(learning_rate_df["Unit"], SUBSESSIONS)
    subsession = 'all trials'
    
    #only plot neurons that were found to encode during specified subsession
    units = learning_rate_df["Unit"].loc[learning_rate_df["Unit"].str.endswith(subsession)]
    units = units.str.removesuffix(f' - {subsession}')
    
    if len(units) < 1:
        return None #skip whole thing if there aren't any units for this session
    
    for i,unit in enumerate(units): #loop thru all units of session
    
        neuron_type = str(utils.get_unit_data(learning_rate_df,f'{unit} - {subsession}','Neuron Type'))
        # print(unit)
        # print(neuron_type)
        
        if 'Q' not in neuron_type: #only do for value neurons
            continue
        
        if 'Unit A' in unit: #exclude vmPFC
            continue
        
        #get values relevant to neuron type
        values = behavior_df[neuron_type]
        
        exclude = 0.05
        excl_top = 1-exclude
        include = (values>exclude) & (values<excl_top)
        
        x = values.iloc[include]
        y = fr_df[unit].iloc[include]
        
        #find the trials corresponding to value bins, then get values and frs of those trials
        value_bins = np.linspace(0, 1, 6)
        binned_value_avg = np.zeros(len(value_bins)-1)
        binned_value_sem = np.zeros_like(binned_value_avg)
        binned_fr_avg = np.zeros_like(binned_value_avg)
        binned_fr_sem = np.zeros_like(binned_value_avg)
        for j, (bot, top) in enumerate(zip(value_bins[:-1], value_bins[1:])):
            trials = np.nonzero((x>bot)&(x<top))[0]
            binned_value_avg[j], binned_value_sem[j] = utils.get_avg_sem(x.iloc[trials],axis=0)
            binned_fr_avg[j], binned_fr_sem[j] = utils.get_avg_sem(y.iloc[trials],axis=0)
        
        #plotit
        fig,ax = plt.subplots(layout="constrained")
        ax.errorbar(binned_value_avg, binned_fr_avg, xerr=binned_value_sem, yerr=binned_fr_sem,color='k')

        ax.scatter(x,y,alpha=0.05,color='k')
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'-',color='red')
        r,pval = sp.stats.pearsonr(x,y)
        ax.set_xlabel(neuron_type)
        ax.set_ylabel('Z-scored FR')
        ax.set_title(f'Value encoding tuning curve. Outer {exclude} excluded.\nRSqr={r**2}')
        ax.set_ylim([-1,1])
        
        fig.suptitle(f'{unit}\n{neuron_type} encoding unit')
        plt.show()
    
    
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
    

