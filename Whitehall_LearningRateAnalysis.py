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
# import statsmodels.api as sm
# import statsmodels.formula.api as sm_api
# import matplotlib.lines as mlines
from matplotlib import pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import BoundaryNorm
# from numpy.polynomial.polynomial import polyfit

import Whitehall_utils as utils
from file_info import get_block_info, get_area_info
# import DecisionMakingBehavior_Whitehall as BehaviorAnalysis
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

class Process_LearningRates:
    
    def __init__(self,
                    value_model='DualBaselineSide',
                    do_even_num_trials=True,
                    normalize_within_trials=False,
                    normalize_across_trials=True,
                    encoding_alpha_thresh=0.005, #significance threshold for each timebin
                    n_consec=3, #num of consecutive significant time bins needed to be determined as encoding):     
                    file_save_name='LearningRates'):
       
        self.lr_dfs = []
        for s,session in enumerate(SESSIONS):
            
            start = time.time()

            try:
                lr_grid_behav_dict = utils.load_pkl('LR_Grid_behavior_dfs_dict',session,DATA_FOLDER)
            except:
                print('Need to run get_learning_rate_grid_behavior first!')
                return
            
            # Get learning rate for each unit
            Spikes, Behav, _ = get_Spikes_Behav_LFP(s,session)
            
            learning_rate_df = self.get_learning_rates(Spikes.psth_dict,Behav,lr_grid_behav_dict,
                                   value_model='DualBaselineSide',
                                   do_even_num_trials=True,
                                   normalize_within_trials=False,
                                   normalize_across_trials=True,
                                   encoding_alpha_thresh=0.005, #significance threshold for each timebin
                                   n_consec=3, #num of consecutive significant time bins needed to be determined as encoding):     
                                   )
                
            utils.save_csv(learning_rate_df,file_save_name,session,DATA_FOLDER)

            utils.get_time_left_big(start,s,len(SESSIONS))
                
            self.lr_dfs.append(learning_rate_df)
       
        self.allsess_lr_df = utils.merge_sessions_df(self.lr_dfs)
        utils.save_csv(self.allsess_lr_df,file_save_name,'',DATA_FOLDER)
        
        return 
    
    @staticmethod
    def get_learning_rate_grid_behavior():
        for s,session in enumerate(SESSIONS):
            
            start = time.time()
            Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
            Behav.get_learning_rate_grid_behavior()
            utils.get_time_left(start,s,len(SESSIONS))
            
        return


    @staticmethod    
    def get_learning_rates(psth_dict,Behavior,lr_grid_behav_dict,
                           value_model='DualBaselineSide',
                           do_even_num_trials=True,
                           normalize_within_trials=False,
                           normalize_across_trials=True,
                           encoding_alpha_thresh=0.005, #significance threshold for each timebin
                           n_consec=3, #num of consecutive significant time bins needed to be determined as encoding):     
                           ):
        
        ## Define functions that will be used here
        def _timebin_lin_reg(psth,behav_df,value_type):
            # regress fr against the given behavior calculated for this learning rate and set of regressors
            # do this for each timebin
            
            fits = [utils.linear_encoding_regression(psth[:,k],behav_df,normalize_across_trials) for k in range(N_TIMEBINS)]
            
            fit_fpvals = np.array([f.f_pvalue for f in fits])
            reg_pvals = np.array([f.pvalues[value_type] for f in fits])
            beta_coefs = np.array([f.params[value_type] for f in fits])
            rsqrs = np.array([f.rsquared for f in fits])
            lls = np.array([f.llf for f in fits])
            
            return fit_fpvals, reg_pvals, beta_coefs, rsqrs, lls
        
        def _lr_timebin_lin_reg(learning_rates,Behavior,psth,trials,regressors):
            # Do timebin_lin_reg() but for any given learning rate(s)
            
            #get behav with any lrs for the current subsess
            Behavior.get_behavior(Q_learning=True,learning_rates=learning_rates,value_model=value_model)
            behav_subsess = Behavior.behavior_df[regressors].iloc[trials]
            
            # regress fr against the behavior calculated for this learning rate and set of regressors
            # do this for each timebin
            fit_fpvals, reg_pvals, beta_coefs, rsqrs, lls = _timebin_lin_reg(psth_subsess,behav_subsess,neuron_type)
            
            return fit_fpvals, reg_pvals, beta_coefs, rsqrs, lls
        
        def _best_ll_lr_timebin_lin_reg(learning_rates,Behavior,psth,trials,regressors):
            # Do lr_timebin_lin_reg() and return the avg of the best N log likelihoods, where N=ENCODING_NCONSEC
            
            _, _, _, _, lls = _lr_timebin_lin_reg(learning_rates,Behavior,psth,trials,regressors)

            return utils.mean_of_top_n_of_arr(lls,n_consec)
        
        
        
        
    
        units = psth_dict.keys
        print(f'{len(units)} neurons')
        
        #this is an array of lrs to test
        learning_rate_grid = lr_grid_behav_dict[value_model]['learning rate grid'] 
        
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
                
                #get psths for unit
                psths = psth_dict[unit] #shape = n_trials x n_timebins
                assert psths.shape == (len(Behavior.behavior_df), N_TIMEBINS)
                
                #normalize if specified
                if normalize_within_trials:
                    psths = sp.stats.zscore(psths,axis=1)
                    
                #get trials for subsession
                psth_subsess = psths[trials,:] #psths for all subsession trials
                
                ## Dual LR Case
                if 'Dual' in value_model: 
                
                    ## Do regression for all learning rate combos
                    ## Keep regressor set and learning rate(s) that fit best
                    best_rsqr = 0.
                    best_learning_rates = ['None','None']
                    neuron_type = 'None'
                
                    ##Grid search
                    for pos_alpha in learning_rate_grid:
                        for neg_alpha in learning_rate_grid:
                            
                            pos_alpha, neg_alpha = np.round([pos_alpha,neg_alpha],decimals=2) # to help with floating point errors
                            behavior_df = lr_grid_behav_dict[value_model]['data'][(pos_alpha, neg_alpha)]
                            
                            # get set of regressors 
                            for value_type,regressors in REGRESSOR_SUPERSET_DICT.items():
                                
                                behav_subsess = behavior_df[regressors].iloc[trials]       
                                
                                # regress fr against the behavior calculated for this learning rate and set of regressors
                                # do this for each timebin
                                fit_fpvals, reg_pvals, beta_coefs, rsqrs, lls = _timebin_lin_reg(psth_subsess,behav_subsess,neuron_type,N_TIMEBINS)
                                
                                # only consider if regression and value regressor are both significant
                                is_signif,signif_bin_idxs = utils.find_sig_bins(fit_fpvals, reg_pvals, encoding_alpha_thresh, n_consec)
                                if is_signif: 
                                    
                                    #avg of best rsqrs only. So as to not punish if more bins are signif but with lower rsqr
                                    avg_rsqr = utils.mean_of_top_n_of_arr(rsqrs,n_consec) 
                                    
                                    # take lr and value type if it yields the best rsqr so far, otherwise continue on
                                    if avg_rsqr > best_rsqr:
                                        best_rsqr = avg_rsqr
                                        best_learning_rates = [pos_alpha,neg_alpha]
                                        neuron_type = value_type
                                        betas = beta_coefs[signif_bin_idxs]
                                        bin_idxs = signif_bin_idxs
    
                
                    ## Refine grid search using optimizer
                    if neuron_type != 'None':
                        regressors = REGRESSOR_SUPERSET_DICT[neuron_type]
                        
                        nll = lambda *args: -1*_best_ll_lr_timebin_lin_reg(*args)
                        x0 = best_learning_rates  # [pos_alpha, neg_alpha]
                        result_A = sp.optimize.minimize(nll, x0, 
                        		   args=(Behavior,psth_subsess,trials,regressors),
                        		   bounds=[(0.01,1), (0.01,1)],
                        		   method='Nelder-Mead',
                                   tol=1e-4)
                        [pos_alpha,neg_alpha] = result_A["x"]
                        
                        ## Redo timebin regs with optimized lrs to see if it is actually better
                        fit_fpvals, reg_pvals, beta_coefs, rsqrs, lls = _lr_timebin_lin_reg([pos_alpha,neg_alpha],Behavior,psth_subsess,neuron_type,regressors)
                        
                        # only consider if regression and value regressor are both significant
                        is_signif,signif_bin_idxs = utils.find_sig_bins(fit_fpvals, reg_pvals, encoding_alpha_thresh, n_consec)
                        if is_signif: 
                            
                            #avg of best rsqrs only. So as to not punish if more bins are signif but with lower rsqr
                            avg_rsqr = utils.mean_of_top_n_of_arr(rsqrs,n_consec) 
                            
                            # take lr and value type if it yields the best rsqr so far
                            if avg_rsqr > best_rsqr:
                                best_rsqr = avg_rsqr
                                best_learning_rates = [pos_alpha,neg_alpha]
                                neuron_type = value_type
                                betas = beta_coefs[signif_bin_idxs]
                                bin_idxs = signif_bin_idxs
                            
                    
                    #save out info for unit
                    lr_row_dict = {'Unit': unit,
                                   'Subsession': stable_or_volatile,
                                   'Neuron Type': neuron_type,
                                   'Pos Learning Rate': best_learning_rates[0],
                                   'Neg Learning Rate': best_learning_rates[1],
                                   'Timebin Indices': bin_idxs,
                                   'Beta Coefs': betas,
                                   'R-Squared': best_rsqr}
                    df_rows_learning_rates.append(lr_row_dict)
                    
                    
                    
                ## Single LR Case    
                if 'Dual' not in value_model:
                    
                    ## Do regression for all learning rate combos
                    ## Keep regressor set and learning rate(s) that fit best
                    best_rsqr = 0.
                    best_learning_rates = 'None'
                    neuron_type = 'None'
                
                    ##Grid search
                    for alpha in learning_rate_grid:
                            
                        alpha = np.round(alpha,decimals=2) # to help with floating point errors
                        behavior_df = lr_grid_behav_dict[value_model]['data'][(alpha)]
                        
                        # get set of regressors 
                        for value_type,regressors in REGRESSOR_SUPERSET_DICT.items():
                            
                            behav_subsess = behavior_df[regressors].iloc[trials]       
                            
                            # regress fr against the behavior calculated for this learning rate and set of regressors
                            # do this for each timebin
                            fit_fpvals, reg_pvals, beta_coefs, rsqrs, lls = _timebin_lin_reg(psth_subsess,behav_subsess,neuron_type,N_TIMEBINS)
                            
                            # only consider if regression and value regressor are both significant
                            is_signif,signif_bin_idxs = utils.find_sig_bins(fit_fpvals, reg_pvals, encoding_alpha_thresh, n_consec)
                            if is_signif: 
                                
                                #avg of best rsqrs only. So as to not punish if more bins are signif but with lower rsqr
                                avg_rsqr = utils.mean_of_top_n_of_arr(rsqrs,n_consec) 
                                
                                # take lr and value type if it yields the best rsqr so far, otherwise continue on
                                if avg_rsqr > best_rsqr:
                                    best_rsqr = avg_rsqr
                                    best_learning_rates = [alpha]
                                    neuron_type = value_type
                                    betas = beta_coefs[signif_bin_idxs]
                                    bin_idxs = signif_bin_idxs
    
                
                    ## Refine grid search using optimizer
                    if neuron_type != 'None':
                        regressors = REGRESSOR_SUPERSET_DICT[neuron_type]
                        
                        nll = lambda *args: -1*_best_ll_lr_timebin_lin_reg(*args)
                        x0 = best_learning_rates  # [alpha]
                        result_A = sp.optimize.minimize(nll, x0, 
                        		   args=(Behavior,psth_subsess,trials,regressors),
                        		   bounds=[(0.01,1), (0.01,1)],
                        		   method='Nelder-Mead',
                                   tol=1e-4)
                        [alpha] = result_A["x"]
                        
                        ## Redo timebin regs with optimized lrs to see if it is actually better
                        fit_fpvals, reg_pvals, beta_coefs, rsqrs, lls = _lr_timebin_lin_reg([alpha],Behavior,psth_subsess,neuron_type,regressors)
                        
                        # only consider if regression and value regressor are both significant
                        is_signif,signif_bin_idxs = utils.find_sig_bins(fit_fpvals, reg_pvals, encoding_alpha_thresh, n_consec)
                        if is_signif: 
                            
                            #avg of best rsqrs only. So as to not punish if more bins are signif but with lower rsqr
                            avg_rsqr = utils.mean_of_top_n_of_arr(rsqrs,n_consec) 
                            
                            # take lr and value type if it yields the best rsqr so far
                            if avg_rsqr > best_rsqr:
                                best_rsqr = avg_rsqr
                                best_learning_rates = [alpha]
                                neuron_type = value_type
                                betas = beta_coefs[signif_bin_idxs]
                                bin_idxs = signif_bin_idxs
                            
                    
                    #save out info for unit
                    lr_row_dict = {'Unit': unit,
                                   'Subsession': stable_or_volatile,
                                   'Neuron Type': neuron_type,
                                   'Learning Rate': best_learning_rates[0],
                                   'Timebin Indices': bin_idxs,
                                   'Beta Coefs': betas,
                                   'R-Squared': best_rsqr}
                    df_rows_learning_rates.append(lr_row_dict)
                
                utils.get_time_left(start,i,len(units))
                                
        learning_rate_df = pd.DataFrame(df_rows_learning_rates) #rows: units x subsessions
        # utils.save_csv(learning_rate_df,'LR',"debug",DATA_FOLDER)

        return learning_rate_df


class LearningRate_Analysis:
    
    def __init__(self,file_name,do_stable_first_only=False,do_volatile_first_only=False):
        
        self.fig_folder = 'LearningRates'
        self.file_name = file_name
       
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
            learning_rate_df = utils.load_csv(file_name,session,DATA_FOLDER)
            self.lr_dfs.append(learning_rate_df)
       
        self.allsess_lr_df = utils.merge_sessions_df(self.lr_dfs)
        
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
        
        self.plot_learning_rate_hist_neuron_type(file_name,self.allsess_lr_df,'Qchosen',f'{file_name}\nQchosen - {suptitle}',self.fig_folder)
        self.plot_learning_rate_hist_neuron_type(file_name,self.allsess_lr_df,'Q1',f'{file_name}\nQ1 - {suptitle}',self.fig_folder)
        self.plot_learning_rate_hist_neuron_type(file_name,self.allsess_lr_df,'Q2',f'{file_name}\nQ2 - {suptitle}',self.fig_folder)
        
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
    def plot_learning_rate_hist_neuron_type(file_name, learning_rate_df, neuron_type, suptitle, fig_folder):
        
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

            if 'dual' in file_name:

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
                    
                    
                    
                if 'dual' not in file_name:
                    
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
 
    @staticmethod    
    def run_scatter_plot_tuning_curves(session_to_plot,file_name):
        
        for s,session in enumerate(SESSIONS):

            #choose which session to do (optional)
            if session_to_plot:
                cond = session == session_to_plot
            else:
                cond = True
            
            if cond:
                start = time.time()
                print(session)
                
                Spikes, Behav, LFP = get_Spikes_Behav_LFP(s,session)
                learning_rate_df = utils.load_csv(file_name,session,DATA_FOLDER)
                
                print('data loaded')
                
                LearningRate_Analysis.plot_scatter_tuning_curves_value(Spikes.fr_df,Behav.behavior_df,learning_rate_df)
                
                utils.get_time_left(start,s,len(SESSIONS))        
            
        return


    @staticmethod    
    def plot_scatter_tuning_curves_value(fr_df,behavior_df,learning_rate_df):
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
    
    def rainbow_plots():
        salkdjf #move plot_temporal_encoding and do_rainbow_plots from encoding analysis to here.
        #make it so taht plot_temporal_encoding loops through units like plot_scatter_tuning_curves_valeu does.

        return
    

 
    
#%% Run

### Get behavior for all learning rate grids (needs only be run once)
# Process_LearningRates.get_learning_rate_grid_behavior()


### Get best learning rate for each neuron

file_name = 'LearningRates_dualbaslineside_evenntrials_acrosstrnorm_alpha005_nconsec3'
_ = Process_LearningRates(  value_model='DualBaselineSide',
                            do_even_num_trials=True,
                            normalize_within_trials=False,
                            normalize_across_trials=True,
                            encoding_alpha_thresh=0.005,
                            n_consec=3,     
                            file_save_name=file_name)

file_name = 'LearningRates_baslineside_evenntrials_acrosstrnorm_alpha005_nconsec3'
_ = Process_LearningRates(  value_model='BaselineSide',
                            do_even_num_trials=True,
                            normalize_within_trials=False,
                            normalize_across_trials=True,
                            encoding_alpha_thresh=0.005,
                            n_consec=3,     
                            file_save_name=file_name)

file_name = 'LearningRates_dualbaslineside_evenntrials_bothnorm_alpha005_nconsec3'
_ = Process_LearningRates(  value_model='DualBaselineSide',
                            do_even_num_trials=True,
                            normalize_within_trials=True,
                            normalize_across_trials=True,
                            encoding_alpha_thresh=0.005,
                            n_consec=3,     
                            file_save_name=file_name)


### Analyze learning rates

# file_name = 'LearningRates_dualbaslineside_evenntrials_acrosstrnorm_alpha005_nconsec3'
# _ = LearningRate_Analysis(file_name,do_stable_first_only=False,do_volatile_first_only=False)

# file_name = 'LearningRates_baslineside_evenntrials_acrosstrnorm_alpha005_nconsec3'
# _ = LearningRate_Analysis(file_name,do_stable_first_only=False,do_volatile_first_only=False)

# file_name = 'LearningRates_dualbaslineside_evenntrials_bothnorm_alpha005_nconsec3'
# _ = LearningRate_Analysis(file_name,do_stable_first_only=False,do_volatile_first_only=False)



