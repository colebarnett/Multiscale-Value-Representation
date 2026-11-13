# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 17:46:55 2026

@author: coleb

Utility functions to be used with Whitehall.py
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
# import sys
# import gif
# import copy
import time
import pickle
import sklearn
# import neurodsp
# import matplotlib
import numpy as np
# import scipy as sp
import pandas as pd
from scipy.stats import zscore
# import matplotlib_venn
import statsmodels.api as sm
# from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
# from statsmodels.stats.outliers_influence import variance_inflation_factor

#### Utility Funcs
def get_time_left(start_time,itr,n_itrs):
    end_time = time.time()
    time_taken = end_time - start_time 
    iters_left = n_itrs - itr - 1
    time_left = np.round(time_taken * iters_left / 60,decimals=1)
    print(f'--------------{itr+1}/{n_itrs} - {time_left} mins left--------------')
    return

def flatten(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result

def get_avg_sem(arr,axis):
    avg = np.nanmean(arr,axis=axis)
    sem = np.nanstd(arr,axis=axis) / np.sqrt(np.shape(arr)[axis])
    return avg,sem

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

def get_block_borders(behavior_df):
    borders = np.nonzero(np.diff(behavior_df['Rew_prob_1']))[0]
    return borders

def get_trials_rewprob_volatility(df,stable_or_volatile,rew_prob1): 
    if stable_or_volatile == 'all trials':
        return list(df.loc[(np.isclose(df['Rew_prob_1'],rew_prob1))].index)
    elif stable_or_volatile == 'stable block':
        return list(df.loc[(df['Stable']==1) & (np.isclose(df['Rew_prob_1'],rew_prob1))].index)
    elif stable_or_volatile == 'volatile block':
        return list(df.loc[(df['Volatile']==1) & (np.isclose(df['Rew_prob_1'],rew_prob1))].index)
    else:
        raise ValueError("Invalid value for stable_or_volatile")

def get_trials_rewprob_volatility_Qchosen(df,stable_or_volatile,rew_prob):
    if stable_or_volatile == 'all trials':
        list1 = list(df.loc[(np.isclose(df['Rew_prob_1'],rew_prob)) & (df['Choice1']==1)].index)
        list2 = list(df.loc[(np.isclose(df['Rew_prob_2'], rew_prob)) & (df['Choice2'] == 1)].index)
    elif stable_or_volatile == 'stable block':
        list1 = list(df.loc[(df['Stable']==1) & (np.isclose(df['Rew_prob_1'],rew_prob)) & (df['Choice1']==1)].index)
        list2 = list(df.loc[(df['Stable'] == 1) & (np.isclose(df['Rew_prob_2'], rew_prob)) & (df['Choice2'] == 1)].index)
    elif stable_or_volatile == 'volatile block':
        list1 = list(df.loc[(df['Volatile'] == 1) & (np.isclose(df['Rew_prob_1'], rew_prob)) & (df['Choice1'] == 1)].index)
        list2 = list(df.loc[(df['Volatile'] == 1) & (np.isclose(df['Rew_prob_2'], rew_prob)) & (df['Choice2'] == 1)].index)
    else:
        raise ValueError("Invalid value for stable_or_volatile")
    return list1 + list2
   
def area_parser(df_or_dict,brain_area):
    
    if type(df_or_dict) == dict:
        
        if 'unit_labels' in df_or_dict.keys():

            match brain_area:
                case 'vmPFC':
                    idxs = [idx for idx,label in enumerate(df_or_dict['unit_labels']) if 'Unit A' in label]
                case 'Cd':
                    idxs = [idx for idx,label in enumerate(df_or_dict['unit_labels']) if 'Unit C' in label]
                case 'OFC':
                    idxs = [idx for idx,label in enumerate(df_or_dict['unit_labels']) if 'Unit D' in label]
                case 'all areas':
                    idxs = [idx for idx,label in enumerate(df_or_dict['unit_labels']) if 'Unit' in label]
                case _:
                    raise ValueError('Invalid brain area')
            
            subset_df_or_dict = {} 
            for key,list_ in df_or_dict.items():
                subset_df_or_dict[key] = [list_[i] for i in idxs]
                    
        else:
            
            match brain_area:
                case 'vmPFC':
                    subset_df_or_dict = {k:v for k,v in df_or_dict.items() if 'Unit A' in k}
                case 'Cd':
                    subset_df_or_dict = {k:v for k,v in df_or_dict.items() if 'Unit C' in k}
                case 'OFC':
                    subset_df_or_dict = {k:v for k,v in df_or_dict.items() if 'Unit D' in k}
                case 'all areas':
                    subset_df_or_dict = df_or_dict
                case _:
                    raise ValueError('Invalid brain area')
        
                
    if type(df_or_dict)==pd.DataFrame:
        
        if 'Unit_labels' in df_or_dict.columns: #unit=row organization
            match brain_area:
                case 'vmPFC':
                    subset_df_or_dict = df_or_dict[df_or_dict['Unit_labels'].str.contains('Unit A')]
                case 'Cd':
                    subset_df_or_dict = df_or_dict[df_or_dict['Unit_labels'].str.contains('Unit C')]
                case 'OFC':
                    subset_df_or_dict = df_or_dict[df_or_dict['Unit_labels'].str.contains('Unit D')]
                case 'all areas':
                    subset_df_or_dict = df_or_dict
                case _:
                    raise ValueError('Invalid brain area')
                    
        elif 'Unit' in df_or_dict.columns: #unit=row organization
            match brain_area:
                case 'vmPFC':
                    subset_df_or_dict = df_or_dict[df_or_dict['Unit'].str.contains('Unit A')]
                case 'Cd':
                    subset_df_or_dict = df_or_dict[df_or_dict['Unit'].str.contains('Unit C')]
                case 'OFC':
                    subset_df_or_dict = df_or_dict[df_or_dict['Unit'].str.contains('Unit D')]
                case 'all areas':
                    subset_df_or_dict = df_or_dict
                case _:
                    raise ValueError('Invalid brain area')
                    
        else: #unit=col organization
            match brain_area:
                case 'vmPFC':
                    subset_df_or_dict = df_or_dict.filter(like='Unit A',axis=1)
                case 'Cd':
                    subset_df_or_dict = df_or_dict.filter(like='Unit C',axis=1)
                case 'OFC':
                    subset_df_or_dict = df_or_dict.filter(like='Unit D',axis=1)
                case 'all areas':
                    subset_df_or_dict = df_or_dict
                case _:
                    raise ValueError('Invalid brain area')
      
    return subset_df_or_dict

def count_units_area(dict_,brain_area):
    
    assert 'unit_labels' in dict_.keys()
    
    match brain_area:
        case 'vmPFC':
            n_units = sum(['Unit A' in label for label in dict_['unit_labels']])
        case 'Cd':
            n_units = sum(['Unit C' in label for label in dict_['unit_labels']])
        case 'OFC':
            n_units = sum(['Unit D' in label for label in dict_['unit_labels']])
        case 'all areas':
            n_units = sum(['Unit' in label for label in dict_['unit_labels']])
        case _:
            raise ValueError('Invalid brain area')
    
    return n_units

def get_trials(behav_df,stable_or_volatile):
    
    all_trials = np.arange(len(behav_df))
    
    match stable_or_volatile:
        case 'stable block':
            trials = all_trials[behav_df['Stable']==1]
        case 'volatile block':
            trials = all_trials[behav_df['Volatile']==1]
        case 'all trials':
            trials = all_trials
        case _:
            raise ValueError('Invalid subsession')
            
    return trials

def get_trials_even_num(behav_df,stable_or_volatile):
    
    all_trials = np.arange(len(behav_df))
    
    match stable_or_volatile:
        case 'stable block':
            trials = all_trials[behav_df['Stable']==1]
        case 'volatile block':
            trials = all_trials[behav_df['Volatile']==1]
        case 'all trials':
            trials = all_trials
        case _:
            raise ValueError('Invalid subsession')
    if stable_or_volatile != 'all trials': 
        num_stable_trials = len(all_trials[behav_df['Stable']==1])
        num_volatile_trials = len(all_trials[behav_df['Volatile']==1])
        lowest_num_trials = np.min([num_stable_trials,num_volatile_trials])
        trials = trials[:lowest_num_trials]
            
    return trials

def get_unique_units(unit_list,SUBSESSIONS):
    for subsession in SUBSESSIONS:
        unit_list = [unit.removesuffix(f' - {subsession}') for unit in unit_list]
        
    return np.unique(np.array(unit_list))

def get_unit_data(df,unit,heading):
    assert unit in df['Unit'].values, f'unit {unit} not found! missing subsession?'
    assert heading in df.columns, f'df heading {heading} not found!'
    desired_row = df[df['Unit']==unit]
    if len(desired_row)>0:
        desired_value = desired_row[heading].values[0]
    else:
        # desired_value = ''
        desired_value = np.nan

    return desired_value

def calc_percent_encoding(pval_df,alpha_threshold):
    pval_df = pval_df.drop('Unit_labels',axis=1)
    df_signif = pval_df < alpha_threshold
    num_signif = df_signif.sum(axis='index')
    num_units = pval_df.shape[0]
    percent_encoding = num_signif / num_units
    return percent_encoding, num_units

#%% Regression
def linear_encoding_regression(spikes,behavior):
    '''
    Parameters
    ----------
    spikes : (1 x num_trials) array of spiking information (psth timebin info or firing rate)
    behavior : (num_trials x num_regressors) array of behavioral information (do not include const. This will be added internally)

    Returns
    -------
    f_pval : float. p-value for entire regression
    pvals : (1 x num_regressors) array of p-values for each regressor
    '''
    assert np.shape(spikes)[0] == np.shape(behavior)[0]
    
    # zscore behavior without losing column titles if possible
    if isinstance(behavior, (pd.DataFrame, pd.Series)):
        behavior_zscr = (behavior - behavior.mean()) / behavior.std()
    else:
        behavior_zscr = zscore(behavior,axis=0)
    
    # try:
    #     model = sm.OLS(zscore(spikes,axis=0), sm.add_constant(behavior,has_constant='add'), missing='raise',hasconst=True)
    # except:
    #     print(behavior)
    #     print(zscore(spikes,axis=0))
    #     xxx
    
    try:
        model = sm.OLS(zscore(spikes,axis=0), sm.add_constant(behavior_zscr,has_constant='add'), missing='raise',hasconst=True)
    except: 
        print(behavior)
        print(spikes)
        raise ValueError
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

def residualize(reg1,reg2):
    res = linear_encoding_regression(reg1,reg2)
    return res.resid #reg1 that cannot be expained by reg2 (is orthogonal to reg2)

def linear_encoding_regression_feature_selection(regressor_list,spikes,behavior_df,alpha_thresh):
    res = linear_encoding_regression(spikes,behavior_df[regressor_list])
    
    # ## take all regressors with signif p value
    # pvals = res.pvalues[1:] #exclude constant (position 0)
    # if res.f_pvalue < alpha_thresh:
    #     signif_regressors = str([str(feat) for feat in np.array(regressor_list)[pvals < alpha_thresh]])
    #     if len(signif_regressors) < 1:
    #         signif_regressors = 'none'
    # else:
    #     signif_regressors = 'none'
        
    ## take regressor with greatest beta coeff
    betas = abs(res.params[1:]) #exclude constant (position 0)
    pvals = res.pvalues[1:] #exclude constant (position 0)
    if res.f_pvalue < alpha_thresh:
        best_regressor = str(np.array(regressor_list)[np.argmax(betas)])
        
        if np.array(pvals)[np.argmax(betas)] < alpha_thresh:
            signif_regressors = best_regressor
        else:
            signif_regressors = 'none'
    else:
        signif_regressors = 'none'
        
    return signif_regressors


def linear_encoding_regression_feature_selection_new(list_of_regressor_lists,spikes,behavior_df,alpha_thresh):
    '''
    Given a list of possible regressor sets (which all include a different value term),
    return the value term which best explains the neuron FR (as judged by r sqr)
    '''
    
    rsqrs, value_terms = [],[] #save the rsqrs and corresponding value terms from each regression

    for i,regressor_list in enumerate(list_of_regressor_lists):
    
        res = linear_encoding_regression(spikes,behavior_df[regressor_list])
        
        #only consider if regression is significant
        if res.f_pvalue < alpha_thresh:
            
            #find value term
            value_reg_list = [reg for reg in regressor_list if 'Q'==reg[0]]
            assert len(value_reg_list) == 1, 'Pls include only one value term per regression!'
            value_reg = value_reg_list[0]
            
            # only consider if value term is also significant
            if res.pvalues[value_reg] < alpha_thresh:
                
                #if regression and value term are both significant, 
                #then save out rsqr for comparison
                rsqrs.append(res.rsquared)
                value_terms.append(value_reg)
    
    if len(rsqrs)==0:
        best_encoding = 'none'
    else:
        best_encoding = value_terms[np.argmax(rsqrs)]
        
    return best_encoding        

def regularization_regression(spikes,behavior, lasso_or_ridge='lasso', regularization_strength=1e-5, tol=1e-4, verbose=False):
    '''
    Data regularization is a technique to perform feature selection by penalizing model weights (coefficients)
    Lasso (L1 regularization) penalizes with model weights, whereas Ridge (L2 regularization) penalizes with the squares of the weights
    Parameters
    ----------
    spikes : (1 x num_trials) array of spiking information (psth timebin info or firing rate)
    behavior : (num_regressors x num_trials) array of behavioral information (do not include const. This will be added internally)
    lasso_or_ridge : str. Either 'lasso' or 'ridge'
    regularization_strength : float on interval (0,inf), where 0 would indicate ordinary least squares

    Returns
    -------
    score : float. r-squared score for model fit
    coefs : (num_regressors x 1) parameter vector
    
    '''
    if verbose:
        print('='*15)
        print('Starting {lasso_or_ridge} Regression..')
    if lasso_or_ridge == 'lasso':
        model = sklearn.linear_model.Lasso(alpha=regularization_strength,tol=tol)
    elif lasso_or_ridge == 'ridge':
        model = sklearn.linear_model.Ridge(alpha=regularization_strength,tol=tol)
    model.fit(behavior, spikes) #no need to add constant since intercept is automatically fit by this method
    score = model.score(behavior, spikes) #r squared
    # a=model.get_params()
    # print(a.keys())
    # xxx
    
    if verbose:
        print('{lasso_or_ridge} model fit!\n')
        print(f'Num iterations: {model.n_iter_}')
        print(f'R^2 score: {score}')
        print(f'Num features seen: {model.n_features_in_}')
        if model.feature_names_in_ is not None:
            print(f'Feature names: {model.feature_names_in_}')
            print('Fit coefficients: ')
            [print(f'{model.feature_names_in_[i]} : {model.coef_[i]}') for i in range(model.n_features_in_)]
        else:
            print('Feature names not found')
            print('Fit coefficients: ')
            [print(f'Feature {i+1} : {model.coef_[i]}') for i in range(model.n_features_in_)]
        print('\n\n')
    return score, model.coef_

def regularization_regression_feature_selection(regressor_list,spikes,behavior_df,lasso_or_ridge,regularization_strength,tol):
    
    score, coefs = regularization_regression(spikes,behavior_df[regressor_list],lasso_or_ridge,regularization_strength,tol,verbose=False)

    thresh = 1
    feats = abs(coefs) > thresh #bool arr i think
    
    if sum(feats) > 0:
        signif_regressors = str([str(feat) for feat in np.array(regressor_list)[feats]])
    else:
        signif_regressors = 'none'
        
    return signif_regressors, coefs

def simple_regression_feature_selection(regressor_list,spikes,behavior_df,alpha_thresh):
    encoding_list = []
    for reg in regressor_list:
        reg_behav = behavior_df[reg]
        f_pval, pval, rsqr = linear_encoding_regression(spikes,reg_behav)
        if f_pval < alpha_thresh:
            encoding_list.append(reg)
            
    if len(encoding_list)<1:
        encoding_list.append('') #in case list is empty
        
    return encoding_list

def simple_regression_plot(regressor,spikes,behavior_df):
    reg_behav = behavior_df[regressor]
    res = linear_encoding_regression(spikes,reg_behav)
    
    fig,ax=plt.subplots()
    ax.plot(reg_behav.values,spikes,'.')
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
        res = linear_encoding_regression(spikes,reg_behav)
        rsqr_list.append(res.rsquared)
        
    return max(rsqr_list)

def find_best_learning_rate(fr_vector,Behavior,REGRESSORS,trials,learning_rates,alpha_thresh,unit_subsession_labeled,neuron_type):
    ## Get regression weights for all learning rate combos
    #loop thru all possible learning rates
    df_rows_regression_weights = []
    for pos_alpha in learning_rates:
        for neg_alpha in learning_rates:
        
            # get beahvior according to current learning rates
            Behavior.get_behavior(Q_learning=True, learning_rates=[pos_alpha,neg_alpha])
            regressor_matrix = Behavior.behavior_df[REGRESSORS].iloc[trials]
        
            # regress fr against this learning rate behavior
            res = linear_encoding_regression(fr_vector, regressor_matrix)
        
            row_dict_regression_weights = {'pos_alpha':pos_alpha, 'neg_alpha':neg_alpha} #each row of the df will be the lrs for each reg
            # Get weights for each regressor
            regressor_names = res.params.keys() #includes const in idx 0
            
            #loop thru all regressors
            for j in range(len(regressor_names)):
                if res.pvalues[regressor_names[j]] < alpha_thresh and res.f_pvalue < alpha_thresh: #if signif
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

    return row_dict_learning_rates


def get_rsqrs_from_LRs(stable_or_volatile, fr_df, Behavior, neuron_type_df, learning_rate_df, do_even_num_trials, REGRESSORS):        

    assert len(Behavior.behavior_df) == len(fr_df), 'Number of trials do not match!'
    # num_trials = len(fr_df)
    
    if do_even_num_trials:
        trials = get_trials_even_num(Behavior.behavior_df,stable_or_volatile)
    else:
        trials = get_trials(Behavior.behavior_df,stable_or_volatile)

    if not len(trials) > 1: #if no trials for current subsession
        print(f'No trials for {stable_or_volatile}. Skipping!')    
        return

    # Get FR for each neuron
    units = fr_df.columns
    
    rsqrs = []

    for i,unit in enumerate(units):

        neuron_type = str(get_unit_data(neuron_type_df, f'{unit} - all trials', 'Neuron Type'))
        if "Q" not in neuron_type:
            continue
        
        #dual lrs
        if f'Pos Learning Rate for {neuron_type}' in learning_rate_df.columns:
            pos_learning_rate = float(get_unit_data(learning_rate_df, f'{unit} - all trials', f'Pos Learning Rate for {neuron_type}'))
            neg_learning_rate = float(get_unit_data(learning_rate_df, f'{unit} - all trials', f'Neg Learning Rate for {neuron_type}'))
            learning_rate = [pos_learning_rate,neg_learning_rate]
        #single lr
        else:
            learning_rate = [float(get_unit_data(learning_rate_df, f'{unit} - all trials', f'Learning Rate for {neuron_type}'))]
            
        if np.any(np.isnan(learning_rate)): #skip units that don't have a fitted LR
            continue
           
        # get beahvior according to  learning rates
        Behavior.get_behavior(Q_learning=True, learning_rates=learning_rate)
        regressor_matrix = Behavior.behavior_df[REGRESSORS].iloc[trials]
        fr = fr_df[unit].iloc[trials] #vector of FRs for subsession trial
        # regress fr against this learning rate behavior
        res = linear_encoding_regression(fr, regressor_matrix)

        rsqrs.append(res.rsquared)
        
    return rsqrs

 

#%% data IO and transforms
def dict_to_arr(d: dict):
    '''
    for when theres a dictionary of arrays of the same size
    '''
    n1 = len(d) # e.g. units
    keys = list(d.keys())
    n2,n3 = np.shape(d[keys[0]]) # e.g. trials x timepoints
    arr = np.zeros((n1,n2,n3))
    for i,ar in enumerate(d.values()):
        assert ar.shape == (n2,n3)
        arr[i,:,:] = ar
    
    return arr # e.g. units x trials x timepoints

def merge_sessions_df(dfs: list):
    '''
    dfs : list of DataFrames
    '''
    return pd.concat(dfs,axis='index',ignore_index=True)

def save_svg(fig,fig_name,folder,proj_path):
    path_name = os.path.join(proj_path,'Figures',folder)
    if not os.path.isdir(path_name):
        os.mkdir(path_name)
    fig.savefig(os.path.join(path_name,f'{fig_name}.svg'),format='svg')
    
    print(f'Figures/{folder}/{fig_name}.svg')
    return

def save_csv(df,df_name,session,data_folder):
    df.to_csv(os.path.join(data_folder,session,f'{session}_{df_name}_df.csv'))
    print(f'{session}_{df_name}_df.csv saved!')
    return

def load_csv(df_name,session,data_folder):
    f = os.path.join(data_folder,session,f'{session}_{df_name}_df.csv')
    df = pd.read_csv(f)
    print(f'{session}_{df_name}_df.csv loaded.')
    return df

def save_pkl(obj,obj_name,session,data_folder):
    with open(os.path.join(data_folder,session,f'{session}_{obj_name}.pkl'),'wb') as f:
        pickle.dump(obj,f)
    print(f'{session}_{obj_name}.pkl saved!')
    return

def load_pkl(obj_name,session,data_folder):
    with open(os.path.join(data_folder,session,f'{session}_{obj_name}.pkl'),'rb') as f:
        obj = pickle.load(f)
    print(f'{session}_{obj_name}.pkl loaded.')
    return obj

def load_pkl_2(obj_name,session,data_folder):
    with open(os.path.join(data_folder,session,f'{obj_name}_{session}.pkl'),'rb') as f:
        obj = pickle.load(f)
    print(f'{obj_name}_{session}.pkl loaded.')
    return obj

def does_pkl_exist(obj_name,session,data_folder):
    return os.path.exists(os.path.join(data_folder,session,f'{session}_{obj_name}.pkl'))
    
def does_csv_exist(df_name,session,data_folder):
    return os.path.exists(os.path.join(data_folder,session,f'{session}_{df_name}_df.csv'))
