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
import pickle
import sklearn
# import neurodsp
# import matplotlib
import numpy as np
# import scipy as sp
import pandas as pd
# import matplotlib_venn
import statsmodels.api as sm
# from matplotlib.lines import Line2D
from matplotlib import pyplot as plt
# from statsmodels.stats.outliers_influence import variance_inflation_factor

from Whitehall import DATA_FOLDER,PROJ_FOLDER,ALPHA_THRESHOLD

#### Utility Funcs
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
    model = sklearn.linear_model.Lasso(alpha=regularization_strength,tol=tol)
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
        f_pval, pval, rsqr = encoding_regression(spikes,reg_behav)
        rsqr_list.append(rsqr)
        
    return max(rsqr_list)
    
def area_parser(df_or_dict,brain_area):
    
    if type(df_or_dict) == dict:
        
        match brain_area:
            case 'vmPFC':
                subset_df_or_dict = {k:v for k,v in df_or_dict.items() if 'Unit A' not in k}
            case 'Cd':
                subset_df_or_dict = {k:v for k,v in df_or_dict.items() if 'Unit C' not in k}
            case 'OFC':
                subset_df_or_dict = {k:v for k,v in df_or_dict.items() if 'Unit D' not in k}
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

def get_trials(behav_df,stable_or_volatile):
    
    all_trials = np.arange(len(behav_df))
    
    match stable_or_volatile:
        case 'stable':
            trials = all_trials[behav_df['Stable']==1]
        case 'volatile':
            trials = all_trials[behav_df['Volatile']==1]
        case _:
            trials = trials = np.arange(len(behav_df))
            
    return trials

def calc_percent_encoding(pval_df,alpha_threshold):
    pval_df = pval_df.drop('Unit_labels',axis=1)
    df_signif = pval_df < alpha_threshold
    num_signif = df_signif.sum(axis='index')
    num_units = pval_df.shape[0]
    percent_encoding = num_signif / num_units
    return percent_encoding, num_units

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

def save_out_svg(fig_name,folder):
    path_name = os.path.join(PROJ_FOLDER,'Figures',folder)
    if not os.path.isdir(path_name):
        os.mkdir(path_name)
    plt.savefig(os.path.join(path_name,f'{fig_name}.svg'),format='svg')
    
    print(f'Figures/{folder}/{fig_name}.svg')
    return

def save_csv(df,df_name,session):
    df.to_csv(os.path.join(DATA_FOLDER,session,f'{session}_{df_name}_df.csv'))
    print(f'{session}_{df_name}_df.csv saved!')
    return

def load_csv(df_name,session):
    f = os.path.join(DATA_FOLDER,session,f'{session}_{df_name}_df.csv')
    df = pd.read_csv(f)
    print(f'{session}_{df_name}_df.csv loaded.')
    return df

def save_pkl(obj,obj_name,session):
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
