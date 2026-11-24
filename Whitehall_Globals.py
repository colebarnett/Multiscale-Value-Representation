# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:58:02 2026

@author: coleb
"""
import os
# import sys
import numpy as np
#### Define Global Constants

OVERWRITE_SPIKES=False
OVERWRITE_LFP=False
LOAD_BEHAV=True
LOAD_SPIKES=False
LOAD_LFP=False
VERBOSE=False

SUBSESSIONS = ['all trials','stable block','volatile block']

SESSIONS = [
            "braz20260728_03_te2536",
            "braz20260729_03_te2539",
            "braz20260730_03_te2542",
            "braz20260731_03_te2545",
            "braz20260803_03_te2548",
            "braz20260804_03_te2551",
            "braz20260805_03_te2554",
            "braz20260806_03_te2557",
            "braz20260810_03_te2560",
            # "braz20260811_03_te2563", #too few trials (1044), excluded
            # "braz20260812_03_te2566", #too few trials (939), excluded
            "braz20260813_03_te2569", 
            "braz20260814_03_te2572", 
            "braz20260817_03_te2575", 
            "braz20260818_03_te2578", 
            "braz20260821_03_te2581", 
            "braz20260824_03_te2584", 
            "braz20260827_03_te2587",
            # "braz20260828_03_te2590", #too few trials (1039), excluded
            "braz20260901_03_te2593",
            "braz20260902_03_te2596",
            "braz20260908_03_te2599",
            "braz20260909_03_te2602",
            "braz20260910_03_te2605",
            "braz20260911_03_te2608",
            "braz20260915_05_te2613",
            "braz20260916_03_te2616",
            "braz20260918_03_te2619",
            "braz20260924_03_te2622",
            "braz20260925_03_te2625",
            "braz20261001_03_te2628",
            # # # "braz20261002_05_te2633", #too few trials (747), excluded
            "braz20261007_03_te2636",
            # # "braz20261008_03_te2639", #too few trials (845), excluded
            # # "braz20261009_03_te2642", #too few trials (1041), excluded
            ]

QCHOSEN_REGRESSORS = [
                    # 'Q1',
                    # 'Q2',
                    # 'Qleft',
                    # 'Qright',
                    # 'Qhigh',
                    # 'Qlow',
                    'Qchosen',
                    # 'Qdiff_',
                    # 'absQdiff',
                    'ChoiceTarg',
                    'ChoiceExploit ⊥ Qch',
                    'ChoiceSide',
                    'Time',
                    # 'Time x Time',
                    # 'Log_Time',
                    # 'State ⊥ Time',
                    # 'RPE',
                    # 'absRPE',
                    # 'Prev_Reward',
                    'Prev_Rew ⊥ Qch', #residualized to Qchosen since they are moderately correlate
                    # 'Qch ⊥ Prev_Rew',     
                    ]

Q1_REGRESSORS = [
                    'Q1',
                    # 'Q2',
                    # 'Qleft',
                    # 'Qright',
                    # 'Qhigh',
                    # 'Qlow',
                    # 'Qchosen',
                    # 'Qdiff_',
                    # 'absQdiff',
                    'ChoiceTarg',
                    'ChoiceExploit ⊥ Q1',
                    'ChoiceSide',
                    'Time',
                    # 'Time x Time',
                    # 'Log_Time',
                    # 'State ⊥ Time',
                    # 'RPE',
                    # 'absRPE',
                    # 'Prev_Reward',
                    'Prev_Rew ⊥ Q1', #residualized for consistency
                    # 'Qch ⊥ Prev_Rew',     
                    ]

Q2_REGRESSORS = [
                    # 'Q1',
                    'Q2',
                    # 'Qleft',
                    # 'Qright',
                    # 'Qhigh',
                    # 'Qlow',
                    # 'Qchosen',
                    # 'Qdiff_',
                    # 'absQdiff',
                    'ChoiceTarg',
                    'ChoiceExploit ⊥ Q2',
                    'ChoiceSide',
                    'Time',
                    # 'Time x Time',
                    # 'Log_Time',
                    # 'State ⊥ Time',
                    # 'RPE',
                    # 'absRPE',
                    # 'Prev_Reward',
                    'Prev_Rew ⊥ Q2', #residualized for consistency
                    # 'Qch ⊥ Prev_Rew',     
                    ]


REGRESSOR_SUPERSET_DICT = {'Qchosen': QCHOSEN_REGRESSORS,
                           'Q1': Q1_REGRESSORS,
                           'Q2': Q2_REGRESSORS,}
ALL_REGRESSORS = list(set().union(*REGRESSOR_SUPERSET_DICT.values())) #all unique regs



REGRESSOR_COLORS = [
                    'lime',
                    'cyan',
                    'orange',
                    'aquamarine',
                    'red',
                    'gray',
                    ]

FREQ_BANDS = {'Theta':(3.5,8.5), # Define freq bands (Hz)
              'Alpha':(8.5,12.5),
              'Beta':(12.5,30.5),
              # 'Gamma':(30.5,200.5),
              'Low Gamma':(30.5,60.5),
              'High Gamma':(60.5,200.5)}

AREAS = ['Cd','OFC']

REW_PROBS = [0., 0.2, 0.4, 0.6, 0.8, 1.]

# VALUE_MODEL = 'DualBaseline'
# VALUE_MODEL = 'DualBaselineSide'
VALUE_MODEL = 'BaselineSide'

ALIGNMENT = 'Targets On' 
# Options are 'Targets On', 'Choice Made', and 'Reward' 
# These correspond to targets appearing, choice being made, and reward info disbursement, respectively

HOLD_DURATION = 0.6 # how long the actual center_hold time is [s]
ENCODING_WINDOW = (0.0,0.6) #window over which neuron encoding is determined [s]

T_BEFORE = 0 #how far to look before alignment point [s]
T_AFTER = 1.2 #how far to look after alignment point [s]
T_BOXCAR_SIZE = 0.1
T_BOXCAR_STEP = 0.05
T_VECTOR = np.round(np.arange(-T_BEFORE, T_AFTER-T_BOXCAR_STEP, T_BOXCAR_STEP),decimals=2)
N_TIMEBINS = len(T_VECTOR)

#NOW DEFINED IN LR ANALYSIS SCRIPT
# ENCODING_ALPHA = 0.005 #significance threshold for each timebin
# ENCODING_NCONSEC = 3 #num of consecutive significant time bins needed to be determined as encoding

# #From old psth way of doing things
# T_BEFORE = 0.1 #how far to look before alignment point [s]
# T_AFTER = 0.7 #how far to look after alignment point [s]
# T_RES = 0.05 #time resolution ; size of time bins [s]
# T_VECTOR = np.round(np.arange(-T_BEFORE+T_RES, T_AFTER, T_RES),decimals=2)
# N_TIMEBINS = len(T_VECTOR)

if 'Dual' in VALUE_MODEL:
    LEARNING_RATES = np.linspace(start=0.05,stop=0.95,num=19) #used for dual learning rates
else:
    LEARNING_RATES = np.linspace(start=0.02,stop=0.98,num=49) #used for single learning rate
    

ALPHA_THRESHOLD = 0.05

# USE_SLIDE = False
# FR_INCLUSION_NUM_TRIALS_SLIDE_WINDOW = 20 #num trials for sliding average for spike min FR inclusion criteria
MIN_FR_THRESH = 1 #Hz
MIN_NUM_TRIALS = 1080 #trials

COLORS = ['red','blue','gold','green','orange','purple']
LINESTYLES = ['-','--',':','-.',(0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10))]
MARKERSTYLES = ['o','^','s','*','d','P','X','p','H','<','>']
HATCHSTYLES = ['','/', '|', '-', '+', 'x', 'o', 'O', '.', '*']
# STABLE_COLOR, VOLATILE_COLOR = 'blue','orange'


#### Define Paths
##Yi's desktop
# BMI_FOLDER = r"C:\Users\crb4972\bmi_python"
# PROJ_FOLDER = r"C:\Users\crb4972\Desktop\whitehall\whitehall\Multiscale-Value-Representation"
# DATA_FOLDER = r"E:\Whitehall\data"

##Cole laptop
PROJ_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis"
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
DATA_FOLDER = os.path.join(PROJ_FOLDER, 'data')

NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns') #Neuroshare python
NSX_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'blackrock')


    
os.chdir(PROJ_FOLDER)