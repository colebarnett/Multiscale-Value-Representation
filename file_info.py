# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:14:42 2026

@author: coleb
"""

'''
this info is gotten from "NHP log.xlsx" on Box
'''


import numpy as np

def get_block_info(session):
    
    #stable first = 6bl*100tr + 48bl*20tr
    #vol first = 24bl*20tr + 12bl*100tr
    #goal is to get ~1000tr per session, 480 volatile and 600 stable
    max_num_trials = 1600 #sufficiently long, monkeys usually only go 1000

    #initiate arrays
    is_stable_block = np.zeros(max_num_trials)
    is_volatile_block = np.zeros(max_num_trials)
    
    def stable_1st():
        is_stable_block[:600] = 1
        is_volatile_block[600:] = 1
        return
    
    def volatile_1st():
        is_stable_block[480:] = 1
        is_volatile_block[:480] = 1   
        return
    
    match session:
        case 'airp20250602_03_te2003':
            stable_1st()
        case 'airp20250604_08_te2011':
            volatile_1st()
        case 'airp20250617_03_te2031':
            stable_1st()
        case 'airp20250618_03_te2034':
            volatile_1st()
        case 'airp20250626_03_te2049':
            stable_1st()
        case 'airp20250701_06_te2058':
            stable_1st()
        case 'airp20250826_05_te2135':
            stable_1st()
        case 'airp20250904_03_te2147':
            volatile_1st()
        case 'airp20250910_02_te2156':
            volatile_1st()
        case 'airp20250912_02_te2160':
            stable_1st()
        case 'airp20250919_02_te2177':
            volatile_1st()    
        case 'airp20251007_02_te2198':
            stable_1st()
        case 'airp20251008_02_te2200':
            volatile_1st()    
        case 'airp20251015_04_te2206':
            stable_1st()
        case 'airp20251016_03_te2209':
            volatile_1st()
        case 'airp20251020_05_te2214':
            stable_1st()
        case 'airp20251021_02_te2216':
            volatile_1st()
        case 'airp20251023_03_te2219':
            stable_1st()
        case 'airp20251028_03_te2226':
            stable_1st()
        case 'airp20251029_05_te2231':
            volatile_1st()
        case 'airp20251030_02_te2233':
            stable_1st()
        case 'airp20251104_02_te2242':
            stable_1st()
        case 'airp20251111_02_te2250':
            volatile_1st()
        case 'airp20260224_03_te2287':
            stable_1st()
        case 'airp20260302_03_te2302':
            volatile_1st()
        case 'airp20260303_02_te2304':
            stable_1st()
        case 'airp20260305_12_te2321':
            stable_1st()
        case 'airp20260310_03_te2324':
            volatile_1st()
        case 'airp20260311_03_te2327':
            stable_1st()
        case 'airp20260403_03_te2337':
            volatile_1st()
        case 'airp20260406_03_te2340':
            stable_1st()
        case 'airp20260408_03_te2343':
            volatile_1st()
        case 'airp20260409_03_te2346':
            stable_1st()
        case 'airp20260414_03_te2352':
            stable_1st()
        case 'airp20260415_03_te2355':
            volatile_1st()
        case 'airp20260624_06_te2448':
            stable_1st()
        case 'airp20260625_05_te2453':
            volatile_1st()
        case 'airp20260630_04_te2468':
            volatile_1st()
        case 'airp20260708_05_te2496':
            volatile_1st()
            
            
        case 'braz20240927_01_te5384':
            stable_1st()
            
        case 'braz20260728_03_te2536':
            stable_1st()
        case 'braz20260729_03_te2539':
            stable_1st()
        case 'braz20260730_03_te2542':
            volatile_1st()
        case 'braz20260731_03_te2545':
            stable_1st()
        case 'braz20260803_03_te2548':
            volatile_1st()
        case 'braz20260804_03_te2551':
            volatile_1st()
        case 'braz20260805_03_te2554':
            stable_1st()
        case 'braz20260806_03_te2557':
            stable_1st()
        case 'braz20260810_03_te2560':
            volatile_1st()
        case 'braz20260811_03_te2563':
            volatile_1st()
        case 'braz20260812_03_te2566':
            stable_1st()
        case 'braz20260813_03_te2569':
            stable_1st()
        case 'braz20260814_03_te2572':
            stable_1st()
        case 'braz20260817_03_te2575':
            stable_1st()
        case 'braz20260818_03_te2578':
            volatile_1st()
        case 'braz20260821_03_te2581':
            volatile_1st()
        case 'braz20260824_03_te2584':
            volatile_1st()
        case 'braz20260827_03_te2587':
            volatile_1st()
        case 'braz20260828_03_te2590':
            stable_1st()
        case 'braz20260901_03_te2593':
            stable_1st()
        case 'braz20260902_03_te2596':
            volatile_1st()
        case 'braz20260908_03_te2599':
            stable_1st()
        case 'braz20260909_03_te2602':
            volatile_1st()
        case 'braz20260910_03_te2605':
            stable_1st()
        case 'braz20260911_03_te2608':
            stable_1st()
            
        case 'braz20260915_05_te2613':
            volatile_1st()
        case 'braz20260916_03_te2616':
            volatile_1st()
        case 'braz20260918_03_te2619':
            stable_1st()
        case 'braz20260924_03_te2622':
            stable_1st()
        case 'braz20260925_03_te2625':
            volatile_1st()
        case 'braz20261001_03_te2628':
            stable_1st()
        case 'braz20261002_05_te2633':
            volatile_1st()
        case _:
            raise ValueError(f"No info about stable/volatile 1st found for {session} !")
            
            
    return is_stable_block, is_volatile_block


def get_area_info(session):
    match session:

        case 'airp20250919_02_te2177':
            session_areas = ['vmPFC','Cd','OFC']
        case 'airp20251015_04_te2206':
            session_areas = ['vmPFC','Cd','OFC']
        case 'airp20251016_03_te2209':
            session_areas = ['vmPFC','Cd','OFC']
        case 'airp20251020_05_te2214':
            session_areas = ['vmPFC','Cd','OFC']
        case 'airp20251021_02_te2216':
            session_areas = ['vmPFC','Cd']
        case 'airp20251023_03_te2219':
            session_areas = ['vmPFC','Cd']
        case 'airp20251028_03_te2226':
            session_areas = ['Cd','OFC']
        case 'airp20251029_05_te2231':
            session_areas = ['vmPFC','Cd','OFC']
        case 'airp20251030_02_te2233':
            session_areas = ['vmPFC','Cd','OFC']
        case 'airp20251104_02_te2242':
            session_areas = ['Cd']
        case 'airp20251111_02_te2250':
            session_areas = ['Cd']
        case 'airp20260624_06_te2448':
            session_areas = ['OFC']
        case 'airp20260625_05_te2453':
            session_areas = ['Cd','OFC']
        case 'airp20260630_04_te2468':
            session_areas = ['vmPFC']
        case 'airp20260708_05_te2496':
            session_areas = ['vmPFC','OFC']
            
            
            
            
        case 'braz20260728_03_te2536':
            session_areas = ['OFC']
        case 'braz20260729_03_te2539':
            session_areas = ['Cd','OFC']
        case 'braz20260730_03_te2542':
            session_areas = ['OFC']
        case 'braz20260731_03_te2545':
            session_areas = ['OFC']
        case 'braz20260803_03_te2548':
            session_areas = ['Cd','OFC']
        case 'braz20260804_03_te2551':
            session_areas = ['Cd','OFC']
        case 'braz20260805_03_te2554':
            session_areas = ['Cd','OFC']
        case 'braz20260806_03_te2557':
            session_areas = ['Cd','OFC']
        case 'braz20260810_03_te2560':
            session_areas = ['Cd','OFC']
        case 'braz20260811_03_te2563':
            session_areas = ['Cd','OFC']
        case 'braz20260812_03_te2566':
            session_areas = ['Cd','OFC']
        case 'braz20260813_03_te2569':
            session_areas = ['Cd','OFC']
        case 'braz20260814_03_te2572':
            session_areas = ['Cd','OFC']
        case 'braz20260817_03_te2575':
            session_areas = ['Cd','OFC']
        case 'braz20260818_03_te2578':
            session_areas = ['Cd','OFC']
        case 'braz20260821_03_te2581':
            session_areas = ['Cd','OFC']
        case 'braz20260824_03_te2584':
            session_areas = ['Cd','OFC']
        case _:
            raise ValueError(f"No info about chs for {session} !")
            
    return session_areas