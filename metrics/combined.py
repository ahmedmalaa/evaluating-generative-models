# -*- coding: utf-8 -*-
"""

Created on Fri Jan 15 15:52:59 2021

@author: boris

"""

import numpy as np

from metrics.feature_distribution import feature_distribution
from metrics.compute_wd import compute_wd
from metrics.compute_identifiability import compute_identifiability
from metrics.fid import compute_frechet_distance
from metrics.parzen import compute_parzen
from metrics.precision_recall import compute_prc

def compute_metrics(orig_data, synth_data, which_metric=None, wd_params=None):
    results = {}
    if which_metric is None:
        if orig_data.shape[1]<100:
            which_metric = ['WD','ID','FD','parzen', 'PR']
        else:
            which_metric = ['WD','FD']
            
    if wd_params is None:
        wd_params = dict()
        wd_params['iterations'] = 10000
        wd_params['h_dim'] = 30
        wd_params['z_dim'] = 10
        wd_params['mb_size'] = 128
    
    # (1) Marginal distributions
    if 'marg' in which_metric:
        
        print('Start computing marginal feature distributions')
        results['feat_dist'] = feature_distribution(orig_data, synth_data)
        print('Finish computing feature distributions')
        print(results['feat_dist'])


    # (2) Wasserstein Distance (WD)
    if 'WD' in which_metric:
        print('Start computing Wasserstein Distance')
        results['wd_measure'] = compute_wd(orig_data, synth_data, wd_params)
        print('WD measure: ',results['wd_measure'])
    
    
    # (3) Identifiability 
    if 'ID' in which_metric:
        print('Start computing identifiability')
        results['identifiability'] = compute_identifiability(orig_data, synth_data)
        print('Identifiability measure: ',results['identifiability'])
    

    # (4) Frechet distance
    if 'FD' in which_metric:
        results['fid_value'] = compute_frechet_distance(orig_data, synth_data)
        print('Frechet distance', results['fid_value'])
        print('Frechet distance/dim', results['fid_value']/synth_data.shape[-1])
    

    # (5) Parzen
    if 'parzen' in which_metric:
        results['parzen_ll'], results['parzen_std'] = compute_parzen(orig_data, synth_data)
        print(f'Parzen Log-Likelihood of test set = {results["parzen_ll"]}, se: {results["parzen_std"]}')

            
    # (6) Precision/Recall
    if 'PR' in which_metric:
        results['PR'] = compute_prc(orig_data,synth_data)
        
    
        
    # (7) Privacy

