# -*- coding: utf-8 -*-
"""

Created on Fri Jan 15 15:52:59 2021

@author: boris

"""


from metrics.feature_distribution import feature_distribution
from metrics.compute_wd import compute_wd
from metrics.compute_identifiability import compute_identifiability
from metrics.fid import compute_frechet_distance
from metrics.parzen import compute_parzen
from metrics.precision_recall import compute_prc
from metrics.prdc import compute_prdc
from metrics.evaluation import compute_alpha_precision

import torch
import numpy as np

def compute_metrics(X, Y, which_metric=None, wd_params=None, model=None, OC4all=False):
    results = {}
    
    if model is not None:
        with torch.no_grad():
            X_out = model(torch.tensor(X).float()).detach().numpy()
            Y_out = model(torch.tensor(Y).float()).detach().numpy()
        if OC4all:
            X, Y = X_out, Y_out
    
    
    if which_metric is None:
        if X.shape[1]<100:
            which_metric = ['WD','ID','FD', 'PRDC']
        else:
            which_metric = ['WD','ID', 'FD', 'PRDC']
            
    if wd_params is None:
        wd_params = dict()
        wd_params['iterations'] = 10000
        wd_params['h_dim'] = 30
        wd_params['z_dim'] = 10
        wd_params['mb_size'] = 128
    
    # (1) Marginal distributions
    if 'marg' in which_metric:
        
        print('Start computing marginal feature distributions')
        results['feat_dist'] = feature_distribution(X, Y)
        print('Finish computing feature distributions')
        print(results['feat_dist'])


    # (2) Wasserstein Distance (WD)
    if 'WD' in which_metric:
        print('Start computing Wasserstein Distance')
        results['wd_measure'] = compute_wd(X, Y, wd_params)
        print('WD measure: ',results['wd_measure'])
    
    
    # (3) Identifiability 
    if 'ID' in which_metric:
        print('Start computing identifiability')
        results['identifiability'] = compute_identifiability(X, Y)
        print('Identifiability measure: ',results['identifiability'])
    

    # (4) Frechet distance
    if 'FD' in which_metric:
        results['fid_value'] = compute_frechet_distance(X, Y)
        print('Frechet distance', results['fid_value'])
        print('Frechet distance/dim', results['fid_value']/Y.shape[-1])
    

    # (5) Parzen
    if 'parzen' in which_metric:
        results['parzen_ll'], results['parzen_std'] = compute_parzen(X, Y, sigma=0.408)
        print(f'Parzen Log-Likelihood of test set = {results["parzen_ll"]}, se: {results["parzen_std"]}')

            
    # (6) Precision/Recall
    if 'PR' in which_metric:
        results['PR'] = compute_prc(X,Y)
    elif 'PRDC' in which_metric:
        prdc_res = compute_prdc(X,Y)
        for key in prdc_res:
            print('PRDC:', key, prdc_res[key])
            results[key] = prdc_res[key]
    
    # (7) OneClass
    if model is not None:
        
        alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, (thresholds, authen) = compute_alpha_precision(X_out, Y_out, model)
        results['alphas'] = alphas
        results['alpha_pc'] = alpha_precision_curve
        results['beta_cv'] = beta_coverage_curve
        results['thresholds'] = thresholds
        results['auten'] = authen
        results['Dpa'] = Delta_precision_alpha
        results['Dcb'] = Delta_coverage_beta
        results['Daut'] = np.mean(authen)
        print(np.min(thresholds), np.max(thresholds))
        print('OneClass: Delta_precision_alpha', results['Dpa'])
        print('OneClass: Delta_coverage_beta  ', results['Dcb'])
        print('OneClass: Delta_autenticity    ', results['Daut'])
        

    return results