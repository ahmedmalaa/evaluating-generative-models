    
# Copyright (c) 2021, Ahmed M. Alaa, Boris van Breugel
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
  
  ----------------------------------------- 
  Metrics implementation
  ----------------------------------------- 

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors

import logging
import torch
import scipy

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
device = 'cpu' # matrices are too big for gpu


def compute_alpha_precision_old(real_data, synthetic_data, emb_center):
    n_steps = 30
    nn_size = 2
    alphas  = np.linspace(0, 1, 30)
    Radii   = [np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas[k]) for k in range(len(alphas))]
    
    synth_center          = torch.tensor(np.mean(synthetic_data, axis=0)).float()
    synth_Radii           = [np.quantile(torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - synth_center) ** 2, dim=1)), alphas[k]) for k in range(len(alphas))]

    alpha_precision_curve = []
    beta_coverage_curve   = []
    
    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))
    synth_to_synth_center = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - synth_center) ** 2, dim=1))
    real_to_center        = torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1))
    
    real_to_synth         = [np.min(np.sum(np.abs(real_data[k, :] - synthetic_data), axis=1)) for k in range(real_data.shape[0])]
    real_to_synth_args    = [np.argmin(np.sum(np.abs(real_data[k, :] - synthetic_data), axis=1)) for k in range(real_data.shape[0])]
    real_to_synth         = torch.tensor(np.array(real_to_synth)).float()
    real_synth_closest    = np.array([synthetic_data[real_to_synth_args[k], :] for k in range(len(real_to_synth_args))])
    
    closest_synth_Radii   = [np.quantile(torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float() - synth_center) ** 2, dim=1)), alphas[k]) for k in range(len(alphas))]
    real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float()- synth_center) ** 2, dim=1))

    real_to_real          = [np.partition(np.sum(np.abs(real_data[k, :] - real_data), axis=1), nn_size)[nn_size-1] for k in range(real_data.shape[0])]
    real_to_real          = torch.tensor(np.array(real_to_real)).float()
    
    real_to_synth_all  = [np.min(np.sum(np.abs(real_data[k, :] - synthetic_data), axis=1)) for k in range(real_data.shape[0])]
    real_to_real_all   = np.array([np.sum(np.abs(real_data[k, :] - real_data), axis=1) for k in range(real_data.shape[0])])
    dist_probs         = [1/np.mean(real_to_synth_all[k] <= real_to_real_all[k, :]) for k in range(real_data.shape[0])]
   
    for k in range(len(Radii)):
        
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)

        beta_coverage        = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).detach().float().numpy())
 
        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)
    
    
    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    Delta_coverage_beta  = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) * (alphas[1] - alphas[0])
    
    dist_ps    = np.array(dist_probs)
    dist_min   = np.min(dist_ps)
    dist_max   = np.max(dist_ps)

    thresholds = np.linspace(dist_min, dist_max, 1000) 
    authen     = np.array([np.mean(dist_ps >= thresholds[k]) for k in range(len(thresholds))])
    
    return alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, (thresholds, authen)