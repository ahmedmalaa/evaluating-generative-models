
# Copyright (c) 2021, Ahmed M. Alaa, Boris van Breugel
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
  
  ----------------------------------------- 
  Metrics implementation
  ----------------------------------------- 

"""

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


def compute_alpha_precision(real_data, synthetic_data, emb_center, alternative_coverage=False):
    

    emb_center = torch.tensor(emb_center, device=device)

    n_steps = 30
    alphas  = np.linspace(0, 1, n_steps)
    k_coverage = 1
    
    Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)
    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))
    
    
    nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _       = nbrs_real.kneighbors(real_data)
    real_to_real          = torch.from_numpy(real_to_real[:,1].squeeze())
   
    synth_center          = torch.tensor(np.mean(synthetic_data, axis=0)).float()
        

    if not alternative_coverage:
        nbrs_synth = NearestNeighbors(n_neighbors = k_coverage, n_jobs=-1, p=2).fit(synthetic_data)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data)
        real_to_synth         = torch.from_numpy(real_to_synth[:,k_coverage-1])
        real_to_synth_args    = real_to_synth_args[:,k_coverage-1]
        real_synth_closest    = synthetic_data[real_to_synth_args]
        real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float()- synth_center) ** 2, dim=1))
        closest_synth_Radii   = np.quantile(real_synth_closest_d, alphas)
    else:
        synth_Radii          = np.quantile(torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - synth_center) ** 2, dim=1)), alphas)
        real_to_center       = torch.sqrt(torch.sum((torch.tensor(real_data).float() - synth_center) ** 2, dim=1))
    
    
    
    alpha_precision_curve = []
    beta_coverage_curve   = []
    
    
    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)

        if alternative_coverage:
            beta_coverage      = np.mean((real_to_center <= synth_Radii[k]).detach().float().numpy())
        else:
            beta_coverage        = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).detach().float().numpy())
 
        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)
    

    # See which one is bigger
    
    synth_to_real, synth_to_real_args = nbrs_real.kneighbors(synthetic_data)    
    synth_to_real         = synth_to_real[:,0].squeeze()
    synth_to_real_args    = synth_to_real_args[:,0].squeeze()
    authen = real_to_real.numpy()[synth_to_real_args] < synth_to_real
    authenticity = np.mean(authen)

    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alpha_precision_curve)-alphas)) * (alphas[1] - alphas[0])
    Delta_coverage_beta   = 1 - 2 * np.sum(np.abs(np.array(beta_coverage_curve)-alphas)) * (alphas[1] - alphas[0])
    
    return alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authenticity
