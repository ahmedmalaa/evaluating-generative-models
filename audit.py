

"""
Author: Boris van Breugel (bv292@cam.ac.uk)
  ----------------------------------------- 
  Auditing implementation
  ----------------------------------------- 

"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

import logging
import torch
import scipy
from generative_models.adsgan import adsgan
from metrics.evaluation import compute_alpha_precision
from metrics.evaluation_old import compute_alpha_precision_old

device = 'cuda' # matrices are too big for gpu


def audit(real_data, params, OC_model):

    
    n_steps = 30
    n_orig = real_data.shape[0]
    nn_size = 2
    alphas  = np.linspace(0, 1, n_steps)
        
    emb_center = torch.tensor(OC_model.c, device='cpu')

    with torch.no_grad():
        X = OC_model(torch.tensor(real_data.to_numpy(), device=OC_model.device).float().to(device)).cpu().detach().numpy()

    Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(X).float() - emb_center) ** 2, dim=1)), alphas)
    alpha_precision_curve = []
    beta_coverage_curve   = []
    nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(X)
    real_to_real, real_to_real_args       = nbrs_real.kneighbors(X)
    real_to_real          = torch.from_numpy(real_to_real[:,1].squeeze())
    
    print('Difference a;lf', (real_to_real_args[:,0]==np.arange(n_orig)).mean())
    real_to_real_args          = real_to_real_args[:,1].squeeze()


    number_per_quantile = np.round(np.quantile(np.arange(n_orig),alphas))
    number_per_quantile = number_per_quantile[1:] - number_per_quantile[:-1] 
    
    r2r = scipy.spatial.distance_matrix(X,X)
    r2r[np.eye(n_orig, dtype='bool')] = np.max(r2r)+1 #just set it large so it's not chosen
    min_r2r = np.min(r2r,axis=1)
    min_r2r_args = np.argmin(r2r,axis=1)    
    print('min_r2r', (min_r2r==0).mean())
    
    print('Difference abs', np.max(np.abs(min_r2r-real_to_real.numpy())))
    print('Difference arguments')
    
    
    synthetic_data = []
    
    generate_more = True
    iteration = 0

    while generate_more:
        print('Iteration:',iteration)
        iteration+=1
        synth_data = adsgan(real_data, params)
        with torch.no_grad():
            Y = OC_model(torch.tensor(synth_data, device=OC_model.device).float().to(device)).cpu().detach().numpy()
        
        
    
        nbrs_synth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(Y)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)
        real_to_synth         = torch.from_numpy(real_to_synth.squeeze())
        real_to_synth_args    = real_to_synth_args.squeeze()
        print('Mean real to synth' , torch.mean(real_to_synth))
        print('mean real to real', torch.mean(real_to_real[real_to_synth_args]))
        # Audit
        #authen = np.ones(len(real_to_synth),dtype='bool')#
        authen = real_to_real[real_to_synth_args] < real_to_synth
        indices_to_use_authen = np.arange(len(authen), dtype = 'int')[authen]
        synth_data = synth_data[indices_to_use_authen]
        print('After auditing out unauthentic points, points remain:',synth_data.shape[0])

        Y = Y[indices_to_use_authen]

        nbrs_synth            = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(Y)
        
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)
        
        real_to_synth         = torch.from_numpy(real_to_synth.squeeze())
        real_to_synth_args    = real_to_synth_args.squeeze()
        
        print('After which the authenticity is', np.mean(np.array(real_to_real[real_to_synth_args] < real_to_synth,dtype='bool')))
        


        # Precisions
        synth_center          = torch.tensor(np.mean(Y, axis=0)).float()
        synth_to_center       = torch.sqrt(torch.sum((torch.tensor(Y).float() - emb_center) ** 2, dim=1))

        real_synth_closest    = Y[real_to_synth_args]
        real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float()- synth_center) ** 2, dim=1))
        closest_synth_Radii   = np.quantile(real_synth_closest_d, alphas)

        n_synth = Y.shape[0]
        indices_available = np.ones(n_synth)
        indices_use = np.zeros(n_synth, dtype = 'bool')
        
        
        generate_more = False

        for k in range(n_steps-1):
            if number_per_quantile[k] != 0:
                
                precision_mask = (synth_to_center <= Radii[k+1]).detach().float().numpy()
                indices_close_enough = np.arange(n_synth,dtype='int')[np.logical_and(precision_mask, indices_available)]
                indices_available = np.logical_not(precision_mask)
                number_to_add = int(min(number_per_quantile[k], len(indices_close_enough)))
                indices_close_enough = indices_close_enough[:number_to_add]
                indices_use[indices_close_enough] = True
                number_per_quantile[k] -= number_to_add
                if number_per_quantile[k] != 0: 
                    generate_more = True
        
        
        synthetic_data.append(synth_data[indices_use])
    
    synthetic_data = np.concatenate(synthetic_data,axis=0)
    with torch.no_grad():
        Y = OC_model(torch.tensor(synthetic_data, device=OC_model.device).float().to(device)).cpu().detach().numpy()
    
    print('new results', compute_alpha_precision(X,Y, emb_center)[3:])
    print('old_results', compute_alpha_precision_old(X,Y, emb_center)[3:-1])

    return synthetic_data