# -*- coding: utf-8 -*-
"""

From 
https://github.com/msmsajjadi/precision-recall-distributions/blob/master/prd_from_image_folders.py
"""

# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)

import metrics.prd_score as prd


def compute_prc(orig_data,synth_data, params=None, plot_path=None, verbose=False):
    if params is None:
        params = {}
        params['num_clusters'] = 20
        params['num_angles'] = 1001
        params['num_runs'] = 10
    real_embeddings = orig_data
    eval_embeddings = synth_data
    if verbose:
        print('computing PRD')
    prd_data = prd.compute_prd_from_embedding(
            eval_data=eval_embeddings,
            ref_data=real_embeddings,
            num_clusters=params['num_clusters'],
            num_angles=params['num_angles'],
            num_runs=params['num_runs'])
    
    precision, recall = prd_data
    
    if verbose:
        print('plotting results')

    f_beta = prd.prd_to_max_f_beta_pair(precision, recall, beta=8)
    print('%.3f %.3f' % (f_beta[0], f_beta[1]))

    return prd_data
    