# -*- coding: utf-8 -*-
"""
Parzen window loglikelihood estimate,
Breuleux, O., Bengio, Y., and Vincent, P. (2011). Quickly generating representative samples from an
RBM-derived process. Neural Computation, 23(8), 2053–2073.


Original code author: Ian Goodfellow
https://github.com/goodfeli/adversarial/blob/master/parzen_ll.py
Modified by Boris van Breugel (bv292@cam.ac.uk)

"""

import numpy as np
import time



def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + np.log(np.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def parzen(data, mu, sigma):
    """
    Credit: Yann N. Dauphin
    """
    x = data

    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma

    E = log_mean_exp(-0.5*(a**2).sum(2))

    Z = mu.shape[1] * np.log(sigma * np.sqrt(np.pi * 2))

    return E - Z



def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)

        if i % 10 == 0:
            print(i, np.mean(times), np.mean(nlls))

    return np.array(nlls)


def cross_validate_sigma(samples, data, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        print(sigma)
        parzen_stats = parzen(samples, sigma)
        tmp = get_nll(data, parzen_stats, batch_size = batch_size)
        lls.append(np.asarray(tmp).mean())
        del parzen_stats
        
    ind = np.argmax(lls)
    return sigmas[ind]


def compute_parzen(orig_data, synth_data, start_sigma=-1, end_sigma=2, num_cv_evals=21, batch_size = 10):
    # Preprocess the data
    orig_data = np.asarray(orig_data)
    synth_data = np.asarray(synth_data)
        
    no, x_dim = np.shape(orig_data)
        
    # Divide train / test
    orig_data_valid = orig_data[:int(no/5),:]
    orig_data_test = orig_data[int(no/5):,:]
        
    synth_data_valid = synth_data[:int(no/5),:]
    synth_data_test = synth_data[int(no/5):,:]
    
    sigma_range = np.logspace(start_sigma, end_sigma, num=num_cv_evals)
    sigma = cross_validate_sigma(synth_data_valid, orig_data_valid, orig_data_valid, sigma_range, batch_size)
    # fit and evaulate
    parzen_est = parzen(synth_data_test, sigma)
    ll = get_nll(orig_data_test, parzen_est, batch_size = batch_size)
    se = ll.std() / np.sqrt(synth_data_test.shape[0])

    print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))
    