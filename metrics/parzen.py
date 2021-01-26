# -*- coding: utf-8 -*-
"""
Parzen window loglikelihood estimate,
Breuleux, O., Bengio, Y., and Vincent, P. (2011). Quickly generating representative samples from an
RBM-derived process. Neural Computation, 23(8), 2053–2073.


Original code author: Yann N.Dauphin and Ian Goodfellow
https://github.com/goodfeli/adversarial/blob/master/parzen_ll.py
Modified by Boris van Breugel (bv292@cam.ac.uk)

"""

import numpy as np
import theano.tensor as T
import theano
from tqdm import tqdm


def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    
    nlls = []
    for i in range(n_batches):
        nll = parzen(x[inds[i::n_batches]])
        nlls.extend(nll)

    return np.array(nlls)


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """

    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * np.sqrt(np.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):
    
    lls = []
    for sigma in tqdm(sigmas):
        print(sigma)
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size = batch_size)
        lls.append(np.asarray(tmp).mean())
        del parzen
        
    ind = np.argmax(lls)
    return sigmas[ind]


def compute_parzen(orig_data, synth_data, sigma=None, start_sigma=-0.5, end_sigma=0.5, num_cv_evals=10, batch_size = 10):
    # Preprocess the data
    orig_data = np.asarray(orig_data)
    synth_data = np.asarray(synth_data)
        
    no, x_dim = np.shape(orig_data)
        
    
    
    if sigma is None:
        # Divide train / test
        orig_data_valid = orig_data[:int(no/5),:]
        orig_data_test = orig_data[int(no/5):,:]
            
        synth_data_valid = synth_data[:int(no/5),:]
        synth_data_test = synth_data[int(no/5):,:]
        sigma_range = np.logspace(start_sigma, end_sigma, num=num_cv_evals)
        sigma = cross_validate_sigma(synth_data_valid, orig_data_valid, sigma_range, batch_size)
    else:
        orig_data_test = orig_data
        synth_data_test = synth_data
    # fit and evaluate
    print('Using Sigma:', sigma)
    parzen = theano_parzen(synth_data_test, sigma)
    ll = get_nll(orig_data_test, parzen, batch_size = batch_size)
    se = ll.std() / np.sqrt(orig_data_test.shape[0])

    return ll.mean(), se