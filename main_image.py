# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:46:17 2021

@author: Boris van Breugel (bv292@cam.ac.uk)
"""

import os
import glob

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

from representations.OneClass import OneClassLayer
from metrics.combined import compute_metrics


from PIL import Image

tf.config.run_functions_eagerly(True)


#%%


OC_params  = dict({"rep_dim": 100, 
                "num_layers": 2, 
                "num_hidden": 200, 
                "activation": "ReLU",
                "dropout_prob": 0.5, 
                "dropout_active": False,
                "LossFn": "SoftBoundary",
                "lr": 1e-3,
                "epochs": 500,
                "warm_up_epochs" : 10,
                "train_prop" : 0.8,
                "weight_decay": 1e-2}   
)   

OC_hyperparams = dict({"Radius": 1, "nu": 1e-2})

which_metric = [['OC','PRDC','ID'],['OC']]



#%%

def reset_weights(model):
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
    for k, initializer in layer.__dict__.items():
        if "initializer" not in k:
            continue
      # find the corresponding variable
        var = getattr(layer, k.replace("_initializer", ""))
        var.assign(initializer(var.shape, var.dtype))
    return model


def identity_flatten(array):
    return array.flatten(array.shape[0],-1)
 
    
def remove_layer(model):
        new_input = model.input
        hidden_layer = model.layers[-2].output
        return Model(new_input, hidden_layer)   


def load_embedder(embedding):
    if embedding['model'] == 'vgg16' or embedding['model'] == 'vgg':
        model = tf.keras.applications.VGG16(include_top = True, weights='imagenet')
        model = remove_layer(model)
        model.trainable = False
    
        
    elif embedding['model'] == 'inceptionv3' or embedding['model'] == 'inception':
        model = tf.keras.applications.InceptionV3(include_top = True, weights='imagenet')
        model = remove_layer(model)
        model.trainable = False
    
    if embedding['randomise']:
        model = reset_weights(model)
        if embedding['dim64']:
            # removes last layer and inserts 64 output
            model = remove_layer(model)
            new_input = model.input
            hidden_layer = tf.keras.layers.Dense(64)(model.layers[-2].output)
            model = Model(new_input, hidden_layer)   
    
    #model.eval()
    return model

def plot_all(x, res, x_axis, metric_keys=None):
    """ Plots results of experiment with varying mode drop/authenticity"""
    if type(res) == type([]):
        plot_legend = False
        res = {'0':res}
    else:
        plot_legend = True
    exp_keys = list(res.keys())
    
    if metric_keys is None:
        metric_keys = res[exp_keys[0]][0].keys() 
    
    for m_key in metric_keys:
        for e_key in exp_keys:
          y = [res[e_key][i][m_key] for i in range(len(x))]
          plt.plot(x, y, label=e_key)
        plt.ylabel(m_key)
        plt.ylim(bottom=0)
        plt.xlabel(x_axis) 
        if plot_legend:
            plt.legend()
        plt.show()


#%% load data and conpute activations


def load_image_batch(files,shape):
    """Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
    """
    images = np.zeros((len(files),shape,shape,3))
    for i in range(len(files)):
        image = load_img(files[i], target_size=(shape,shape))
        images[i] = img_to_array(image)
    images = preprocess_input(images)

    images = tf.convert_to_tensor(images)
    return images

def load_all_images(files):
    """Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi*wi)
    """
    images = np.zeros((len(files),28**2))
    for i in range(len(files)):
        image = np.array(Image.open(files[i])).reshape(1,-1)
        images[i] = image
    images = preprocess_input(images)
    return images




def get_activations_from_files(files, embedder, batch_size=None, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- embedding        : embedding type (Inception, VGG16, None)
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, embedding_size) that contains the
       activations of the given tensor when feeding e.g. inception with the query tensor.
    """
    
    n_imgs = len(files)
    
    if batch_size is None:
        batch_size = n_imgs
    elif batch_size > n_imgs:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_imgs
    n_batches = n_imgs//batch_size + 1
    

    pred_arr = np.empty((n_imgs,embedder.output.shape[-1]))
    input_shape = embedder.input.shape[1]
    
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_imgs:
            end = start+batch_size
        else:
            end = n_imgs
        
 
        batch = load_image_batch(files[start:end], input_shape)
        pred = embedder(batch)
        pred_arr[start:end] = pred
        del batch #clean up memory
        
    if verbose:
        print(" done")
    
    return pred_arr

    
#%% MNIST imbalance experiment
def activation_loader_per_class(path_set, embedding, verbose = False):
    print('#### Embedding info',embedding, '#####')
    activations = []
    results = []
    # Load embedder function
    if embedding is not None:
        embedder = load_embedder(embedding)

    # Check if folder exists
    for label in range(10):
        path = os.path.join(path_set,str(label))
        if not os.path.exists(path):
            raise RuntimeError("Invalid path: %s" % path)
        # Don't embed data if no embedding is given 
        if embedding is None:
            files = list(glob.glob(os.path.join(path,'**/*.jpg'),recursive=True)) + list(glob.glob(os.path.join(path,'**/*.png'),recursive=True)) 
            act = load_all_images(files)    
    
        else:
            act_filename = f'{path}/act_{embedding["model"]}_{embedding["dim64"]}_{embedding["randomise"]}'
            # Check if embeddings are already available
            if load_act and os.path.exists(f'{act_filename}.npz'):
                print('Loaded activations from', act_filename)
                data = np.load(f'{act_filename}.npz',allow_pickle=True)
                act, _ = data['act'], data['embedding']
            # Otherwise compute embeddings
            else:
                if load_act:
                    print('Could not find activation file', act_filename)
                print('Calculating activations')
                files = list(glob.glob(os.path.join(path,'**/*.jpg'),recursive=True)) + list(glob.glob(os.path.join(path,'**/*.png'),recursive=True)) 
                act = get_activations_from_files(files, embedder, batch_size=64*8, verbose=verbose)
                # Save embeddings
                if save_act:
                    np.savez(f'{act_filename}', act=act,embedding=embedding)
        
        activations.append(act)
            
    return activations

def experiments(paths, embedding, OC_params, OC_hyperparams):
    
    #activations = activation_loader_per_class(paths, embedding)
    # random assignment
    path = paths[0]
    act_filename = f'{path}/act_{embedding["model"]}_{embedding["dim64"]}_{embedding["randomise"]}'
    print('Loading original data activations from', act_filename)
    data = np.load(f'{act_filename}.npz',allow_pickle=True)
    X = data['act']
    
    Y_per_class = activation_loader_per_class(paths[1],embedding)
    Y_zero = Y_per_class[0]
    num_per_class = len(Y_per_class[1])
    
    Y_other = np.concatenate(Y_per_class[1:],axis=0)
    step_size = 0.04
    
    # Train OC model
    OC_filename = f'metrics/OC_model_{dataset}_{embedding["model"]}_{embedding["dim64"]}.pkl' 
            
    if train_OC:
        OC_params['input_dim'] = X.shape[1]
        if OC_params['rep_dim'] is None:
            OC_params['rep_dim'] = X.shape[1]
        # Check center definition !
        OC_hyperparams['center'] = torch.ones(OC_params['rep_dim'])*10
        
        OC_model = OneClassLayer(params=OC_params, 
                                 hyperparams=OC_hyperparams)
        OC_model.fit(X,verbosity=True)
        if save_OC:
            pickle.dump((OC_model, OC_params, OC_hyperparams),open(OC_filename,'wb'))
    else:
        OC_model, OC_params, OC_hyperparams = pickle.load(open(OC_filename,'rb'))
    
    
    #simultaneous dropping
    p_mode_drops = np.arange(0,1+step_size,step_size)
    

    res_sim = []
    res_seq = []
    
    for p_mode_drop in p_mode_drops:
        mode_drop = np.random.rand(len(Y_other))<p_mode_drop
        Y_A = Y_zero[:num_per_class+np.sum(mode_drop)]
        Y_B = Y_other[1-mode_drop]
        Y = np.concatenate((Y_A, Y_B), axis=0)
        res = compute_metrics(X, Y, which_metric, model = OC_model)
        res_sim.append(res)
 
    # sequential dropping    
    for p_mode_drop in p_mode_drops:
        n_mode_drop = np.sum(np.random.rand(len(Y_other))<p_mode_drop)
        Y_A = Y_zero[:num_per_class+n_mode_drop]
        Y_B = Y_other[:-n_mode_drop]
        Y = np.concatenate(Y_A, Y_B, axis=0)
        res = compute_metrics(X, Y, which_metric, model = OC_model)
        res_seq.append(res)
    
    
    res = {}
    res['Simultaneous'] = res_sim
    res['Sequential'] = res_seq
    plot_all(p_mode_drops, res, r'p')
    
    
    # Copied proportion
    step_size = 0.1
    p_copied = np.arange(0,1+step_size,step_size)
    
    res = []

    for p in p_copied:
        c = int(p*len(X))
        Y_p = np.concatenate((X[:c],Y[c:]),axis=0)
        res_ = compute_metrics(X, Y_p, model=OC_model)
        res.append(res_)    
        
    plot_all(p_copied, res, r'$p_{copied}$') 
    

    
    





#%% main

def main(paths, embedding, OC_params, OC_hyperparams, load_act=True, save_act=True,  verbose = False):
    ''' Calculates the FID of two paths. '''
    
    print('#### Embedding info',embedding, '#####')
    activations = []
    results = []
    # Load embedder function
    if embedding is not None:
        embedder = load_embedder(embedding)

    # Loop through datasets
    for path_index, path in enumerate(paths):
        print('============ Path', path, '============')
        # Check if folder exists
        if not os.path.exists(path):
            raise RuntimeError("Invalid path: %s" % path)
        # Don't embed data if no embedding is given 
        if embedding is None:
            files = list(glob.glob(os.path.join(path,'**/*.jpg'),recursive=True)) + list(glob.glob(os.path.join(path,'**/*.png'),recursive=True)) 
            act = load_all_images(files)    
    
        else:
            act_filename = f'{path}/act_{embedding["model"]}_{embedding["dim64"]}_{embedding["randomise"]}'
            # Check if embeddings are already available
            if load_act and os.path.exists(f'{act_filename}.npz'):
                print('Loaded activations from', act_filename)
                print(act_filename)
                data = np.load(f'{act_filename}.npz',allow_pickle=True)
                act, _ = data['act'], data['embedding']
            # Otherwise compute embeddings
            else:
                if load_act:
                    print('Could not find activation file', act_filename)
                print('Calculating activations')
                files = list(glob.glob(os.path.join(path,'**/*.jpg'),recursive=True)) + list(glob.glob(os.path.join(path,'**/*.png'),recursive=True)) 
                act = get_activations_from_files(files, embedder, batch_size=64*8, verbose=verbose)
                # Save embeddings
                if save_act:
                    np.savez(f'{act_filename}', act=act,embedding=embedding)
                    
        activations.append(act)            
        
        # compute metrics
        if path_index == 0:
            OC_filename = f'metrics/OC_model_{dataset}_{embedding["model"]}_{embedding["dim64"]}.pkl' 
            
            if train_OC:
                
                OC_params['input_dim'] = act.shape[1]
    
                if OC_params['rep_dim'] is None:
                    OC_params['rep_dim'] = act.shape[1]
                # Check center definition !
                OC_hyperparams['center'] = torch.ones(OC_params['rep_dim'])
                
                OC_model = OneClassLayer(params=OC_params, 
                                         hyperparams=OC_hyperparams)
                OC_model.fit(act,verbosity=True)
                if save_OC:
                    pickle.dump((OC_model, OC_params, OC_hyperparams),open(OC_filename,'wb'))
            
            else:
                OC_model, OC_params, OC_hyperparams = pickle.load(open(OC_filename,'rb'))
            
            print(OC_params)
            print(OC_hyperparams)
            OC_model.eval()
            
        else:
            results.append([compute_metrics(activations[0], act, model=OC_model), path])
        
        
    return results


#%%

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    #for embed in ['inceptionv3','vgg','identity']:
    methods = ['WGAN-GP','DCGAN','VAE','CGAN', 'ADS-GAN']
    load_act = True
    save_act = False
    nul_path = ['data/mnist/original/testing']
    conditional_path = ['data/mnist/cgan_exp']
    random_path = ['data/mnist/random']
    other_paths = [f'data/mnist/synth/{method}' for method in methods]
    paths = random_path # nul_path + random_path + other_paths
    
    dataset = 'MNIST'
    train_OC = False
    save_OC= True
    
    #embeddings.append(None)
    embeddings = []
    
    embeddings.append({'model':'inceptionv3',
                 'randomise': False, 'dim64': False})
    embeddings.append({'model':'vgg16',
                 'randomise': False, 'dim64': False})
    #embeddings.append({'model':'vgg16',
    #             'randomise': True, 'dim64': False})
    embeddings.append({'model':'vgg16',
                'randomise': True, 'dim64': True})
    
    outputs = []
    experiments(nul_path+conditional_path, embeddings[2], OC_params, OC_hyperparams)
    #for embedding in embeddings:

            #output = main(paths, embedding, OC_params, OC_hyperparams, load_act, save_act,verbose=True)
            #outputs.append(output)
    