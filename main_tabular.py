# pylint: disable=redefined-outer-name

import os
import pickle
import time  # pylint: disable=unused-import
import glob

# for predictive tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import torch
import tensorflow as tf

from representations.OneClass import OneClassLayer
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



#%% Import functions
from generative_models.adsgan import adsgan
from generative_models.gan import gan
from generative_models.pategan import pategan
from generative_models.vae import vae
from generative_models.dpgan import dpgan

from metrics.combined import compute_metrics
import metrics.prd_score as prd  # pylint: disable=unused-import

from audit import audit

if utils.check_tf2():
    from main_image import get_activation
    from main_image import plot_all
else:
    print("TF2 not found, cannot import from main_image")

#%% Constants and settings

# Options: ("main", "main_from_files", "lambda", "audit",)
run_experiment = "main"

# Options: ("orig", "random", "adsgan", "wgan", "vae", "gan", "dpgan")
methods =['adsgan', 'wgan', 'vae', 'gan', 'dpgan']

# Options: ("covid", "bc")
dataset = 'covid'

original_data_dir = 'data/tabular/original'
synth_data_dir = 'data/tabular/synth'
visual_dir = 'visualisations'



debug_train = False
debug_metrics = False
just_metrics = False


# Save synthetic data iff we're training
# Train generative models
do_train = True
save_synth = False
# just relevant for ADS-GAN


# Train OneClass representation model
train_OC = True
# If train is true, save new model (overwrites old OC model)
save_OC = False
 
which_metric = [['ID','OC','WD','FD', 'parzen', 'PRDC'],['OC']]


if utils.check_tf2():
    tf.random.set_seed(2021)
else:
    tf.compat.v1.set_random_seed(2021)
np.random.seed(2021)


# OneClass representation model
OC_params  = {
    "rep_dim": None, 
    "num_layers": 4, 
    "num_hidden": 32, 
    "activation": "ReLU",
    "dropout_prob": 0.2, 
    "dropout_active": False,
    "LossFn": "SoftBoundary",
    "lr": 2e-3,
    "epochs": 1000,
    "warm_up_epochs" : 20,
    "train_prop" : 1.0,
    "weight_decay": 2e-3
}   

lambda_ = 0.1

OC_hyperparams = {
    "Radius": 1, 
    "nu": 1e-2
}



#%% Data loading

def load_breast_cancer_data():
    # pylint: disable=no-member
    data = load_breast_cancer()
    X = MinMaxScaler().fit_transform(data.data)
    df = pd.DataFrame(X, columns=data.feature_names)
    target = 'target'
    df[target] = data.target
    df = df.dropna(axis=0, how='any')
    return df


def load_covid_data():
    df = pd.read_csv(f'{original_data_dir}/brazilian_covid_data.csv')
    df.rename(columns={'is_dead':'target'},inplace=True)
    # drop redundant columns that are contained in other columns
    df = df.drop(columns=['Sex','Race','SG_UF_NOT','Age_40','Age_40_50',
                      'Age_50_60','Age_60_70','Age_70','Branca'])
    X = MinMaxScaler().fit_transform(df)
    
    df = pd.DataFrame(X,columns = df.columns)
    return df


def load_mnist_data(path, embedding_no=3):
    embeddings = []
    embeddings.append({'model':'inceptionv3',
                 'randomise': False, 'dim64': False})
    embeddings.append({'model':'vgg16',
                 'randomise': False, 'dim64': False})
    embeddings.append({'model':'vgg16',
                 'randomise': True, 'dim64': False})
    embeddings.append({'model':'vgg16',
                'randomise': True, 'dim64': True})
    return get_activation(path, 
                          embedding=embeddings[embedding_no]) 



#%% Feature importance plots

def feature_importance_plot(X,y, num_features = 10):
    
    forest = RandomForestClassifier()
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
               axis=0)
    indices = np.argsort(importances)[::-1]
    indices = indices[:num_features]
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
    
    
    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices],
            color="r", yerr=std[indices], align="center")
    tick_names = indices # X.columns[indices]
    plt.xticks(range(X.shape[1]), tick_names)
    plt.xlim([-1, X.shape[1]])
    plt.show()


def feature_importance_comparison(X, Y, method_names=None):
    
    num_methods = len(X)
    
    n_trees = 1000
    importances = []
    stds = []
    
    for i in range(num_methods):
        forest = ExtraTreesClassifier(n_trees)
        forest.fit(X[i], Y[i])
        importances.append(forest.feature_importances_)
        stds.append(np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0))
        
        if method_names is not None and i>0:
            print(f'Correlation of importances or method {method_names[i-1]}:')
            print(np.corrcoef(importances[0],importances[i])[0,1])
    
    if method_names is not None:
        bar_comparison(importances, 
                   stds, labels=method_names, tick_names = X[0].columns, save_name = 'all_feat_importance')

    return [importances, stds]


def cv_predict_scores(X, y, classifier, n_splits=5):
    """
    Computes CV accuracy and AUROC
    
    Parameters
    ----------
    X : numpy, pandas
        Data
    y : numpy, pandas
        Labels
    classifier : 
        Predictive model
    n_splits : int, optional
        Number of CV splits. The default is 6.
        
    Returns
    -------
    mean_auc : float
        CV AUC
    mean_acc : float
        CV accuracy.

    """
    cv = StratifiedKFold(n_splits=n_splits)
        
    accs = np.zeros(n_splits)
    aucs = np.zeros(n_splits) 
    
    X = np.array(X)
    y = np.array(y)
    
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        accs[i] = classifier.score(X[test], y[test])
        
        scores = classifier.predict_proba(X[test])
        
        auc = metrics.roc_auc_score(y[test], scores[:,1])
        aucs[i] = auc
        
    mean_acc = np.mean(accs)
    mean_auc = np.mean(aucs)
    std_acc = np.std(accs)
    std_auc = np.std(aucs)    
    
    return mean_acc, mean_auc, std_acc, std_auc


def transfer_scores(X, y, X_s, y_s, classifier):
    
    X = np.array(X)
    y = np.array(y)
    X_s = np.array(X_s)
    y_s = np.array(y_s)
    
    classifier.fit(X_s, y_s)
    acc = classifier.score(X, y)
        
    scores = classifier.predict_proba(X)        
    auc = metrics.roc_auc_score(y, scores[:,1])
    
    return acc, auc
    

def predictive_model_comparison(orig_X, orig_Y, synth_X, synth_Y, method_name=None, models=None):
    if models is None:
        models = [LogisticRegression(max_iter=300), 
              KNeighborsClassifier(), 
              MLPClassifier(max_iter=100),
              RandomForestClassifier(),
              SVC(probability=True),
              GaussianNB()
              ]
        model_names = ['Logistic', 'KNeighbour', 'MLP', 'Forest', 'SVM', 
                       'GaussNB']
    num_models = len(models)
    accs = np.zeros(num_models)
    aucs = np.zeros(num_models)
    synth_accs = np.zeros(num_models)
    synth_aucs = np.zeros(num_models)
    transf_accs = np.zeros(num_models)
    transf_aucs = np.zeros(num_models)
    std_accs = np.zeros(num_models)
    std_aucs = np.zeros(num_models)
    std_synth_accs = np.zeros(num_models)
    std_synth_aucs = np.zeros(num_models)
    std_transf_accs = np.zeros(num_models)
    std_transf_aucs = np.zeros(num_models)
    
    
    for i, model in enumerate(models):
        print('### Predictability scores for model', model)
        #original dataset performance
        acc, auc, std_acc, std_auc = cv_predict_scores(orig_X, orig_Y, model)
        print('Accuracy original:', acc)
        print('AUC original     :', auc)
        accs[i] = acc
        aucs[i] = auc
        std_accs[i] = std_acc
        std_aucs[i] = std_auc
    
        #synthetic dataset performance
        acc, auc, std_acc, std_auc = cv_predict_scores(synth_X, synth_Y, model)
        print('Accuracy synthetic:', acc)
        print('AUC synthetic     :', auc)
        synth_accs[i] = acc
        synth_aucs[i] = auc
        std_synth_accs[i] = std_acc
        std_synth_aucs[i] = std_auc
        
        ## how training on synthetic data performs on original data
        acc, auc = transfer_scores(orig_X, orig_Y, synth_X, synth_Y, model)
        print('Accuracy transfer:', acc)
        print('AUC transfer     :', auc)
        transf_accs[i] = acc
        transf_aucs[i] = auc
        std_transf_accs[i] = std_acc
        std_transf_aucs[i] = std_auc              
    
    # plot results
    if method_name is not None:
        bar_comparison([accs, synth_accs, transf_accs], 
                       [std_accs, std_synth_accs, std_transf_accs], 
                       tick_names=model_names, save_name = f'{method_name}_pred_accs')
        bar_comparison([aucs, synth_aucs, transf_aucs], 
                       [std_aucs, std_synth_aucs, std_transf_aucs], 
                       tick_names=model_names, save_name = f'{method_name}_pred_aucs')
        
    return {'acc':{'original':accs,'synthetic':synth_accs, 'transfer': transf_accs},
            'std_acc': {'original': std_accs, 'synthetic': std_synth_accs, 'transfer':std_transf_accs},
            'auc':{'original':aucs,'synthetic':synth_aucs, 'transfer': transf_aucs},
            'std_auc': {'original': std_aucs, 'synthetic': std_synth_aucs, 'transfer':std_transf_aucs}}
            


#%% Misc
    
def bar_comparison(vectors, std=None, labels=None, tick_names=None, save_name = None, max_length = 10):
    
    num_bars = len(vectors)
    vector = vectors[0]
    indices = np.argsort(vector)[::-1]
    indices = indices[:max_length]
    fig, ax = plt.subplots()
    tot_bar_width = 0.7
    width = tot_bar_width/num_bars
    x = np.arange(len(indices)) 
    
    if tick_names is None:
        tick_names = range(len(vector))
    
    if labels is None:
        labels = ['Original', 'Synthetic', 'Transfer']
    
    for i, vec in enumerate(vectors):
        xbar = x - tot_bar_width/2 + (i+1/2)*width
        if std is not None:    
            ax.bar(xbar, vec[indices],  yerr=std[i][indices], width=width, label=labels[i])
        else:
            ax.bar(xbar, vec[indices], width=width, label=labels[i])

    # df.set_index('a', inplace=True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    ticks = np.array(tick_names, dtype='object')
    print(indices,ticks)
    ticks = ticks[indices]
    plt.xticks(x, ticks)
    plt.legend()
    plt.xlim([-1, len(indices)])
    if save_name is not None:
        plt.savefig(f'{visual_dir}/{dataset}_{save_name}.jpg')
    plt.show()


def roc(X, y, classifier, n_splits=6, pos_label = 2):  # pylint: disable=unused-argument
    cv = StratifiedKFold(n_splits=n_splits)
    
    tprs = []
    aucs = []
    accs = np.zeros(n_splits)
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots()
    X = np.array(X)
    y = np.array(y)
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        accs[i] = classifier.score(X[test], y[test])
        
        viz = metrics.plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_acc = np.mean(accs)
    
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    
    plt.show()
    
    return mean_auc, mean_acc    


#%%  
# Set settings:

    
def main_from_files(OC_params, OC_hyperparams):
    if dataset == 'mnist':
        orig_data = load_mnist_data('data/mnist/original/testing')
    
    elif dataset == 'ts':
        orig_data = np.load(glob.glob('data/ts/original/*')[0])
        synth_files = glob.glob('data/ts/synthetic/*')
    
    print('Shape original data:', orig_data.shape)

    OC_filename = f'metrics/OC_model_{dataset}.pkl'    
    if train_OC or not os.path.exists(OC_filename):
        print('### Training OC embedding model')
        
        if OC_params['rep_dim'] is None:
            OC_params['rep_dim'] = orig_data.shape[1]
        # Check center definition !
        OC_hyperparams['center'] = torch.ones(OC_params['rep_dim'])*10
        OC_params['input_dim'] = orig_data.shape[1]
        OC_model = OneClassLayer(params=OC_params, 
                                 hyperparams=OC_hyperparams)
        OC_model.fit(orig_data, verbosity=True)
        if save_OC:
            pickle.dump((OC_model, OC_params, OC_hyperparams),open(OC_filename,'wb'))
    else:
        OC_model,OC_params, OC_hyperparams = pickle.load(open(OC_filename,'rb'))
        print('### Loaded OC embedding model')
        print('Parameters:', OC_params)
        print('Hyperparameters:', OC_hyperparams)
        
    # parameters for generative models
    params = dict()
    params["iterations"] = 2000
    params["h_dim"] = 200
    params["z_dim"] = 32
    params["mb_size"] = 128

    
    results = {}
    for path in synth_files:
        print(f'\n============== {path} ==============')
        synth_data = np.load(path)
        if np.sum(np.isnan(synth_data))>0:
            print('Data contains NaN. Will skip this path')
            continue
        print('#### Computing static metrics')
        results_metrics = compute_metrics(orig_data, synth_data, 
                                          which_metric=which_metric, 
                                          wd_params = params, model=OC_model)
        results[path] = results_metrics
    return results
    

def get_OC_model(OC_filename, train_OC, X=None, OC_params=None, OC_hyperparams=None):
    if train_OC or not os.path.exists(OC_filename):
        
        OC_params['input_dim'] = X.shape[1]

        if OC_params['rep_dim'] is None:
            OC_params['rep_dim'] = X.shape[1]
        # Check center definition !
        OC_hyperparams['center'] = torch.ones(OC_params['rep_dim'])#*10
        
        OC_model = OneClassLayer(params=OC_params, 
                                 hyperparams=OC_hyperparams)
        OC_model.fit(X,verbosity=True)
        if save_OC:
            pickle.dump((OC_model, OC_params, OC_hyperparams),open(OC_filename,'wb'))
    
    else:
        OC_model, OC_params, OC_hyperparams = pickle.load(open(OC_filename,'rb'))
        print('### Loaded OC embedding model')
        print('Parameters:', OC_params)
        print('Hyperparameters:', OC_hyperparams)
    
    OC_model.eval()
    return OC_model, OC_params, OC_hyperparams


def experiment_audit(OC_params, OC_hyperparams):
    plt.close('all')

    # Load data
    if dataset == 'bc':
        orig_data = load_breast_cancer_data()  
    elif dataset == 'covid':
        orig_data = load_covid_data()  
    else:
        raise ValueError('Not a valid dataset name given')
    
    
    OC_params['input_dim'] = orig_data.shape[1]
    print(orig_data.shape)
    n_orig = orig_data.shape[0]
    
    OC_filename = f'metrics/OC_model_{dataset}.pkl'    
    OC_model,OC_params, OC_hyperparams = get_OC_model(OC_filename, train_OC, orig_data.to_numpy(), OC_params, OC_hyperparams)
    
    # parameters for generative models
    params = dict()
    params["iterations"] = 2000
    params["h_dim"] = 100
    params["z_dim"] = 10
    params["mb_size"] = 128
    
    method = 'adsgan'    
    params['gen_model_name'] = method
    params['lambda'] = 1
    
    
    synth_data = audit(orig_data, params, OC_model)
    print('Shape after auditing:',synth_data.shape)

    results_after_auditing = compute_metrics(orig_data.to_numpy(), synth_data, 
                                        which_metric = which_metric, 
                                        wd_params = params, model=OC_model)
    
    
    

    return results_after_auditing, synth_data


def experiment_lambda_adsgan(OC_params, OC_hyperparams, lambdas=None):
    plt.close('all')
    if lambdas is None:
        lambdas = np.exp(np.arange(-3,3,0.2))
        lambdas[0] = 0
    # Load data
    if dataset == 'bc':
        orig_data = load_breast_cancer_data()  
    elif dataset == 'covid':
        orig_data = load_covid_data()  
    else:
        raise ValueError('Not a valid dataset name given')
    
    
    OC_params['input_dim'] = orig_data.shape[1]
    print(orig_data.shape)
    OC_filename = f'metrics/OC_model_{dataset}.pkl'    
    OC_model,OC_params, OC_hyperparams = get_OC_model(OC_filename, train_OC, orig_data.to_numpy(), OC_params, OC_hyperparams)
        
    # parameters for generative models
    params = dict()
    params["iterations"] = 2000
    params["h_dim"] = 200
    params["z_dim"] = 10
    params["mb_size"] = 128
    params['lambda_tester'] = True
        
    all_results = []
    
    
    method = 'adsgan'    
    params['gen_model_name'] = method

    for lambda_ in lambdas:
        
        print('Lambda is ',lambda_)
        params["lambda"] = lambda_
        synth_data = adsgan(orig_data, params)
            
    
        print('#### Computing static metrics')
        results_metrics = compute_metrics(orig_data.to_numpy(), synth_data, 
                                          which_metric = which_metric, 
                                          wd_params = params, model=OC_model)
        
        results_metrics['lambda'] = lambda_
        all_results.append(results_metrics)
    
    plot_all(lambdas, all_results, r'$\lambda$',name='ads-gan_lambda_test_'+dataset)
    
    return all_results


def main(OC_params, OC_hyperparams):
    plt.close('all')
    
    prc_curves = []
    
    # Load data
    if dataset == 'bc':
        orig_data = load_breast_cancer_data()  
    elif dataset == 'covid':
        orig_data = load_covid_data()  
    else:
        raise ValueError('Not a valid dataset name given')
    
    # Some different data definitions
    #orig_train_index = round(len(orig_data)*train_ratio)
    orig_X, orig_Y = orig_data.drop(columns=['target']), orig_data.target
    
    OC_params['input_dim'] = orig_data.shape[1]
    OC_filename = f'metrics/OC_model_{dataset}.pkl'    
    OC_model,OC_params, OC_hyperparams = get_OC_model(OC_filename, train_OC, orig_data.to_numpy(), OC_params, OC_hyperparams)
        
    # parameters for generative models
    params = dict()
    params["iterations"] = 2000
    params["h_dim"] = 200
    params["z_dim"] = 10
    params["mb_size"] = 128
        
    metric_results = {}
    pred_perf = {}
    X, Y = [orig_data.drop(columns=['target'])], [np.array(orig_data.target, dtype='bool')]
    print(np.unique(np.array(orig_data.target)))
    for method in methods:
        print(f'\n============== {method} ==============')
        filename =  f'{synth_data_dir}/{dataset}_{method}.csv'
        
        params['gen_model_name'] = method
        
        if method != 'adsgan':
            params['lambda'] = 0
        else:
            params["lambda"] = lambda_
        print('Lambda is ', lambda_)
        
    
        # Synthetic data generation
        if method == 'orig':
            # This is for sanity checking metrics
            synth_data = orig_data.to_numpy()
        elif method == 'random':
            # Also for sanity check
            synth_data = np.random.uniform(size=orig_data.shape)
        elif do_train:
            if method in ['wgan','gan']:
                synth_data = gan(orig_data, params)
            elif method == 'adsgan':
                synth_data = adsgan(orig_data, params)
            elif method == 'pategan':
                params_pate = {
                    'n_s': 1, 'batch_size': 128, 
                    'k': 100, 'epsilon': 100, 'delta': 0.0001, 'lambda': 1
                }
                synth_data = pategan(orig_data.to_numpy(), params_pate)  
            elif method=='vae':
                synth_data = vae(orig_data, params)
            elif method == "dpgan":
                customized_dpgan_params = {
                    "inputDim": orig_data.shape[1],
                    "sigma": 10.0,  # NOTE: DP noise level.
                    "nEpochs": 1000,
                    "batchSize": 1024,
                    "pretrainEpochs": 200, 
                    "pretrainBatchSize": 128,
                }
                synth_data = dpgan(orig_data.to_numpy(), customized_dpgan_params)
                
            if save_synth:
                pickle.dump((synth_data, params),open(filename,'wb'))
        
        else:
            synth_data, params = pickle.load(open(filename,'rb'))
        
        for i in range(synth_data.shape[1]):
            if len(np.unique(orig_data.to_numpy()[:, i])) == 2:
                synth_data[:, i] = np.array(np.round(synth_data[:, i]), dtype='int')
    

        print('#### Computing static metrics')
        results_metrics = compute_metrics(orig_data.to_numpy(), synth_data, 
                                          which_metric=which_metric, 
                                          wd_params = params, model=OC_model)
        
        if 'OC' in which_metric[1]:
            apc = results_metrics['alpha_pc_OC']
            bcc = results_metrics['beta_cv_OC'] 
            alpha = results_metrics['alphas']
            prc_curves.append([alpha, apc])
        
        metric_results[method] = results_metrics
        


        synth_data = pd.DataFrame(synth_data,columns = orig_data.columns)  
        synth_X, synth_Y = synth_data.drop(columns=['target']), synth_data.target
        
        X.append(synth_X)
        Y.append(synth_Y)
        # synth_train_index = round(len(synth_data)*train_ratio)
        # synth_X_train, synth_X_test = synth_X[:synth_train_index], synth_X[synth_train_index:]
        # synth_Y_train, synth_Y_test = synth_Y[:synth_train_index], synth_Y[synth_train_index:]
        
        
        ### predictive performance
        
        ## RANKING: 
        # how ranking (accuracy and AUC) of different models compares
        # between the synthetic and original dataset
        #plt.close('all')
        print('#### Computing predictive metrics')
        
        if debug_metrics or just_metrics:
            pred_perf[method] = None
        else:
            pred_perf[method] = predictive_model_comparison(orig_X, orig_Y, synth_X, synth_Y, method_name=method)
        
        
        
    if just_metrics:
        return metric_results
        
    feat_imp = feature_importance_comparison(X,Y, method_names=['orig']+methods)
    
    #if 'OC' in which_metric[1]:
    #    prd.plot([prc_curves], out_path=None)

    results={'metric_results': metric_results}
    results['pred_perf'] = pred_perf
    results['feat_imp'] = feat_imp
    return results, [X,Y]



if __name__ == '__main__':

    if run_experiment == "main":
        # Normal tabular data:
        results, synth_data = main(OC_params, OC_hyperparams)
        pickle.dump(synth_data, open(f'results/synth_data{dataset}{round(time.time())}.pkl','wb'))
        pickle.dump(results, open(f'results/{dataset}{round(time.time())}.pkl','wb'))
    
    elif run_experiment == "audit":
        # Audit experiment:
        results_audit, synth_audit = experiment_audit(OC_params, OC_hyperparams)
        pickle.dump(results_audit, open(f'results/audit_{dataset}{round(time.time())}.pkl','wb'))
        pickle.dump(synth_audit, open(f'results/data_synth_audit_{dataset}{round(time.time())}.pkl','wb'))
    
    elif run_experiment == "lambda":
        # Lambda experiment:
        results_lamb = experiment_lambda_adsgan(OC_params, OC_hyperparams, lambdas=None)
        pickle.dump(results_lamb, open(f'results/{dataset}{round(time.time())}.pkl','wb'))    
    
    elif run_experiment == "main_from_files":
        results = main_from_files(OC_params, OC_hyperparams)
