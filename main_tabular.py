#%% Import necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# for predictive tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


import pickle



#%% Import functions
from generative_models.adsgan import adsgan
from generative_models.pategan import pategan
from generative_models.vae import vae



from metrics.feature_distribution import feature_distribution
from metrics.compute_wd import compute_wd
from metrics.compute_identifiability import compute_identifiability
from metrics.fid import compute_frechet_distance


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
                      'Age_50_60','Age_60_70','Age_70'])
    X = MinMaxScaler().fit_transform(df)
    df = pd.DataFrame(X,columns = df.columns)
    return df

    

#%% Feature importance plots

def feature_importance_plot(X,y):
    
    forest = RandomForestClassifier()
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
               axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
    
    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    tick_names = indices # X.columns[indices]
    plt.xticks(range(X.shape[1]), tick_names)
    plt.xlim([-1, X.shape[1]])
    plt.show()


def feature_importance_comparison(X,y, X_s, y_s):
    
    n_trees = 1000
    
    forest = ExtraTreesClassifier(n_trees)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    
    forest = ExtraTreesClassifier(n_trees)
    forest.fit(X_s, y_s)
    importances_s = forest.feature_importances_
    std_s = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    
    print('Correlation of importances:')
    print(np.corrcoef(importances,importances_s)[0,1])
    
    bar_comparison([importances, importances_s], 
                   [std, std_s], save_name = 'feat_importance')

    
def cv_predict_scores(X, y, classifier, n_splits=6):
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
    

def predictive_model_comparison(orig_X, orig_Y, synth_X, synth_Y, models=None):
    if models is None:
        models = [LogisticRegression(), 
              KNeighborsClassifier(), 
              MLPClassifier(max_iter=1000),
              RandomForestClassifier()]
        model_names = ['LogReg', 'KNeighbour', 'MLP', 'RandForest']
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
    plt.close('all')
    bar_comparison([accs, synth_accs, transf_accs], 
                   [std_accs, std_synth_accs, std_transf_accs], 
                   tick_names=model_names, save_name = 'pred_accs')
    bar_comparison([aucs, synth_aucs, transf_aucs], 
                   [std_aucs, std_synth_aucs, std_transf_aucs], 
                   tick_names=model_names, save_name = 'pred_aucs')
       


    
    

    

#%% Misc
    
def bar_comparison(vectors, std=None, labels=None, tick_names=None, save_name = None):
    
    num_bars = len(vectors)
    vector = vectors[0]
    indices = np.argsort(vector)[::-1]
    fig, ax = plt.subplots()
    tot_bar_width = 0.7
    width = tot_bar_width/num_bars
    x = np.arange(len(vector)) 
    
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
    ticks = ticks[indices]
    plt.xticks(x, ticks)
    plt.legend()
    plt.xlim([-1, len(vector)])
    if save_name is not None:
        plt.savefig(f'{visual_dir}/{dataset}_{method}_{save_name}.jpg')
    plt.show()


def roc(X, y, classifier, n_splits=6, pos_label = 2):
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
dataset = 'covid'
method = 'vae' #adsgan, wgan, gan, vae
do_train = True
original_data_dir = 'data/tabular/original'
synth_data_dir = 'data/tabular/synth'
visual_dir = 'visualisations'
debug_train = False

def main():
    plt.close('all')
    
    #Save synthetic data iff we're training
    save_synth = do_train
    
    filename =  f'{synth_data_dir}/{dataset}_{method}.csv'
    
        
    # parameters for ADS-GAN and Wasserstein distance metrics
    params = dict()
    params["lambda"] = 0.1
    params["iterations"] = 10000
    params["h_dim"] = 30
    params["z_dim"] = 10
    params["mb_size"] = 128
    params['gen_model_name'] = method
    
    if method != 'adsgan':
        params['lambda'] = 0
        
    train_ratio = 0.8
    
    # Load data
    if dataset == 'bc':
        orig_data = load_breast_cancer_data()  
    elif dataset == 'covid':
        orig_data = load_covid_data()  
    
    orig_train_index = round(len(orig_data)*train_ratio)
    orig_X, orig_Y = orig_data.drop(columns=['target']), orig_data.target
    orig_X_train, orig_X_test = orig_X[:orig_train_index], orig_X[orig_train_index:]
    orig_Y_train, orig_Y_test = orig_Y[:orig_train_index], orig_Y[orig_train_index:]
    
    
    # Synthetic data generation
    if do_train:
        if method in ['wgan','gan', 'adsgan']:
            synth_data = adsgan(orig_data, params)
        elif method == 'pategan':
            params_pate = {'n_s': 1, 'batch_size': 128, 
                 'k': 100, 'epsilon': 100, 'delta': 0.0001, 'lambda': 1}
            
        
            synth_data = pategan(orig_data.to_numpy(), params_pate)  
        elif method=='vae':
            synth_data = vae(orig_data, params)
            
        if save_synth:
            pickle.dump((synth_data, params),open(filename,'wb'))
    
    else:
        synth_data, params = pickle.load(open(filename,'rb'))
    
    if debug_train:
        return synth_data
    ## Performance measures from ADS-GAN paper
    # (1) Feature marginal distributions
    feat_dist = feature_distribution(orig_data, synth_data)
    print("Finish computing feature distributions")
    
    # (2) Wasserstein Distance (WD)
    print("Start computing Wasserstein Distance")
    wd_measure = compute_wd(orig_data, synth_data, params)
    print("WD measure: " + str(wd_measure))
    
    frechet_distance = compute_frechet_distance(orig_data, synth_data)
    print("Frechet distance: " + str(frechet_distance))
    
    # (3) Identifiability 
    identifiability = compute_identifiability(orig_data, synth_data)
    print("Identifiability measure: " + str(identifiability))
    
    
    
    
    # Some different data definitions
    synth_data = pd.DataFrame(synth_data,columns = orig_data.columns)
    
    synth_train_index = round(len(synth_data)*train_ratio)
    synth_X, synth_Y = synth_data.drop(columns=['target']), synth_data.target
    synth_X_train, synth_X_test = synth_X[:synth_train_index], synth_X[synth_train_index:]
    synth_Y_train, synth_Y_test = synth_Y[:synth_train_index], synth_Y[synth_train_index:]
    
    
    
    ### predictive performance
    
    ## RANKING: 
    # how ranking (accuracy and AUC) of different models compares
    # between the synthetic and original dataset
    plt.close('all')
    predictive_model_comparison(orig_X, orig_Y, synth_X, synth_Y)
    
    
    # example of ROC computation
    #roc(synth_X, synth_Y, LogisticRegression())
    
    
    ### Feature importance between orig and synth data
    feature_importance_comparison(orig_X, orig_Y, synth_X, synth_Y)
    
    return orig_data, synth_data


if __name__ == '__main__':
    orig_data, synth_data = main()
