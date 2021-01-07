# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:37:08 2020

@author: boris
"""


import torch
import torch.optim as optim

from model import LinearVAE
#from model import CausalVAE, LinearVAE
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np



# print settings numpy
np.set_printoptions(suppress=True)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X):
        'Initialization'
        self.X = X
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        # Load data and get label
        X = self.X[index]
        return X



### loss functions

def gaussian_likelihood(x_hat, x, logscale):
    scale = torch.exp(logscale)
    dist = torch.distributions.Normal(x, scale)
    # measure prob of seeing measurements under p(x|y)
    likelihood = dist.log_prob(x_hat)
    return likelihood
    

def kl_div(mu, std):
    std = torch.exp(std)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)
    return torch.distributions.kl.kl_divergence(p,q)


def elbo_loss(x_hat, x, mu, logvar, log_scale):
    
    recon_loss = -reconstruction_likelihood(x_hat, x, log_scale)
    prior_loss = kl_div(mu, logvar)
    loss = recon_loss.mean()+prior_loss.mean()
    return loss



def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        optimizer.zero_grad()
        x = data
        x = x.to(device)
        
        
        x_hat, mu, logvar, _  = model(x)
        loss = elbo_loss(x_hat, x, mu, logvar, model.log_scale)
        
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def validate(model, dataloader, verbose = False):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            x = data
            x = x.to(device)
            
            
            x_hat, mu, logvar = model(x)
            #x_hat_noisy, _, _, x_hat_noisy_2 = model(x_noisy)
            
            loss = elbo_loss(x_hat, x, mu, logvar, model.log_scale)
            running_loss += loss.item()
        
            
    
    
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss




class vae(df):
    def __init__(self, df):
        
        self.X = df.to_numpy()
        self.n,self.d = self.X.shape
        
        
        self.extra_layer = 3
        self.num_latents = self.d
        self.factor = 1
        
        # learning parameters
        self.epochs = 10
        self.batch_size = 32
        self.lr = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        prop_train = 0.8
    
        cut_off = round(self.n*prop_train)
        train_data = Dataset(np.array(self.X[:cut_off], dtype = 'float32'))
        val_data = Dataset(np.array(self.X[cut_off:], dtype = 'float32'))
        
        
        # training and validation data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False
        )
    
    
    model = LinearVAE(d,num_latents, num_extra_layers = extra_layer, factor=factor).to(device)
    num_par = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_par)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    noise_prior = 'gaussian'
    if noise_prior == 'gaussian':
        reconstruction_likelihood = gaussian_likelihood
    
    
    train_loss = []
    val_loss = []
    val_epoch_loss = validate(model, val_loader, verbose = True)
    print('Start training')
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = fit(model, train_loader)
        val_epoch_loss = validate(model, val_loader, verbose = True)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")
    
 
    
    