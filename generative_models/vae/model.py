# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:36:00 2020

@author: boris
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from generate_synthetic import ancestors, directed2moralised
from collections import OrderedDict


class LinearVAE(nn.Module):
    def __init__(self, d, num_latents, num_extra_layers=0,factor=1):
        super(LinearVAE, self).__init__()
        self.num_latents = num_latents
        self.d = d
        self.num_extra_layers = num_extra_layers
        # encoder

        self.enc1 = nn.Linear(in_features=self.d, out_features=self.d*factor)
        if self.num_extra_layers:
            enc2 = []
            dec2 = []
            for i in range(self.num_extra_layers):
                enc2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                enc2.append(('r'+str(i),nn.ReLU()))
                dec2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                dec2.append(('r'+str(i),nn.ReLU()))
            enc2 = nn.Sequential(OrderedDict(enc2))
            self.enc2 = enc2
            dec2 = nn.Sequential(OrderedDict(dec2))
            self.dec2 = dec2
        self.enc3 = nn.Linear(in_features=self.d*factor, out_features=self.num_latents*2)
        
        # decoder 
        self.dec1 = nn.Linear(in_features=self.num_latents, out_features=self.d*factor)
           
        self.dec3 = nn.Linear(in_features=self.d*factor, out_features=self.d)
        
        
        self.log_scale = nn.Parameter(torch.zeros(d))
    # def add_noise(X, sigma=1/5):
    #     "Take input and add some noise"
    #     noise_term = torch.randn_like(X)
    #     return X + noise_term
        
        
    def reparameterize(self, mu, log_var):
        """
        :mu: mean from the encoder's latent space
        :log_var: log variance from the encoder's latent space
        """
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        return q.rsample()
 
        
        
    def forward(self, xor, m = None):
        if m is None:
            m = torch.ones_like(xor)# encoding
        for i in range(1):
            x = F.relu(self.enc1(xor))
            if self.num_extra_layers>0:
                x = self.enc2(x)
            
            x = self.enc3(x).view(-1, 2, self.num_latents)
            # get `mu` and `log_var`
            mu = x[:, 0, :] # the first feature values as mean
            log_var = x[:, 1, :] # the other feature values as variance
            # get the latent vector through reparameterization
            z = self.reparameterize(mu, log_var)
     
            # decoding
            x = F.relu(self.dec1(z))
            if self.num_extra_layers:
                x = self.dec2(x)
            x = self.dec3(x)
            #x = m * xor+(1-m) * x
        return x, mu, log_var, None
    


class CausalVAE(nn.Module):
    def __init__(self, d, adjacency, noise_features = 20, num_extra_layers = True, factor=1, no_noise=True):
        super(CausalVAE, self).__init__()
        self.num_latents = d
        self.d = d
        self.A = adjacency
        self.no_noise = no_noise
        self.log_scale = nn.Parameter(torch.zeros(d))
        self.num_extra_layers = num_extra_layers
        # encoder

        self.enc1 = nn.Linear(in_features=self.d, out_features=self.d*factor)
        if self.num_extra_layers:
            enc2 = []
            for i in range(self.num_extra_layers):
                enc2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                enc2.append(('r'+str(i),nn.ReLU()))
            enc2 = nn.Sequential(OrderedDict(enc2))
            self.enc2 = enc2
        self.enc3 = nn.Linear(in_features=self.d*factor, out_features=self.num_latents*2)
        
        self.enc4 = nn.Linear(in_features=self.num_latents, out_features = self.d)
        
        self.B = self.A.copy()
        self.B[np.eye(self.d,dtype='bool')] = True
        
        # decoder 
        self.dec1 = nn.ModuleList()
        if self.num_extra_layers:
            self.dec2 = nn.ModuleList()
        self.dec3 = nn.ModuleList()
        self.noise1 = nn.ModuleList()
        self.noise2 = nn.ModuleList()
        
        for i in range(self.d):
            num_incident = self.B[:,i].sum()
            self.dec1.append(nn.Linear(in_features=num_incident, out_features=self.d*factor))
            dec2 = []
            for i in range(self.num_extra_layers):
                dec2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                dec2.append(('r'+str(i),nn.ReLU()))
            dec2 = nn.Sequential(OrderedDict(dec2))
            self.dec2.append(dec2)
            self.dec3.append(nn.Linear(in_features=self.d*factor, out_features=1))
            if ~no_noise:
                self.noise1.append(nn.Linear(in_features=1, out_features = noise_features))
                self.noise2.append(nn.Linear(in_features=noise_features, out_features=1))

    # def add_noise(X, sigma=1/5):
    #     "Take input and add some noise"
    #     noise_term = torch.randn_like(X)
    #     return X + noise_term
        
        
    def reparameterize(self, mu, log_var):
        """
        :mu: mean from the encoder's latent space
        :log_var: log variance from the encoder's latent space
        """
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        return q.rsample()
    
        
        
    def forward(self, xor, m):
        # encoding
        shape_x = xor.shape
        for loop in range(1):
            x = F.relu(self.enc1(xor))
            if self.num_extra_layers:
                x = self.enc2(x)
            x = self.enc3(x)
            #eps_y = x[:,:self.d*2].view(-1, 2, self.d)
            z = x.view(-1, 2, self.num_latents)
            
            #eps_z = x[:,self.d:self.d+self.d]
            # get `mu` and `log_var`
            mu = z[:, 0, :] # the first feature values as mean
            log_var = z[:, 1, :] # the other feature values as variance
            # get the latent vector through reparameterization
            eps_y = self.reparameterize(mu, log_var)
            
            # y_recon = self.enc4(z)
            # x = y_recon
            #eps_y = z#[:,:self.d]
            if ~self.no_noise:
                eps_z = z[:,self.d:]
            # decoding
            y_recon = eps_y
            x = torch.zeros(shape_x)
            #print(shape_x)
            for i in range(self.d):
                y = y_recon[:,self.B[:,i]]
                y = self.dec1[i](y)
                y = F.relu(y)
                if self.num_extra_layers>0:
                    y = self.dec2[i](y)
                y_recon[:,i] = self.dec3[i](y).squeeze()            
                
                if self.no_noise:
                    x[:,i] = y_recon[:,i]
                else:
                    z_i = F.relu(self.noise1[i](eps_z[:,i].unsqueeze(1)))
                    z_i = self.noise2[i](z_i).squeeze()
                    x[:,i] = y_recon[:,i] + z_i     
            
                #x[:,i] = m[:,i]*xor[:,i]+(1-m[:,i])*x[:,i]
            
        return x, mu, log_var, y_recon
    
   
    
   
    
class CausalVAE2(nn.Module):
    def __init__(self, d, adjacency, noise_features = 20, num_extra_layers = False, factor=1, no_noise=True):
        super(CausalVAE2, self).__init__()
        self.num_latents = d
        self.d = d
        self.A = ancestors(adjacency)
        self.no_noise = no_noise
        self.log_scale = nn.Parameter(torch.zeros(d))
        self.num_extra_layers = num_extra_layers
        # encoder
        
        self.enc1 = nn.Linear(in_features=self.d, out_features=self.d*factor)
        if self.num_extra_layers:
            enc2 = []
            for i in range(self.num_extra_layers):
                enc2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                enc2.append(('r'+str(i),nn.ReLU()))
            enc2 = nn.Sequential(OrderedDict(enc2))
            self.enc2 = enc2
        self.enc3 = nn.Linear(in_features=self.d*factor, out_features=self.num_latents*2)
        
        
        self.B = self.A.copy()
        self.B[np.eye(self.d,dtype='bool')] = True
        
        # decoder 
        self.dec1 = nn.ModuleList()
        if self.num_extra_layers:
            self.dec2 = nn.ModuleList()
        self.dec3 = nn.ModuleList()
        self.noise1 = []
        self.noise2 = []
        
        for i in range(self.d):
            num_incident = self.B[:,i].sum()
            print('incident', i, num_incident)
            self.dec1.append(nn.Linear(in_features=num_incident, out_features=self.d*factor))
            dec2 = []
            for i in range(self.num_extra_layers):
                dec2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                dec2.append(('r'+str(i),nn.ReLU()))
            dec2 = nn.Sequential(OrderedDict(dec2))
            self.dec2.append(dec2)
            self.dec3.append(nn.Linear(in_features=self.d*factor, out_features=1))
            if ~no_noise:
                self.noise1.append(nn.Linear(in_features=1, out_features = noise_features))
                self.noise2.append(nn.Linear(in_features=noise_features, out_features=1))
        
        
        
        
    def reparameterize(self, mu, log_var):
        """
        :mu: mean from the encoder's latent space
        :log_var: log variance from the encoder's latent space
        """
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        return q.rsample()
    
        
        
    def forward(self, x, m = None):
        # encoding
        shape_x = x.shape
        for loop in range(1):
            x = F.relu(self.enc1(x))
            if self.num_extra_layers>0:
                x = self.enc2(x)
            x = self.enc3(x)
            #eps_y = x[:,:self.d*2].view(-1, 2, self.d)
            z = x.view(-1, 2, self.num_latents)
            
            #eps_z = x[:,self.d:self.d+self.d]
            # get `mu` and `log_var`
            mu = z[:, 0, :] # the first feature values as mean
            log_var = z[:, 1, :] # the other feature values as variance
            # get the latent vector through reparameterization
            eps_y = self.reparameterize(mu, log_var)
            
            # y_recon = self.enc4(z)
            # x = y_recon
            #eps_y = z#[:,:self.d]
            if ~self.no_noise:
                eps_z = z[:,self.d:]
            # decoding
            y_recon = torch.zeros_like(x)#eps_y
            x = torch.zeros(shape_x)
            #print(shape_x)
            for i in range(self.d):
                y = eps_y[:,self.B[:,i]]
                y = self.dec1[i](y)
                y = F.relu(y)
                if self.num_extra_layers>0:
                    y = self.dec2[i](y)
                y_recon[:,i] = self.dec3[i](y).squeeze()            
                
                if self.no_noise:
                    x[:,i] = y_recon[:,i]
                else:
                    z_i = F.relu(self.noise1[i](eps_z[:,i].unsqueeze(1)))
                    z_i = self.noise2[i](z_i).squeeze()
                    x[:,i] = y_recon[:,i] + z_i     
                
        return x, mu, log_var, y_recon
    
    
    
class CausalVAE3(nn.Module):
    def __init__(self, d, adjacency, noise_features = 20, num_extra_layers = True, factor=1, no_noise=True):
        super(CausalVAE3, self).__init__()
        self.num_latents = d
        self.d = d
        self.A = directed2moralised(adjacency)
        self.no_noise = no_noise
        self.log_scale = nn.Parameter(torch.zeros(d))
        self.num_extra_layers = num_extra_layers
        # encoder

        self.enc1 = nn.Linear(in_features=self.d, out_features=self.d*factor)
        if self.num_extra_layers:
            enc2 = []
            for i in range(self.num_extra_layers):
                enc2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                enc2.append(('r'+str(i),nn.ReLU()))
            enc2 = nn.Sequential(OrderedDict(enc2))
            self.enc2 = enc2
        self.enc3 = nn.Linear(in_features=self.d*factor, out_features=self.num_latents*2)
        
        self.enc4 = nn.Linear(in_features=self.num_latents, out_features = self.d)
        
        self.B = self.A.copy()
        self.B[np.eye(self.d,dtype='bool')] = True
        
        # decoder 
        self.dec1 = nn.ModuleList()
        if self.num_extra_layers:
            self.dec2 = nn.ModuleList()
        self.dec3 = nn.ModuleList()
        self.noise1 = nn.ModuleList()
        self.noise2 = nn.ModuleList()
        
        for i in range(self.d):
            num_incident = self.B[:,i].sum()
            self.dec1.append(nn.Linear(in_features=num_incident, out_features=self.d*factor))
            dec2 = []
            for i in range(self.num_extra_layers):
                dec2.append(('e'+str(i),nn.Linear(in_features=self.d*factor, out_features=self.d*factor)))
                dec2.append(('r'+str(i),nn.ReLU()))
            dec2 = nn.Sequential(OrderedDict(dec2))
            self.dec2.append(dec2)
            self.dec3.append(nn.Linear(in_features=self.d*factor, out_features=1))
            if ~no_noise:
                self.noise1.append(nn.Linear(in_features=1, out_features = noise_features))
                self.noise2.append(nn.Linear(in_features=noise_features, out_features=1))

    # def add_noise(X, sigma=1/5):
    #     "Take input and add some noise"
    #     noise_term = torch.randn_like(X)
    #     return X + noise_term
        
        
    def reparameterize(self, mu, log_var):
        """
        :mu: mean from the encoder's latent space
        :log_var: log variance from the encoder's latent space
        """
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        return q.rsample()
    
        
        
    def forward(self, xor, m):
        # encoding
        shape_x = xor.shape
        for loop in range(1):
            x = F.relu(self.enc1(xor))
            if self.num_extra_layers:
                x = self.enc2(x)
            x = self.enc3(x)
            #eps_y = x[:,:self.d*2].view(-1, 2, self.d)
            z = x.view(-1, 2, self.num_latents)
            
            #eps_z = x[:,self.d:self.d+self.d]
            # get `mu` and `log_var`
            mu = z[:, 0, :] # the first feature values as mean
            log_var = z[:, 1, :] # the other feature values as variance
            # get the latent vector through reparameterization
            eps_y = self.reparameterize(mu, log_var)
            
            # y_recon = self.enc4(z)
            # x = y_recon
            #eps_y = z#[:,:self.d]
            if ~self.no_noise:
                eps_z = z[:,self.d:]
            # decoding
            y_recon = eps_y
            x = torch.zeros(shape_x)
            #print(shape_x)
            for i in range(self.d):
                y = y_recon[:,self.B[:,i]]
                y = self.dec1[i](y)
                y = F.relu(y)
                if self.num_extra_layers>0:
                    y = self.dec2[i](y)
                y_recon[:,i] = self.dec3[i](y).squeeze()            
                
                if self.no_noise:
                    x[:,i] = y_recon[:,i]
                else:
                    z_i = F.relu(self.noise1[i](eps_z[:,i].unsqueeze(1)))
                    z_i = self.noise2[i](z_i).squeeze()
                    x[:,i] = y_recon[:,i] + z_i     
            
                #x[:,i] = m[:,i]*xor[:,i]+(1-m[:,i])*x[:,i]
            
        return x, mu, log_var, y_recon
    
   
