"""
Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar, 
"Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
A harmonizing advancement for AI in medicine," 
IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
Paper link: https://ieeexplore.ieee.org/document/9034117
Last updated Date: December 22th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

Minor modifications made by Boris van Breugel (bv292@cam.ac.uk) and Evgeny Saveliev (e.s.saveliev@gmail.com).
-----------------------------
adsgan.py
- Generate synthetic data for GAN framework
(1) Use original data to generate synthetic data
"""

#%% Import necessary packages
import tensorflow as tf
import numpy as np

from tqdm import tqdm

tf.compat.v1.disable_eager_execution()

def gan(orig_data, params):
    """Generate synthetic data for ADSGAN framework.
    
    Args:
        orig_data: original data
        params: Network parameters
            mb_size: mini-batch size
            z_dim: random state dimension
            h_dim: hidden state dimension
            lambda: identifiability parameter
            iterations: training iterations
            
    Returns:
        synth_data: synthetically generated data
    """
        
    # Reset the tensorflow graph
    tf.compat.v1.reset_default_graph()
    
    ## Parameters        
    # Feature no
    x_dim = len(orig_data.columns)        
    # Sample no
    no = len(orig_data)        
    
    # Batch size        
    mb_size = params['mb_size']
    # Random variable dimension
    z_dim = params['z_dim'] 
    # Hidden unit dimensions
    h_dim = params['h_dim']        
    # Training iterations
    iterations = params['iterations']
    # GAN type
    gen_model_name = params['gen_model_name']
    # WGAN-GP parameters
    lam = 10
    lr = 1e-4        

    #%% Data Preprocessing
    orig_data = np.asarray(orig_data)

    def data_normalization(orig_data, epsilon = 1e-8):
                
        min_val = np.min(orig_data, axis=0)
        
        normalized_data = orig_data - min_val
        
        max_val = np.max(normalized_data, axis=0)
        normalized_data = normalized_data / (max_val + epsilon)
        
        normalization_params = {"min_val": min_val, "max_val": max_val}
        
        return normalized_data, normalization_params
    
    def data_renormalization(normalized_data, normalization_params, epsilon = 1e-8):
        
        renormalized_data = normalized_data * (normalization_params['max_val'] + epsilon)
        renormalized_data = renormalized_data + normalization_params['min_val']
        
        return renormalized_data
    
    orig_data, normalization_params = data_normalization(orig_data)
        
    #%% Necessary Functions

    # Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape = size, stddev = xavier_stddev)        
                
    # Sample from uniform distribution
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size = [m, n])
                
    # Sample from the real data
    def sample_X(m, n):
        return np.random.permutation(m)[:n]    
         
    #%% Placeholder
    # Feature
    X = tf.compat.v1.placeholder(tf.float32, shape = [None, x_dim])     
    # Random Variable        
    Z = tf.compat.v1.placeholder(tf.float32, shape = [None, z_dim])
            
    #%% Discriminator
    # Discriminator
    D_W1 = tf.Variable(xavier_init([x_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim,h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
        
    D_W3 = tf.Variable(xavier_init([h_dim,1]))
    D_b3 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        
    #%% Generator
    G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim,h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    G_W3 = tf.Variable(xavier_init([h_dim,h_dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W4 = tf.Variable(xavier_init([h_dim, x_dim]))
    G_b4 = tf.Variable(tf.zeros(shape=[x_dim]))
        
    theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]

    #%% Generator and discriminator functions
    def generator(z):
        G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        G_h3 = tf.nn.tanh(tf.matmul(G_h2, G_W3) + G_b3)
        G_log_prob = tf.nn.sigmoid(tf.matmul(G_h3, G_W4) + G_b4)
                
        return G_log_prob
        
    def discriminator(x):
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        out = (tf.matmul(D_h2, D_W3) + D_b3)
                
        return out
        
    #%% Structure
    G_sample = generator(Z)
    D_real = discriminator(X)
    D_fake = discriminator(G_sample) 
    
    if gen_model_name=='wgan':
            
        # Replacement of Clipping algorithm to Penalty term
        # 1. Line 6 in Algorithm 1
        eps = tf.random.uniform([mb_size, 1], minval = 0., maxval = 1.)
        X_inter = eps*X + (1. - eps) * G_sample
    
        # 2. Line 7 in Algorithm 1
        grad = tf.gradients(ys=discriminator(X_inter), xs=[X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(input_tensor=(grad)**2 + 1e-8, axis = 1))
        grad_pen = lam * tf.reduce_mean(input_tensor=(grad_norm - 1)**2)
    
        # Loss function
        D_loss = tf.reduce_mean(input_tensor=D_fake) - tf.reduce_mean(input_tensor=D_real) + grad_pen
    
    
    
    elif gen_model_name == 'gan':
        D_loss = tf.reduce_mean(input_tensor=D_fake) - tf.reduce_mean(input_tensor=D_real)
    
    G_loss = -tf.reduce_mean(input_tensor=D_fake)
    
    
    # Solver
    D_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(D_loss, var_list = theta_D))
    G_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(G_loss, var_list = theta_G))
                        
    #%% Iterations
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
                
    # Iterations
    for it in tqdm(range(iterations)):
        # Discriminator training
        for _ in range(5):        
            Z_mb = sample_Z(mb_size, z_dim)                        

            X_idx = sample_X(no,mb_size)                
            X_mb = orig_data[X_idx,:]    
                        
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, Z: Z_mb})
                        
        # Generator Training
        Z_mb = sample_Z(mb_size, z_dim)     
                
        X_idx = sample_X(no,mb_size)                
        X_mb = orig_data[X_idx,:]    
                                        
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {X: X_mb, Z: Z_mb})
        
    #%% Output Generation
    synth_data = sess.run([G_sample], feed_dict = {Z: sample_Z(no, z_dim)})
    synth_data = synth_data[0]
        
    # Renormalization
    synth_data = data_renormalization(synth_data, normalization_params)
    
    # Binary features
    for i in range(x_dim):
        if len(np.unique(orig_data[:, i])) == 2:
            synth_data[:, i] = np.round(synth_data[:, i])
     
    return synth_data