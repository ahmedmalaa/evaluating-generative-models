"""
Code author: Boris van Breugel (bv292@cam.ac.uk)

Based on code by Jinsung Yoon (jsyoon0823@gmail.com)
	
-----------------------------

Generate synthetic data with VAE framework
(1) Use original data to generate synthetic data
"""

#%% Import necessary packages
import tensorflow as tf
import numpy as np

from tqdm import tqdm


def vae(orig_data, params):
    """Generate synthetic data for VAE framework.
    
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
    # X_recon no
    no = len(orig_data)                
    
    # Batch size                
    mb_size = params['mb_size']
    # Latent representation dimension
    z_dim = params['z_dim']
    # Hidden unit dimensions
    h_dim = params['h_dim']                
    # Identifiability parameter
    
    # Training iterations
    iterations = params['iterations']
    # VAE type
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
                            
    # X_recon from uniform distribution
    def X_recon_Z(m, n):
        return np.random.randn(m, n)
                            
    # X_recon from the real data
    def X_recon_X(m, n):
        return np.random.permutation(m)[:n]  

    def sample_Z(m,n):
        return tf.random.normal((m,n), 0, 1, dtype=tf.float32)
             
    #%% Placeholder
    # Feature
    X = tf.compat.v1.placeholder(tf.float32, shape = [None, x_dim])         
    X_recon = tf.compat.v1.placeholder(tf.float32, shape = [None, x_dim])         
    # Random Variable                
    Z = tf.compat.v1.placeholder(tf.float32, shape = [None, z_dim])
    mu = tf.compat.v1.placeholder(tf.float32, shape = [None, z_dim])
    logvar = tf.compat.v1.placeholder(tf.float32, shape = [None, z_dim])
    
    
    #%% Encoder
    E_W1 = tf.Variable(xavier_init([x_dim, h_dim]))
    E_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    E_W2e = tf.Variable(xavier_init([h_dim, h_dim]))
    E_b2e = tf.Variable(tf.zeros(shape=[h_dim]))
    
    
    E_W_sigma = tf.Variable(xavier_init([h_dim,z_dim]))
    E_b_sigma = tf.Variable(tf.zeros(shape=[z_dim]))
    
    E_W_mu = tf.Variable(xavier_init([h_dim,z_dim]))
    E_b_mu = tf.Variable(tf.zeros(shape=[z_dim]))
    
    
    # Decoder
    
    
    D_W3 = tf.Variable(xavier_init([z_dim,h_dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    D_W2d = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2d = tf.Variable(tf.zeros(shape=[h_dim]))
    
    
    D_W4 = tf.Variable(xavier_init([h_dim, x_dim]))
    D_b4 = tf.Variable(tf.zeros(shape=[x_dim]))
        
    theta = [E_W1, E_W_sigma, E_W_mu, D_W3, D_W4, E_b1, 
               E_b_mu, E_b_sigma, D_b3, D_b4,
               E_W2e, E_b2e, D_W2d, D_b2d]
    
    #%% Generator and discriminator functions
    def encoder(x):
        E_h1 = tf.nn.tanh(tf.matmul(x, E_W1) + E_b1)
        E_h2 = tf.nn.tanh(tf.matmul(E_h1, E_W2e) + E_b2e)
        E_hmu = tf.nn.tanh(tf.matmul(E_h2, E_W_mu) + E_b_mu)
        E_hsigma = tf.matmul(E_h1, E_W_sigma) + E_b_sigma
        return E_hmu, E_hsigma
    
    def decoder(z):
        D_h3 = tf.nn.tanh(tf.matmul(z, D_W3) + D_b3)
        D_h4 = tf.nn.tanh(tf.matmul(D_h3, D_W2d) + D_b2d)
        x_recon = tf.nn.sigmoid(tf.matmul(D_h4, D_W4) + D_b4)
        return x_recon
            
        
        
    #%% Structure
    mu, logvar = encoder(X)
    Z = mu + tf.exp(logvar/2) * tf.random.normal(tf.shape(input=mu), 0, 1, dtype=tf.float32)
    
    X_recon = decoder(Z)
    
    
    
    
    loss1 = tf.reduce_mean(input_tensor=tf.square(X_recon-X))
    loss2 = 0.5 * tf.reduce_mean(input_tensor=tf.square(mu) + tf.exp(logvar) - logvar - 1, axis=1)
    
    loss = loss1 + loss2
    # Solver
    
    solver = (tf.compat.v1.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5).minimize(loss, var_list = theta))
                                            
    #%% Iterations
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
                            
    # Iterations
    for it in tqdm(range(iterations)):
        # Discriminator training
                    
        X_idx = X_recon_X(no,mb_size)
        X_mb = orig_data[X_idx,:]
                                                                        
        _, E_loss1_curr, E_loss2_curr = sess.run([solver, loss1, loss2], feed_dict = {X: X_mb})
            
    #%% Output Generation
    synth_data = sess.run([X_recon], feed_dict = {Z: np.random.randn(no, z_dim)})
    synth_data = synth_data[0]
    print(synth_data.shape)
            
    # Renormalization
    synth_data = data_renormalization(synth_data, normalization_params)
    
    # Binary features
    for i in range(x_dim):
        if len(np.unique(orig_data[:, i])) == 2:
            synth_data[:, i] = np.round(synth_data[:, i])
     
    return synth_data