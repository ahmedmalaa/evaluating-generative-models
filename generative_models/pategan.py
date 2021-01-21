'''PATE-GAN function'''

# Necessary packages
import tensorflow as tf
import numpy as np
import warnings
#warnings.filterwarnings("ignore")

tf.compat.v1.disable_eager_execution()

from sklearn.linear_model import LogisticRegression


def pate_lambda (x, teacher_models, lambda_):
    '''Returns PATE_lambda(x).
    
    Args:
        - x: feature vector
        - teacher_models: a list of teacher models
        - lambda_: parameter
        
    Returns:
        - n0, n1: the number of label 0 and 1, respectively
        - out: label after adding laplace noise.
  '''
      
    y_hat = list()
                
    for teacher in teacher_models:                        
        temp_y = teacher.predict(np.reshape(x, [1,-1]))
        y_hat = y_hat + [temp_y]
    
    y_hat = np.asarray(y_hat)
    n0 = sum(y_hat == 0)
    n1 = sum(y_hat == 1)
    
    lap_noise = np.random.laplace(loc=0.0, scale=lambda_)
    
    out = (n1+lap_noise) / float(n0+n1)
    out = int(out>0.5)
                
    return n0, n1, out 


def pategan(x_train, parameters):
    '''Basic PATE-GAN framework.
    
    Args:
        - x_train: training data
        - parameters: PATE-GAN parameters
            - n_s: the number of student training iterations
            - batch_size: the number of batch size for training student and generator
            - k: the number of teachers
            - epsilon, delta: Differential privacy parameters
            - lambda_: noise size
            
    Returns:
        - x_train_hat: generated training data by differentially private generator
    '''
    
    # Reset the graph
    tf.compat.v1.reset_default_graph()
        
    # PATE-GAN parameters
    # number of student training iterations
    n_s = parameters['n_s']
    # number of batch size for student and generator training
    batch_size = parameters['batch_size']
    # number of teachers
    k = parameters['k']
    # epsilon
    epsilon = parameters['epsilon']
    # delta
    delta = parameters['delta']
    # lambda_
    lambda_ = parameters['lambda']
    
    # Other parameters
    # alpha initialize
    L = 20
    alpha = np.zeros([L])
    # initialize epsilon_hat
    epsilon_hat = 0
        
    # Network parameters
    no, dim = x_train.shape
    # Random sample dimensions
    z_dim = int(dim)
    # Student hidden dimension
    student_h_dim = int(dim)
    # Generator hidden dimension
    generator_h_dim = int(4*dim)    
    
    ## Partitioning the data into k subsets
    x_partition = list()
    partition_data_no = int(no/k)
        
    idx = np.random.permutation(no)
        
    for i in range(k):
        temp_idx = idx[int(i*partition_data_no):int((i+1)*partition_data_no)]
        temp_x = x_train[temp_idx, :]            
        x_partition = x_partition + [temp_x]        
    
    ## Necessary Functions for buidling NN models
    # Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape = size, stddev = xavier_stddev)        
                
    # Sample from uniform distribution
    def sample_Z(m, n):
        return np.random.uniform(0., 1., size = [m, n])
         
    ## Placeholder
    # PATE labels
    Y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])    
    # Random Variable        
    Z = tf.compat.v1.placeholder(tf.float32, shape = [None, z_dim])
     
    ## NN variables     
    # Student
    S_W1 = tf.Variable(xavier_init([dim, student_h_dim]))
    S_b1 = tf.Variable(tf.zeros(shape=[student_h_dim]))
        
    S_W2 = tf.Variable(xavier_init([student_h_dim,1]))
    S_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_S = [S_W1, S_W2, S_b1, S_b2]
        
    # Generator

    G_W1 = tf.Variable(xavier_init([z_dim, generator_h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

    G_W2 = tf.Variable(xavier_init([generator_h_dim,generator_h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[generator_h_dim]))

    G_W3 = tf.Variable(xavier_init([generator_h_dim,dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
        
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## Models
    def generator(z):
        G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        G_out = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
                
        return G_out
        
    def student(x):
        S_h1 = tf.nn.relu(tf.matmul(x, S_W1) + S_b1)
        S_out = tf.matmul(S_h1, S_W2) + S_b2
                
        return S_out
            
    ## Loss    
    G_sample = generator(Z)
    S_fake = student(G_sample)
    
    S_loss = tf.reduce_mean(input_tensor=Y * S_fake) - tf.reduce_mean(input_tensor=(1-Y) * S_fake)
    G_loss = -tf.reduce_mean(input_tensor=S_fake)
    
    # Optimizer
    S_solver = (tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
                            .minimize(-S_loss, var_list=theta_S))
    G_solver = (tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4)
                            .minimize(G_loss, var_list=theta_G))
    
    clip_S = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_S]
    
    ## Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    

    min_iterations = 1
    iteration=0
    ## Iterations
    while epsilon_hat < epsilon or iteration < min_iterations:            
        iteration+=1
            
        # 1. Train teacher models
        teacher_models = list()
        
        for _ in range(k):
                                
            Z_mb = sample_Z(partition_data_no, z_dim)
            G_mb = sess.run(G_sample, feed_dict = {Z: Z_mb})
                                
            temp_x = x_partition[i]
            idx = np.random.permutation(len(temp_x[:, 0]))
            X_mb = temp_x[idx[:partition_data_no], :]
                                
            X_comb = np.concatenate((X_mb, G_mb), axis = 0)
            Y_comb = np.concatenate((np.ones([partition_data_no,]), 
                                                             np.zeros([partition_data_no,])), axis = 0)
                                
            model = LogisticRegression()
            model.fit(X_comb, Y_comb)
            teacher_models = teacher_models + [model]
                        
        # 2. Student training
        for _ in range(n_s):
                    
            Z_mb = sample_Z(batch_size, z_dim)
            G_mb = sess.run(G_sample, feed_dict = {Z: Z_mb})
            Y_mb = list()
                        
            for j in range(batch_size):                                
                n0, n1, r_j = pate_lambda(G_mb[j, :], teacher_models, lambda_)
                Y_mb = Y_mb + [r_j]
             
                # Update moments accountant
                q = np.log(2 + lambda_ * abs(n0 - n1)) - np.log(4.0) - \
                        (lambda_ * abs(n0 - n1))
                q = np.exp(q)
                                
                # Compute alpha
                for l in range(L):
                    temp1 = 2 * (lambda_**2) * (l+1) * (l+2)
                    temp2 = (1-q) * ( ((1-q)/(1-q*np.exp(2*lambda_)))**(l+1) ) + \
                                    q * np.exp(2*lambda_ * (l+1))
                    alpha[l] = alpha[l] + np.min([temp1, np.log(temp2)])
                
            # PATE labels for G_mb    
            Y_mb = np.reshape(np.asarray(Y_mb), [-1,1])
                                
            # Update student
            _, D_loss_curr, _ = sess.run([S_solver, S_loss, clip_S], 
                                                                     feed_dict = {Z: Z_mb, Y: Y_mb})
        
        # Generator Update                
        Z_mb = sample_Z(batch_size, z_dim)
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {Z: Z_mb})
        print('G loss',G_loss_curr)
        print('D_loss', D_loss_curr)
        print(np.mean(Y_mb))
                
        # epsilon_hat computation
        curr_list = list()                
        for l in range(L):
            temp_alpha = (alpha[l] + np.log(1/delta)) / float(l+1)
            curr_list = curr_list + [temp_alpha]
                
        epsilon_hat = np.min(curr_list)
        print(epsilon_hat)                

    ## Outputs
    x_train_hat = sess.run([G_sample], feed_dict = {Z: sample_Z(no, z_dim)})[0]

    for i in range(dim):
        if len(np.unique(x_train[:, i])) == 2:
            x_train_hat[:, i] = np.round(x_train_hat[:, i])
            
    return x_train_hat


## Main
if __name__ == '__main__':
    
    x_train = np.random.normal(0, 1, [10000,5])
        
    # Normalization
    for i in range(len(x_train[0, :])):
        x_train[:, i] = x_train[:, i] - np.min(x_train[:, i])
        x_train[:, i] = x_train[:, i] / (np.max(x_train[:, i]) + 1e-8)
    
    
    parameters = {'n_s': 1, 'batch_size': 1000, 
                                'k': 100, 'epsilon': 100, 'delta': 0.0001, 'lambda': 1}

    x_train_new = pategan(x_train, parameters)
