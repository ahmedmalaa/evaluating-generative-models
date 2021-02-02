# pylint: disable=redefined-outer-name

import os
import time
import argparse
import pickle
import copy

from numpy import arange, random, ceil, mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from .utilize import (  # pylint: disable=unused-import
    data_readf, c2b, c2bcolwise, splitbycol, gene_check, statistics, dwp, load_MIMICIII, fig_add_noise
)

from .impl import data
from .impl import fc as model


class DPGAN(object):
    def __init__(
        self, 
        dp_enabled,
        g_net, 
        d_net, 
        ae_net, 
        z_sampler, 
        decompressDims, 
        aeActivation, 
        dataType, 
        batchSize, 
        cilpc, 
        n_discriminator_update, 
        learning_rate, 
        sigma,
        cg,
        trainX,
        testX
    ):
        
        self.dp_enabled = dp_enabled
        self.g_net = g_net
        self.d_net = d_net
        self.ae_net = ae_net
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.inputDim
        self.z_dim = self.g_net.randomDim
        self.decompressDims = decompressDims
        self.aeActivation = aeActivation
        self.dataType = dataType
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.keep_prob = tf.placeholder('float')
        self.batchSize = batchSize
        self.cilpc = cilpc
        self.n_discriminator_update = n_discriminator_update
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.cg = cg

        self.trainX = trainX
        self.testX = testX

        self.loss_ae, self.decodeVariables = self.ae_net(self.x) # AE, autoencoder
        self.x_ = self.g_net(self.z) # G, get generated data
        self.d_loss, self.g_loss, self.y_hat_real, self.y_hat_fake, _ = self.d_net(self.x, self.x_, self.keep_prob, self.decodeVariables, reuse=False) # D, in the beginning, no reuse

        self.nBatches = int(ceil(float(self.trainX.shape[0]) / float(self.batchSize))) # number of batch if using training set

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(2.5e-5),weights_list=[var for var in tf.global_variables() if 'W' in var.name])

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): # WGAN version, with medGAN consideration
            self.optimize_ae = tf.train.AdamOptimizer().minimize(self.loss_ae + sum(all_regs), var_list=self.ae_net.vars)
            if self.dp_enabled:
                # DP case
                self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)  # DP case
                grads_and_vars = self.d_rmsprop.compute_gradients(self.d_loss + self.reg, var_list=self.d_net.vars)
                dp_grads_and_vars = []  # noisy version
                for gv in grads_and_vars:  # for each pair
                    g = gv[0]  # get the gradient, type in loop one: Tensor("gradients/AddN_37:0", shape=(4, 4, 1, 64), dtype=float32)
                    # print(g) # shape of all vars
                    if g is not None:  # skip None case
                        g = self.dpnoise(g, self.batchSize)  # add noise on the tensor, type in loop one: Tensor("Add:0", shape=(4, 4, 1, 64), dtype=float32)
                    dp_grads_and_vars.append((g, gv[1]))
                self.d_rmsprop_new = self.d_rmsprop.apply_gradients(dp_grads_and_vars) # should assign to a new optimizer
            else:
                # non-DP case
                self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) \
                    .minimize(self.d_loss + self.reg, var_list=self.d_net.vars)
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate) \
                .minimize(self.g_loss + self.reg, var_list=self.g_net.vars+list(self.decodeVariables.values()))

        self.d_clip = [v.assign(tf.clip_by_value(v, -1*self.cilpc,  self.cilpc)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.initialize_all_variables())
        self.g_loss_store = []  # store loss of generator
        self.d_loss_store = []  # store loss of discriminator
        self.wdis_store = []  # store Wasserstein distance, new added

    def train_autoencoder(self, pretrainEpochs, pretrainBatchSize):
        '''Pre-training autoencoder'''
        use_test_set = not (self.testX is None)
        nTrainBatches = int(ceil(float(self.trainX.shape[0])) / float(pretrainBatchSize))
        if use_test_set:
            nTestBatches = int(ceil(float(self.testX.shape[0])) / float(pretrainBatchSize))
        for epoch in range(pretrainEpochs):
            idx = random.permutation(self.trainX.shape[0]) # shuffle training data in each epoch
            trainLossVec = []
            for i in range(nTrainBatches):
                batchX = self.trainX[idx[i * pretrainBatchSize:(i + 1) * pretrainBatchSize]]
                randomZ = self.z_sampler(pretrainBatchSize, self.z_dim)
                _, loss = self.sess.run([self.optimize_ae, self.loss_ae], feed_dict={self.x: batchX, self.z: randomZ})
                trainLossVec.append(loss)
            if use_test_set:
                idx = random.permutation(self.testX.shape[0])
                testLossVec = []
                for i in range(nTestBatches):
                    batchX = self.testX[idx[i * pretrainBatchSize:(i + 1) * pretrainBatchSize]]
                    loss = self.sess.run(self.loss_ae, feed_dict={self.x: batchX})
                    testLossVec.append(loss)
            print('Pretrain_Epoch:%d, trainLoss:%f' % (epoch, mean(trainLossVec)))
            if use_test_set:
                print('Pretrain_Epoch:%d, testLoss:%f' % (epoch, mean(testLossVec)))

    def train(self, nEpochs, batchSize):
        start_time = time.time()
        idx = arange(self.trainX.shape[0])
        for epoch in range(nEpochs):
            for i in range(self.nBatches):
                rd_loss = 0
                rg_loss = 0
                for _ in range(self.n_discriminator_update): # train discriminator
                    batchIdx = random.choice(idx, size=batchSize, replace=False)
                    batchX = self.trainX[batchIdx]
                    randomZ = self.z_sampler(batchSize, self.z_dim)
                    if self.dp_enabled:
                        # DP case
                        _, rd_loss = self.sess.run([self.d_rmsprop_new, self.d_loss], feed_dict={self.x: batchX, self.z: randomZ, self.keep_prob: 1.0})
                    else:
                        # non-DP case
                        _, rd_loss = self.sess.run([self.d_rmsprop, self.d_loss], feed_dict={self.x: batchX, self.z: randomZ, self.keep_prob: 1.0})
                    self.sess.run(self.d_clip)

                randomZ = self.z_sampler(batchSize, self.z_dim) # train generator
                _, rg_loss = self.sess.run([self.g_rmsprop, self.g_loss], feed_dict={self.x: batchX, self.z: randomZ, self.keep_prob: 1.0})

                if i % 50 == 0: # print out loss
                    print('Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                          (time.time() - start_time, rd_loss, rg_loss))

                # store rd_loss and rg_loss, new added
                self.g_loss_store.append(rg_loss)  # g_loss will decrease, here is not self.g_loss nor self.g_loss_reg
                self.d_loss_store.append(rd_loss)  # d_loss will increase
                self.wdis_store.append(-1*rd_loss)  # Wasserstein distance will decrease

        z_sample = self.z_sampler(self.trainX.shape[0], self.z_dim) # generate EHR from generator, after finish training
        x_gene = self.sess.run(self.x_, feed_dict={self.z: z_sample})
        dec = self.decoder(x_gene)
        x_gene_dec = self.sess.run(dec) # generated data
        
        # NOTE: This is not relevant to our case:
        # x_gene_dec = c2bcolwise(self.trainX, x_gene_dec, self.adj) # binarize generated data by setting the same portion of elements to 1 as the training set, these elements have highest original value
        
        print("please check this part, make sure it is correct")
        testX_text = f"self.testX.shape: {self.testX.shape}." if self.testX is not None else "self.testX is None."
        print(f"self.trainX.shape: {self.trainX.shape}, x_gene.shape: {x_gene.shape}, x_gene_dec.shape: {x_gene_dec.shape}, {testX_text}")
        
        return self.trainX, x_gene_dec

    def decoder(self, x_fake): # this function is specifically to make sure the output of generator goes through the decoder
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_' + str(i)]), self.decodeVariables['aed_b_' + str(i)]))
            i += 1
        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_' + str(i)]), self.decodeVariables['aed_b_' + str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, self.decodeVariables['aed_W_' + str(i)]), self.decodeVariables['aed_b_' + str(i)]))
        return x_decoded

    def dpnoise(self, tensor, batchSize):
        '''add noise to tensor'''
        s = tensor.get_shape().as_list()  # get shape of the tensor
        rt = tf.random_normal(s, mean=0.0, stddev=self.sigma * self.cg)
        t = tf.add(tensor, tf.scalar_mul((1.0 / batchSize), rt))
        return t

    def loss_store2(self, x_train, x_gene, db):
        with open('./result/genefinalfig/x_train.pickle', 'wb') as fp:
            pickle.dump(x_train, fp)
        with open('./result/genefinalfig/generated.pickle', 'wb') as fp:
            pickle.dump(x_gene, fp)
        bins = 100
        plt.hist(x_gene, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of distribution of generated data')
        plt.xlabel('Generated data value')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/WGAN-Generated-data-distribution.jpg')
        plt.close()
        with open('./result/lossfig/wdis.pickle', 'wb') as fp:
            pickle.dump(self.wdis_store, fp)
        t = arange(len(self.wdis_store))
        plt.plot(t, self.wdis_store, 'r--')
        plt.xlabel('Iterations')
        plt.ylabel('Wasserstein distance')
        plt.savefig('./result/lossfig/WGAN-W-distance.jpg')
        plt.close()
        rv_pre, gv_pre, rv_pro, gv_pro = dwp(x_train, x_gene, self.testX, db)
        print('Totally ' + str(len(rv_pre)) + ' of coordinates are left')
        with open('./result/genefinalfig/rv_pre.pickle', 'wb') as fp:
            pickle.dump(rv_pre, fp)
        with open('./result/genefinalfig/gv_pre.pickle', 'wb') as fp:
            pickle.dump(gv_pre, fp)
        with open('./result/genefinalfig/rv_pro.pickle', 'wb') as fp:
            pickle.dump(rv_pro, fp)
        with open('./result/genefinalfig/gv_pro.pickle', 'wb') as fp:
            pickle.dump(gv_pro, fp)
        rv_pre, gv_pre, rv_pro, gv_pro = fig_add_noise(rv_pre), fig_add_noise(gv_pre), fig_add_noise(rv_pro), fig_add_noise(gv_pro)
        plt.scatter(rv_pre, gv_pre)
        plt.title('Dimension-wise prediction, lr')
        plt.xlabel('Real data')
        plt.ylabel('Generated data')
        plt.savefig('./result/genefinalfig/WGAN-dim-wise-prediction.jpg')
        plt.close()
        plt.scatter(rv_pro, gv_pro)
        plt.title('Dimension-wise probability, lr')
        plt.xlabel('Real data')
        plt.ylabel('Generated data')
        plt.savefig('./result/genefinalfig/WGAN-dim-wise-probability.jpg')
        plt.close()

    def loss_store(self, x_train, x_gene, validation_ratio):
        '''store everything new added'''
        with open('./result/genefinalfig/generated.pickle', 'wb') as fp:
            pickle.dump(x_gene, fp)
        bins = 100
        plt.hist(x_gene, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of distribution of generated data')
        plt.xlabel('Generated data value')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/generated_value.jpg')
        plt.close()
        with open('./result/lossfig/wdis.pickle', 'wb') as fp:
            pickle.dump(self.wdis_store, fp)
        t = arange(len(self.wdis_store))
        plt.plot(t, self.wdis_store, 'r--')
        plt.xlabel('Iterations')
        plt.ylabel('Wasserstein distance')
        plt.savefig('./result/lossfig/wdis.jpg')
        plt.close()
        precision_r_all = []
        precision_g_all = []
        recall_r_all = []
        recall_g_all = []
        acc_r_all = []
        acc_g_all = []
        f1score_r_all = []
        f1score_g_all = []
        auc_r_all = []
        auc_g_all = []
        MIMIC_data, dim_data = x_train, len(x_train[0])
        for col in range(dim_data):
            print(col)
            trainX, testX = splitbycol(self.dataType, validation_ratio, col, MIMIC_data)
            if trainX == []:
                print("skip this coordinate")
                continue
            geneX = gene_check(col, x_gene) # process generated data by column
            if geneX == []:
                print("skip this coordinate")
                continue
            precision_r, precision_g, recall_r, recall_g, acc_r, acc_g, f1score_r, f1score_g, auc_r, auc_g = statistics(trainX, geneX, testX, col)
            if precision_r == []:
                print("skip this coordinate")
                continue
            precision_r_all.append(precision_r)
            precision_g_all.append(precision_g)
            recall_r_all.append(recall_r)
            recall_g_all.append(recall_g)
            acc_r_all.append(acc_r)
            acc_g_all.append(acc_g)
            f1score_r_all.append(f1score_r)
            f1score_g_all.append(f1score_g)
            auc_r_all.append(auc_r)
            auc_g_all.append(auc_g)
        plt.hist(precision_r_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of precision on each dimension of training data, lr')
        plt.xlabel('Precision (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_precision_r.jpg')
        plt.close()
        plt.hist(precision_g_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of precision on each dimension of generated data, lr')
        plt.xlabel('Precision (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_precision_g.jpg')
        plt.close()
        plt.hist(recall_r_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of recall on each dimension of training data, lr')
        plt.xlabel('Recall (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_recall_r.jpg')
        plt.close()
        plt.hist(recall_g_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of recall on each dimension of generated data, lr')
        plt.xlabel('Recall (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_recall_g.jpg')
        plt.close()
        plt.hist(acc_r_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of accuracy on each dimension of training data, lr')
        plt.xlabel('Accuracy (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_acc_r.jpg')
        plt.close()
        plt.hist(acc_g_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of accuracy on each dimension of generated data, lr')
        plt.xlabel('Accuracy (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_acc_g.jpg')
        plt.close()
        plt.hist(f1score_r_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of f1score on each dimension of training data, lr')
        plt.xlabel('f1score (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_f1score_r.jpg')
        plt.close()
        plt.hist(f1score_g_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of f1score on each dimension of generated data, lr')
        plt.xlabel('f1score (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_f1score_g.jpg')
        plt.close()
        plt.hist(auc_r_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of AUC on each dimension of training data, lr')
        plt.xlabel('AUC (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_AUC_r.jpg')
        plt.close()
        plt.hist(auc_g_all, bins, facecolor='red', alpha=0.5)
        plt.title('Histogram of AUC on each dimension of generated data, lr')
        plt.xlabel('AUC (total number: ' + str(len(precision_r_all)) + ' )')
        plt.ylabel('Frequency')
        plt.savefig('./result/genefinalfig/hist_AUC_g.jpg')
        plt.close()


DEFAULT_PARAMS = {
    "dp_enabled": True,
    "dataType": 'real',
    "embeddingDim": 128,
    "randomDim": 128,
    "generatorDims": list((128, 128)),  # NOTE: + [embeddingDim] in _process_params()
    "discriminatorDims": (256, 128, 1),
    "compressDims": list(),  # NOTE:  + [embeddingDim] in _process_params()
    "decompressDims": list(),  # NOTE: + [inputDim] in _process_params()
    "l2scale": 2.5e-5,
    "pretrainEpochs": 100, 
    "pretrainBatchSize": 128,
    "nEpochs": 1000, 
    "batchSize": 1024,
    "cilpc": 0.01,
    "n_discriminator_update": 2,
    "learning_rate": 5e-4, 
    "sigma": 1.6,  # NOTE: Used to regulate DP noise level.
    "cg": 1.0,  # Difficult to estimate, set cg to 1 and vary sigma for our purposes.
}

def _process_params(params, input_dim):
    params["inputDim"] = input_dim
    params["generatorDims"] = params["generatorDims"] + [params["embeddingDim"]]
    params["compressDims"] = params["compressDims"] + [params["embeddingDim"]]
    params["decompressDims"] = params["decompressDims"] + [params["inputDim"]]
    return params


def dpgan(orig_data, customized_params):
    """
    Execute dpgan training and synthetic data generation.

    Args:
        orig_data (np.ndarray): Original data array.
        customized_params (dict): A dict with customized parameters for DPGAN.
            Required key-value: "inputDim", the input dimension number of the data.
            Optional key-values: see the defaults in dpgan.py --> DEFAULT_PARAMS. 
    """
    # Parameters:
    assert "inputDim" in tuple(customized_params.keys())
    params = copy.deepcopy(DEFAULT_PARAMS)
    for k, v in customized_params.items():
        assert k not in ("generatorDims", "compressDims", "decompressDims")
        params[k] = v
    params = _process_params(params=params, input_dim=customized_params["inputDim"])

    # Activations:
    if params["dataType"] == 'binary':
        aeActivation = tf.nn.tanh
    else:
        aeActivation = tf.nn.relu
    generatorActivation = tf.nn.relu
    discriminatorActivation = tf.nn.relu

    # Train and generate data:
    zs = data.NoiseSampler()
    ae_net = model.Autoencoder(params["inputDim"], params["l2scale"], params["compressDims"], aeActivation, params["decompressDims"], params["dataType"])
    g_net = model.Generator(params["randomDim"], params["l2scale"], params["generatorDims"], generatorActivation, params["dataType"])
    d_net = model.buildDiscriminator(params["inputDim"], params["discriminatorDims"], discriminatorActivation, params["decompressDims"], aeActivation, params["dataType"], params["l2scale"])
    wgan = DPGAN(
        params["dp_enabled"], g_net, d_net, ae_net, zs, params["decompressDims"], aeActivation, params["dataType"], params["batchSize"], 
        params["cilpc"], params["n_discriminator_update"], params["learning_rate"], params["sigma"], params["cg"], orig_data, testX=None)
    wgan.train_autoencoder(params["pretrainEpochs"], params["pretrainBatchSize"])
    x_train, x_gene = wgan.train(params["nEpochs"], params["batchSize"])

    return x_gene


# NOTE: Original main code flow:
if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='MIMIC-III')
    parser.add_argument('--model', type=str, default='fc')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # Parameters
    dp_enabled = False
    dataType = 'binary'
    inputDim = 1071 # 1071 for original data, other: 1071 (in paper), 512, 64
    embeddingDim = 128
    randomDim = 128
    generatorDims = list((128, 128)) + [embeddingDim]
    discriminatorDims = (256, 128, 1)
    compressDims = list(()) + [embeddingDim]
    decompressDims = list(()) + [inputDim]
    # bnDecay = 0.99
    l2scale = 2.5e-5 # WGAN: 2.5e-5, GAN: 0.001
    pretrainEpochs = 100 #2, 100
    pretrainBatchSize = 128
    nEpochs = 1000 #2, 1000
    batchSize = 1024
    cilpc = 0.01
    n_discriminator_update = 2
    learning_rate = 5e-4 # GAN: 0.001
    # adj = 1.0
    db = 0.5
    sigma = 6000.0  # assign it manually
    cg = 1.0
    # bn_train = True
    validation_ratio = 0.25
    top = 1071 # 1071 for original data, other: 1071 (in paper), 512, 64
    
    if dataType == 'binary':
        aeActivation = tf.nn.tanh
    else:
        aeActivation = tf.nn.relu
    
    generatorActivation = tf.nn.relu
    discriminatorActivation = tf.nn.relu

    trainX, testX, _ = load_MIMICIII(dataType, validation_ratio, top) # load whole dataset, self.top is dummy here

    zs = data.NoiseSampler()
    ae_net = model.Autoencoder(inputDim, l2scale, compressDims, aeActivation, decompressDims, dataType)
    g_net = model.Generator(randomDim, l2scale, generatorDims, generatorActivation, dataType)
    d_net = model.buildDiscriminator(inputDim, discriminatorDims, discriminatorActivation, decompressDims, aeActivation, dataType, l2scale)
    wgan = DPGAN(
        dp_enabled, g_net, d_net, ae_net, zs, decompressDims, aeActivation, dataType, batchSize, 
        cilpc, n_discriminator_update, learning_rate, sigma, cg, trainX, testX)
    wgan.train_autoencoder(pretrainEpochs, pretrainBatchSize) # Pre-training autoencoder
    x_train, x_gene = wgan.train(nEpochs, batchSize)
    wgan.loss_store2(x_train, x_gene, db)
