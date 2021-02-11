# pylint: disable=unsubscriptable-object

import copy

import numpy as np
import tensorflow as tf
import json
from scipy.stats import mode

from . import data_utils
from . import plotting
from . import model
# from . import utils

from time import time
from .mmd import median_pairwise_distance, mix_rbf_mmd2_and_ratio

import warnings
warnings.simplefilter("ignore")

tf.logging.set_verbosity(tf.logging.ERROR)


DEFAULT_EXP_NAME = "MYEXP"
DEFAULT_PARAMS = {
    "custom_experiment": False,
    "settings_file": "",
    "data": DEFAULT_EXP_NAME,  # (EXPERIMENT_NAME, "sine")
    "num_samples": 14000,
    "seq_length": 30,
    "num_signals": 4,
    "normalise": False,
    "cond_dim": 0,
    "max_val": 1,
    "one_hot": False,
    "predict_labels": False,
    "scale": 0.1,
    "freq_low": 1.0,
    "freq_high": 5.0,
    "amplitude_low": 0.1,
    "amplitude_high": 0.9,
    "multivariate_mnist": False,
    "full_mnist": False,
    "data_load_from": "",
    "resample_rate_in_min": 15,
    "hidden_units_g": 100,
    "hidden_units_d": 100,
    "kappa": 1,
    "latent_dim": 5,
    "batch_mean": False,
    "learn_scale": False,
    "learning_rate": 0.1,
    "batch_size": 28,
    "num_epochs": 100,
    "D_rounds": 5,
    "G_rounds": 1,
    "use_time": False,
    "WGAN": False,
    "WGAN_clip": False,
    "shuffle": True,
    "wrong_labels": False,
    "identifier": DEFAULT_EXP_NAME,  # (EXPERIMENT_NAME, "sine")
    "dp": False,
    "l2norm_bound": 1e-05,
    "batches_per_lot": 1,
    "dp_sigma": 1e-05,
    "num_generated_features": 4
}

DEFAULT_train_frac = 0.7  # (0.7, 0.3) split.

DEFAULT_vis_freq = 1
DEFAULT_eval_freq = 1

dp_trace_enabled = False
target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]  # For privacy accountant.


def rgan(ori_data, parameters):

    out_data = None
    model.CUSTOM_EXPERIMENT = True  # Custom modifications enabled.

    # --- get settings --- #
    settings = copy.deepcopy(DEFAULT_PARAMS)
    settings.update(parameters)

    # --- get data, split --- #
    if settings["custom_experiment"] is True:
        n_samples = ori_data.shape[0]
        threshold = int(DEFAULT_train_frac * n_samples)
        samples = dict()
        samples["train"], samples["vali"] = ori_data[:threshold, :, :], ori_data[threshold:, :, :]
        assert samples["train"].shape[0] + samples["vali"].shape[0] == n_samples
        samples["test"] = None
        pdf, labels = None, {"train": None, "vali": None, "test": None}
        settings["num_samples"] = n_samples  # Overwrite if incorrect.
    else:
        samples, pdf, labels = data_utils.get_samples_and_labels(settings)
    
    print(f"Train set shape:\n{samples['train'].shape}")
    print(f"Validation set shape:\n{samples['vali'].shape}")
    if samples["test"] is not None:
        print(f"Test set shape:\n{samples['test'].shape}")
    else:
        print("Test set not provided.")

    # --- save settings, data --- #
    print('Ready to run with settings:')
    for (k, v) in settings.items(): 
        v_ = f"'{v}'" if isinstance(v, str) else v
        print(f"'{k}': {v_}")
    json.dump(
        settings, 
        open('./generative_models/rgan/experiments/settings/' + settings["identifier"] + '.txt', 'w'), 
        indent=0
    )

    if not settings["data"] == 'load':
        data_path = './generative_models/rgan/experiments/data/' + settings["identifier"] + '.data.npy'
        np.save(data_path, {'samples': samples, 'pdf': pdf, 'labels': labels})
        print('Saved training data to', data_path)

    # --- build model --- #

    Z, X, CG, CD, CS = model.create_placeholders(
        settings["batch_size"], 
        settings["seq_length"], 
        settings["latent_dim"], 
        settings["num_signals"], 
        settings["cond_dim"])

    discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
    discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
    generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale']
    generator_settings = dict((k, settings[k]) for k in generator_vars)

    CGAN = (settings["cond_dim"] > 0)
    if CGAN: assert not settings["predict_labels"]

    D_loss, G_loss = model.GAN_loss(
        Z=Z, 
        X=X, 
        generator_settings=generator_settings, 
        discriminator_settings=discriminator_settings, 
        kappa=settings["kappa"], 
        cond=CGAN, 
        CG=CG, 
        CD=CD, 
        CS=CS, 
        wrong_labels=settings["wrong_labels"]
    )
    D_solver, G_solver, priv_accountant = model.GAN_solvers(
        D_loss=D_loss, 
        G_loss=G_loss, 
        learning_rate=settings["learning_rate"], 
        batch_size=settings["batch_size"], 
        total_examples=samples['train'].shape[0], 
        l2norm_bound=settings["l2norm_bound"],
        batches_per_lot=settings["batches_per_lot"], 
        sigma=settings["dp_sigma"], 
        dp=settings["dp"]
    )
    G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG)

    # --- evaluation --- #

    # frequency to do visualisations
    vis_freq = DEFAULT_vis_freq  # Original: max(14000//settings["num_samples"], 1)
    eval_freq = DEFAULT_eval_freq  # Original: max(7000//settings["num_samples"], 1)

    # get heuristic bandwidth for mmd kernel from evaluation samples
    heuristic_sigma_training = median_pairwise_distance(samples['vali'])
    best_mmd2_so_far = 1000

    # optimise sigma using that (that's t-hat)
    if settings["custom_experiment"] is True:
        eval_size_target = samples['train'].shape[0] + samples['vali'].shape[0]
        batch_multiplier = eval_size_target//settings["batch_size"] + 1
    else:
        eval_size_target = 5000
        batch_multiplier = eval_size_target//settings["batch_size"]
    eval_size = batch_multiplier*settings["batch_size"]
    # print("eval_size", eval_size)
    eval_eval_size = int(0.2*eval_size)
    eval_real_PH = tf.placeholder(
        tf.float32, [eval_eval_size, settings["seq_length"], settings["num_generated_features"]])
    eval_sample_PH = tf.placeholder(
        tf.float32, [eval_eval_size, settings["seq_length"], settings["num_generated_features"]])
    n_sigmas = 2
    sigma = tf.get_variable(
        name='sigma', 
        shape=n_sigmas, 
        initializer=tf.constant_initializer(value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas)))
    )
    mmd2, that = mix_rbf_mmd2_and_ratio(X=eval_real_PH, Y=eval_sample_PH, sigmas=sigma)
    with tf.variable_scope("SIGMA_optimizer"):
        sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
        # sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
        # sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
    sigma_opt_iter = 2000
    sigma_opt_thresh = 0.001
    sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    vis_Z = model.sample_Z(
        batch_size=settings["batch_size"], 
        seq_length=settings["seq_length"], 
        latent_dim=settings["latent_dim"], 
        use_time=settings["use_time"]
    )
    if CGAN:
        vis_C = model.sample_C(
            batch_size=settings["batch_size"], 
            cond_dim=settings["cond_dim"], 
            max_val=settings["max_val"], 
            one_hot=settings["one_hot"]
        )
        if 'mnist' in settings["data"]:
            if settings["one_hot"]:
                if settings["cond_dim"] == 6:
                    vis_C[:6] = np.eye(6)
                elif settings["cond_dim"] == 3:
                    vis_C[:3] = np.eye(3)
                    vis_C[3:6] = np.eye(3)
                else:
                    raise ValueError(settings["cond_dim"])
            else:
                if settings["cond_dim"] == 6:
                    vis_C[:6] = np.arange(settings["cond_dim"])
                elif settings["cond_dim"] == 3:
                    vis_C = np.tile(np.arange(3), 2)
                else:
                    raise ValueError(settings["cond_dim"])
        elif 'eICU_task' in settings["data"]:
            vis_C = labels['train'][
                np.random.choice(labels['train'].shape[0], 
                settings["batch_size"], replace=False), :]
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
    else:
        vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
        vis_C = None

    vis_real_indices = np.random.choice(len(samples['vali']), size=6)
    vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
    if not labels['vali'] is None:
        vis_real_labels = labels['vali'][vis_real_indices]
    else:
        vis_real_labels = None
    if settings["data"] == 'mnist':
        if settings["predict_labels"]:
            assert labels['vali'] is None
            n_labels = 1
            if settings["one_hot"]: 
                n_labels = 6
                lab_votes = np.argmax(vis_real[:, :, -n_labels:], axis=2)
            else:
                lab_votes = vis_real[:, :, -n_labels:]
            labs, _ = mode(lab_votes, axis=1) 
            samps = vis_real[:, :, :-n_labels]
        else:
            labs = None
            samps = vis_real
        if settings["multivariate_mnist"]:
            plotting.save_mnist_plot_sample(
                samples=samps.reshape(-1, settings["seq_length"]**2, 1), 
                idx=0, 
                identifier=settings["identifier"] + '_real', 
                n_samples=6, 
                labels=labs)
        else:
            plotting.save_mnist_plot_sample(
                samples=samps, 
                idx=0, 
                identifier=settings["identifier"] + '_real', 
                n_samples=6, 
                labels=labs)
    elif 'eICU' in settings["data"]:
        plotting.vis_eICU_patients_downsampled(
            pat_arrs=vis_real, 
            time_step=settings["resample_rate_in_min"], 
            identifier=settings["identifier"] + '_real', 
            idx=0)
    else:
        plotting.save_plot_sample(
            samples=vis_real, 
            idx=0, 
            identifier=settings["identifier"] + '_real', 
            n_samples=6, 
            num_epochs=settings["num_epochs"])

    # for dp
    if dp_trace_enabled:
        dp_trace = open('./generative_models/rgan/experiments/traces/' + settings["identifier"] + '.dptrace.txt', 'w')
        dp_trace.write('epoch ' + ' eps' .join(map(str, target_eps)) + '\n')

    trace = open('./generative_models/rgan/experiments/traces/' + settings["identifier"] + '.trace.txt', 'w')
    trace.write('epoch time D_loss G_loss mmd2 that pdf real_pdf\n')

    # --- train --- #
    train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length', 
                'latent_dim', 'num_generated_features', 'cond_dim', 'max_val', 
                'WGAN_clip', 'one_hot']
    train_settings = dict((k, settings[k]) for k in train_vars)

    t0 = time()
    best_epoch = 0
    print('epoch\ttime\tD_loss\tG_loss\tmmd2\tthat\tpdf_sample\tpdf_real')
    for epoch in range(settings["num_epochs"]):
        D_loss_curr, G_loss_curr = model.train_epoch(
            epoch=epoch, 
            samples=samples['train'], 
            labels=labels['train'],
            sess=sess, 
            Z=Z, 
            X=X, 
            CG=CG, 
            CD=CD, 
            CS=CS,
            D_loss=D_loss, 
            G_loss=G_loss,
            D_solver=D_solver, 
            G_solver=G_solver, 
            **train_settings
        )
        # -- eval -- #

        # visualise plots of generated samples, with/without labels
        if epoch % vis_freq == 0:
            if CGAN:
                vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
            else:
                vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
            plotting.visualise_at_epoch(
                vis_sample=vis_sample, 
                data=settings["data"], 
                predict_labels=settings["predict_labels"], 
                one_hot=settings["one_hot"], 
                epoch=epoch, 
                identifier=settings["identifier"], 
                num_epochs=settings["num_epochs"],
                resample_rate_in_min=settings["resample_rate_in_min"], 
                multivariate_mnist=settings["multivariate_mnist"], 
                seq_length=settings["seq_length"], 
                labels=vis_C
            )
    
        # compute mmd2 and, if available, prob density
        if (epoch % eval_freq == 0) or (epoch == settings["num_epochs"] - 1):
            ## how many samples to evaluate with?
            eval_Z = model.sample_Z(
                batch_size=eval_size, 
                seq_length=settings["seq_length"], 
                latent_dim=settings["latent_dim"], 
                use_time=settings["use_time"]
            )
            if 'eICU_task' in settings["data"]:
                eval_C = labels['vali'][np.random.choice(labels['vali'].shape[0], eval_size), :]
            else:
                eval_C = model.sample_C(
                    batch_size=eval_size, 
                    cond_dim=settings["cond_dim"], 
                    max_val=settings["max_val"], 
                    one_hot=settings["one_hot"]
                )
            eval_sample = np.empty(shape=(eval_size, settings["seq_length"], settings["num_signals"]))
            for i in range(batch_multiplier):
                if CGAN:
                    eval_sample[i*settings["batch_size"]:(i+1)*settings["batch_size"], :, :] = \
                        sess.run(
                            G_sample, 
                            feed_dict={
                                Z: eval_Z[i*settings["batch_size"]:(i+1)*settings["batch_size"]], 
                                CG: eval_C[i*settings["batch_size"]:(i+1)*settings["batch_size"]]
                            }
                        )
                else:
                    eval_sample[i*settings["batch_size"]:(i+1)*settings["batch_size"], :, :] = \
                        sess.run(
                            G_sample, 
                            feed_dict={Z: eval_Z[i*settings["batch_size"]:(i+1)*settings["batch_size"]]}
                        )
            eval_sample = np.float32(eval_sample)
            eval_real = np.float32(
                samples['vali'][np.random.choice(len(samples['vali']), size=batch_multiplier*settings["batch_size"]), 
                :, :])

            eval_eval_real = eval_real[:eval_eval_size]
            eval_test_real = eval_real[eval_eval_size:]
            eval_eval_sample = eval_sample[:eval_eval_size]
            eval_test_sample = eval_sample[eval_eval_size:]
            
            ## MMD
            # reset ADAM variables
            sess.run(tf.initialize_variables(sigma_opt_vars))
            sigma_iter = 0
            that_change = sigma_opt_thresh*2
            old_that = 0
            while that_change > sigma_opt_thresh and sigma_iter < sigma_opt_iter:
                new_sigma, that_np, _ = sess.run(
                    [sigma, that, sigma_solver], 
                    feed_dict={eval_real_PH: eval_eval_real, eval_sample_PH: eval_eval_sample}
                )
                that_change = np.abs(that_np - old_that)
                old_that = that_np
                sigma_iter += 1
            opt_sigma = sess.run(sigma)
            
            # print("------------")
            # print("REAL:")
            # print(eval_test_real)
            # print("GENERATED:")
            # print(eval_test_sample)
            # print("------------")

            mmd2, that_np = sess.run(
                mix_rbf_mmd2_and_ratio(X=eval_test_real, Y=eval_test_sample, biased=False, sigmas=sigma)
            )
        
            ## save parameters
            if mmd2 < best_mmd2_so_far and epoch > 10:
                best_epoch = epoch
                best_mmd2_so_far = mmd2
                model.dump_parameters(settings["identifier"] + '_' + str(epoch), sess)
                
                if settings["custom_experiment"] is True:
                    out_data = eval_sample[:eval_size_target, :, :].copy()
        
            ## prob density (if available)
            if not pdf is None:
                pdf_sample = np.mean(pdf(eval_sample[:, :, 0]))
                pdf_real = np.mean(pdf(eval_real[:, :, 0]))
            else:
                pdf_sample = 'NA'
                pdf_real = 'NA'
        else:
            # report nothing this epoch
            mmd2 = 'NA'
            that = 'NA'
            pdf_sample = 'NA'
            pdf_real = 'NA'
        
        ## get 'spent privacy'
        if settings["dp"] and dp_trace_enabled:
            spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
            # get the moments
            deltas = []
            for (spent_eps, spent_delta) in spent_eps_deltas:
                deltas.append(spent_delta)
            dp_trace.write(str(epoch) + ' ' + ' '.join(map(str, deltas)) + '\n')
            if epoch % 10 == 0: dp_trace.flush()

        ## print
        t = time() - t0
        try:
            print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f\t%.2f\t%.2f' % (
                epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))
        except TypeError:       # pdf are missing (format as strings)
            print('%d\t%.2f\t%.4f\t%.4f\t%.5f\t%.0f\t %s\t %s' % (
                epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real))

        ## save trace
        trace.write(' '.join(map(str, [epoch, t, D_loss_curr, G_loss_curr, mmd2, that_np, pdf_sample, pdf_real])) + '\n')
        if epoch % 10 == 0: 
            trace.flush()
            plotting.plot_trace(
                identifier=settings["identifier"], 
                xmax=settings["num_epochs"], 
                dp=settings["dp"]
            )

        if settings["shuffle"]:     # shuffle the training data 
            perm = np.random.permutation(samples['train'].shape[0])
            samples['train'] = samples['train'][perm]
            if labels['train'] is not None:
                labels['train'] = labels['train'][perm]
        
        if epoch % 50 == 0:
            model.dump_parameters(identifier=settings["identifier"] + '_' + str(epoch), sess=sess)

    trace.flush()
    plotting.plot_trace(identifier=settings["identifier"], xmax=settings["num_epochs"], dp=settings["dp"])
    model.dump_parameters(identifier=settings["identifier"] + '_' + str(epoch), sess=sess)

    if settings["custom_experiment"] is True and out_data is None:
        out_data = eval_sample[:eval_size_target, :, :].copy()
    print("Final generated data shape:", eval_sample.shape)

    return out_data
