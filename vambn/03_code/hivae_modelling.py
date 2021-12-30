import os
import re
import csv
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import tensorflow as tf
import configparser
import warnings
warnings.filterwarnings("ignore")

# originally from loglik_models_missing_normalize
class c_loglik_models_missing_normalize:
    """
    Compute log-likelihood based on different datatypes.
    """

    def __init__(self, batch_data, list_type, theta, normalization_params):
        """
        Read and assign variables.
        """
        
        self.batch_data = batch_data
        self.list_type = list_type
        self.theta = theta
        self.normalization_params = normalization_params


    def loglik_real(self):
        """
        Compute log-likelihood for 'real' type data.
        Return log p(x), log p(x_missing), params, and samples.
        """
        
        # define empty dict and epsilon constant
        output = dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        # get batch-normed data and miss mask
        # convert miss mask to float
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask, tf.float32)
        
        # get mean and variance of batch normed data
        # limit the range of variance to a range of epsilon value and infinity
        data_mean, data_var = self.normalization_params
        data_var = tf.clip_by_value(data_var, epsilon, np.inf)
        
        # get mean and variance of theta
        # limit the range of variance to only positive values
        est_mean, est_var = self.theta
        est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1.0) 
        
        # affine transformation of the parameters
        est_mean = tf.sqrt(data_var) * est_mean + data_mean
        est_var = data_var * est_var
        
        # compute loglik
        log_p_x = -0.5 * tf.reduce_sum(tf.math.squared_difference(data,est_mean)/est_var,1) \
            - int(self.list_type['dim'])*0.5*tf.math.log(2*np.pi) - 0.5*tf.reduce_sum(tf.math.log(est_var),1)
        
        # multiply computed log-likelihood with miss mask
        # sample from normal distribution of the parameters
        # append all result as output
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = [est_mean, est_var]
        output['samples'] = tf.compat.v1.distributions.Normal(est_mean,tf.sqrt(est_var)).sample()
        
        # return log p(x), log p(x_missing), params, and samples
        return output


    def loglik_pos(self):
        """
        Compute log-likelihood for 'real positive' type data (log-normal distribution).
        Return log p(x), log p(x_missing), params, and samples.
        """

        # define empty dict and epsilon constant
        output = dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        # get mean and variance of batch normed data
        # limit the range of variance to a range of epsilon value and infinity
        data_mean_log, data_var_log = self.normalization_params
        data_var_log = tf.clip_by_value(data_var_log, epsilon, np.inf)
        
        # get batch-normed data and miss mask
        # get log of data
        # convert miss mask to float
        data, missing_mask = self.batch_data
        data_log = tf.math.log(1.0 + data)
        missing_mask = tf.cast(missing_mask, tf.float32)
        
        # get mean and variance of theta
        # limit the range of variance to only positive values
        est_mean, est_var = self.theta
        est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1.0)
        
        # affine transformation of the parameters
        est_mean = tf.sqrt(data_var_log) * est_mean + data_mean_log
        est_var = data_var_log * est_var
        
        # compute loglik
        log_p_x = -0.5 * tf.reduce_sum(tf.math.squared_difference(data_log,est_mean)/est_var,1) \
            - 0.5*tf.reduce_sum(tf.math.log(2*np.pi*est_var),1) - tf.reduce_sum(data_log,1)
        
        # multiply computed log-likelihood with miss mask
        # sample from normal distribution of the parameters
        # append all result as output
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = [est_mean, est_var]
        output['samples'] = tf.exp(tf.compat.v1.distributions.Normal(est_mean,tf.sqrt(est_var)).sample()) - 1.0

        # return log p(x), log p(x_missing), params, and samples    
        return output


    def loglik_cat(self):
        """
        Compute log-likelihood for 'categorical' type data.
        Return log p(x), log p(x_missing), params, and samples.
        """

        # define empty dict
        output=dict()
        
        # get batch-normed data and miss mask
        # convert miss mask to float
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask, tf.float32)
        
        # get mean and variance of theta
        log_pi = self.theta
        
        # compute loglik
        log_p_x = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=data)
        
        # multiply computed log-likelihood with miss mask
        # sample from categorical distribution of the parameters, convert to one-hot encoding
        # append all result as output
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = log_pi
        output['samples'] = tf.one_hot(tf.compat.v1.distributions.Categorical(probs=tf.nn.softmax(log_pi)).sample(),depth=int(self.list_type['dim']))
        
        # return log p(x), log p(x_missing), params, and samples
        return output


    def loglik_ordinal(self):
        """
        Compute log-likelihood for 'ordinal' type data.
        Return log p(x), log p(x_missing), params, and samples.
        """

        # define empty dict and epsilon constant
        output=dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        # get batch-normed data and miss mask
        # convert miss mask to float
        # get batch size
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask, tf.float32)
        batch_size = tf.shape(data)[0]
        
        # get mean and variance of theta
        # reshape the mean 
        # limit the range of variance to only positive values
        # force the outputs of the network to increase with the categories
        # compute sigmoid and theta mean
        partition_param, mean_param = self.theta
        mean_value = tf.reshape(mean_param,[-1,1])
        theta_values = tf.cumsum(tf.clip_by_value(tf.nn.softplus(partition_param), epsilon, 1e20), 1)
        sigmoid_est_mean = tf.nn.sigmoid(theta_values - mean_value)
        mean_probs = tf.concat([sigmoid_est_mean, tf.ones([batch_size,1], tf.float32)], 1) - tf.concat([tf.zeros([batch_size,1], tf.float32), sigmoid_est_mean], 1)
        
        # compute samples from an ordinal distribution
        true_values = tf.one_hot(tf.reduce_sum(tf.cast(data,tf.int32),1)-1, int(self.list_type['dim']))
        
        # compute loglik
        log_p_x = tf.math.log(tf.clip_by_value(tf.reduce_sum(mean_probs*true_values,1),epsilon,1e20))
        
        # multiply computed log-likelihood with miss mask
        # sample from categorical distribution of the parameters
        # append all result as output
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = mean_probs
        output['samples'] = tf.sequence_mask(1+tf.compat.v1.distributions.Categorical(logits=tf.math.log(tf.clip_by_value(mean_probs,epsilon,1e20))).sample(), int(self.list_type['dim']),dtype=tf.float32)
        
        # return log p(x), log p(x_missing), params, and samples  
        return output


    def loglik_count(self):
        """
        Compute log-likelihood for 'count' type data.
        Return log p(x), log p(x_missing), params, and samples.
        """

        # define empty dict and epsilon constant
        output=dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        # get batch-normed data and miss mask
        # convert miss mask to float
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask, tf.float32)
        
        # get mean and variance of theta
        # limit the range of variance to only positive values
        est_lambda = self.theta
        est_lambda = tf.clip_by_value(tf.nn.softplus(est_lambda), epsilon, 1e20)
        
        # calculate log p(x) with log poisson loss
        log_p_x = -tf.reduce_sum(tf.nn.log_poisson_loss(targets=data,log_input=tf.math.log(est_lambda),compute_full_loss=True),1)
        
        # multiply computed log-likelihood with miss mask
        # sample from poisson distribution of the parameters
        # append all result as output
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = est_lambda
        output['samples'] = tf.compat.v1.distributions.Poisson(est_lambda).sample()
        
        # return log p(x), log p(x_missing), params, and samples   
        return output


class c_HIVAE_Modelling:
    """
    Perform model training on HIVAE with the best hyperparameters from grid search.
    """

    def __init__(self):
        """
        Read config file and assign configs as variables.
        """
        
        # disable matplotlib interactive mode
        plt.ioff()

        # read Python config file
        cp = configparser.RawConfigParser()
        config_file = r'/vambn/02_config/config_python.txt'
        assert os.path.exists(config_file)
        cp.read(config_file)
        print(f'[*] Config file sections: {cp.sections()}')

        # assign configs as variables and list
        self.data_python = cp.get('HyperparameterOptimization', 'path_data_python')
        self.results = cp.get('HyperparameterOptimization', 'path_results')

        self.sample_size = int(cp.get('Modelling', 'sample_size')) 
        self.sds = [int(i) for i in cp.get('Modelling', 'sds').split(',')]
        self.seed = int(cp.get('Modelling', 'seed_n')) 
        
        self.python_names = cp.get('Modelling', 'path_python_names') 
        self.saved_networks = cp.get('Modelling', 'path_saved_networks') 
        self.train_stats = cp.get('Modelling', 'path_train_stats') 
        self.VP_misslist = cp.get('Modelling', 'path_VP_misslist') 
        self.metaenc = cp.get('Modelling', 'path_metaenc') 
        self.embedding_plot = cp.get('Modelling', 'path_embedding_plot') 
        self.reconRP = cp.get('Modelling', 'path_reconRP') 
        self.training_logliks = cp.get('Modelling', 'path_training_logliks') 
        self.virtual_ppts = cp.get('Modelling', 'path_virtual_ppts') 
        self.decoded_VP = cp.get('Modelling', 'path_decoded_VP') 
        self.virtual_logliks = cp.get('Modelling', 'path_virtual_logliks') 
        

    # originally from 1_ADNI_HIVAE_training.ipynb
    def train_HIVAE(self):
        """
        Train HIVAE model for all variable groups based on the best hyperparams.
        """
        
        ##################### READ DATA ############################################

        # read data files for every variable group
        # sort data files
        files = [i for i in os.listdir(self.data_python) if not '_type' in i and not '_missing' in i and i not in '.DS_Store']
        files.sort()
        print("[*] Files: ")
        print(files)

        # get the best hyperparam config
        # get file names, convert and sort file names
        best_hyper = pd.read_csv(self.results)
        ls_hyper_files = best_hyper['files'].tolist()
        ls_hyper_files.sort()
        print("[*] Best hyper (current file): ")
        print(ls_hyper_files)

        # if data file list is not the same length with hyperparam file list
        # print error message
        # if data file list is the same length with hyperparam file list
        # add a s_dimension column
        if files != ls_hyper_files:
            print("[*] ERROR!!")
        else:
            best_hyper['sdims'] = self.sds
        
        # display the best hyperparam configs for each variable group
        print("[*] Best hyperparameters (all):")
        print(best_hyper)

        ##################### TRAIN MODEL ########################################

        # for each variable group
        # get best hyperparam config for the current variable group
        # set setting for model config
        # train HIVAE model
        for f in files:
            print(f'\n[*] Currently training model for: {f}. Please wait...')
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            settings = self.set_settings(opts, nepochs=opts['epochs'].iloc[0], modload=False, save=True)
            self.train_network(settings)
        print("[*] Model training done for all modules.")

        ##################### GET EMBEDDINGS #####################################

        # get network embeddings of the trained model
        # define empty lists
        dat = list()
        dfs = list()

        # for each variable group
        for f in files:
            # get best hyperparam config for the current variable group
            # add a n_batch column
            # set setting for model config
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            # run encoding part of HIVAE
            encs, encz, d = self.enc_network(settings)

            # make deterministic embeddings
            # read subject file csv
            # get s and z encoded embeddings and merge as one dataframe
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv', '', f)}_subj.csv")['x']
            sc = pd.DataFrame({'scode_'+re.sub('.csv','',f) : pd.Series(np.array([i for i in encs])), 'SUBJID' : subj})
            zc = pd.DataFrame({'zcode_'+re.sub('.csv','',f) : pd.Series(np.array([i[0] for i in encz])), 'SUBJID' : subj})
            enc = pd.merge(sc, zc, on = 'SUBJID')
            
            # save encoded embedding as metadata csv
            # append embedding to variable dfs
            # append d to variable dat
            enc.to_csv(f"{self.saved_networks}{re.sub('.csv','',f)}_meta.csv", index=False)
            dfs.append(enc)
            dat.append(d)

        # join metadata
        # load metadata csv file for all variable groups
        # merge all as one dataframe, save it
        enc_vars = [pd.read_csv(f"{self.saved_networks}{re.sub('.csv','',f)}_meta.csv") for f in files]
        meta = self.merge_dat(enc_vars)
        meta[meta.columns[['Unnamed' not in i for i in meta.columns]]].to_csv(self.metaenc, index=False)

        # define variable metadata without subjid and scode columns
        metadata = meta[meta.columns.drop(list(meta.filter(regex='SUBJID|scode_')))]
        print('[*] Metadata:')
        print(metadata)
        print(f'[*] Metadata NaN: {metadata.isnull().sum().sum()}')

        # display embedding scatter matrix based on metadata
        plt.clf()
        plt.figure()
        fig = scatter_matrix(metadata, figsize=[15, 15], marker=".", s=10, diagonal="kde")
        for ax in fig.ravel():
            ax.set_xlabel(re.sub('_VIS|zcode_', '', ax.get_xlabel()), fontsize=20, rotation=90)
            ax.set_ylabel(re.sub('_VIS|zcode_', '', ax.get_ylabel()), fontsize=20, rotation=90)
        
        # define the subtitle of the embedding plot and save the figure
        plt.suptitle('HI-VAE embeddings (deterministic)', fontsize=20)
        plt.savefig(self.embedding_plot, bbox_inches='tight')

        ##################### REAL PATIENT RECONSTRUCTION #########################

        # real patient decoding (reconstruction)
        # load the metadata for all variable group
        # define empty lists
        meta = pd.read_csv(self.metaenc)
        recon = list()
        recdfs = list()

        # for each variable group data file
        for f in files:
            # replace placeholders in template
            # get best hyperparam config for the current variable group
            # add nbatch column
            # set setting for model config
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            # get zcode and scode metadata
            # execute the decoder part of HIVAE 
            # append the reconstruction output to variable recon
            zcodes = meta['zcode_'+re.sub('.csv', '', f)]
            scodes = meta['scode_'+re.sub('.csv', '', f)]
            rec = self.dec_network(settings, zcodes, scodes)
            recon.append(rec)
            
            # load subject and column names from csv
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_subj.csv")['x']
            names = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_cols.csv")['x']

            # get reconstruction output
            # append column names and subjectid to the dataframe
            # append the dataframe to variable recdfs
            recd = pd.DataFrame(rec)
            recd.columns = names
            recd['SUBJID'] = subj
            recdfs.append(recd)

        # merge all reconstructed output on subjid
        # save the output as csv
        data_recon = self.merge_dat(recdfs)
        data_recon.to_csv(self.reconRP, index=False)

        ##################### GET LOG-LIKELIHOODS ###################################

        # get likelihoods
        # read metadata file for all variable groups
        # define empty list
        meta = pd.read_csv(self.metaenc)
        dfs = list()

        # for each variable group data file
        for f in files:
            # get best hyperparam config for the current variable group
            # add a n_batch column
            # set setting for model config
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            # get zcode and scode metadata
            zcodes = meta['zcode_'+re.sub('.csv', '', f)]
            scodes = meta['scode_'+re.sub('.csv', '', f)]
            
            # execute the decoder log-likelihood part of HIVAE
            # get the mean of log-likelihood
            loglik = self.dec_network_loglik(settings, zcodes, scodes)
            loglik = np.nanmean(np.array(loglik).T, axis=1)

            # load subject from csv
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv', '', f)}_subj.csv")['x']

            # convert mean of log-likelihood as a dataframe
            # add file name, subjid columns
            # append the dataframe to variable dfs
            dat = pd.DataFrame(loglik)
            dat.columns = [f]
            dat['SUBJID'] = subj
            dfs.append(dat)

        # merge all mean of log-likelihood on subjid
        # save the output as csv
        decoded = self.merge_dat(dfs)
        decoded.to_csv(self.training_logliks, index=False)

        print('[*] HI-VAE model training script completed.')


    # originally from 2_ADNI_HIVAE_VP_decoding_and_counterfactuals.ipynb
    def decode_HIVAE(self):
        """
        Execute the decoder part of HIVAE model for all variable groups based on the best hyperparams.
        """
        
        # read data files for every variable group
        # sort data files
        files = [i for i in os.listdir(self.data_python) if not '_type' in i and not '_missing' in i and not '.DS_Store' in i]
        files = sorted(files)
        print(f"[*] File type: {type(files)}")
        print("[*] Files: ")
        print(files)

        # get the best hyperparam config
        # get file names, convert and sort file names
        best_hyper = pd.read_csv(self.results)

        # if data file list is not the same length with hyperparam file list
        # print error message
        if any(files != best_hyper['files']):
            print("[*] ERROR!!")
        
        # if data file list is the same length with hyperparam file list
        # add a s_dimension column
        else:
            best_hyper['sdims'] = self.sds
        
        # display the best hyperparam configs for each variable group
        print("[*] Best hyperparameters:")
        print(best_hyper)

        ##################### VIRTUAL PATIENT RECONSTRUCTION ########################

        # read virtual patient decoding csv file
        # define empty lists
        VPcodes = pd.read_csv(self.virtual_ppts)
        dfs = list()
        virt = list()

        # for each variable group
        for f in files:
            # get best hyperparam config for the current variable group
            # add a n_batch column
            # set setting for model config
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            # get zcode and scode metadata
            zcodes = VPcodes['zcode_'+ re.sub('.csv', '', f)]
            scodes = VPcodes['scode_'+ re.sub('.csv', '', f)] if 'scode_'+re.sub('.csv', '', f) in VPcodes.columns else np.zeros(zcodes.shape)
    
            # execute the decoder part of HIVAE
            dec = self.dec_network(settings, zcodes, scodes, VP=True)

            # load subject and column names from csv
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_subj.csv")['x']
            names = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_cols.csv")['x']

            # get decoded output
            # append column names and subjectid to the dataframe
            # append the dataframe to variable recdfs and dfs
            dat = pd.DataFrame(dec)
            dat.columns = names
            dat['SUBJID'] = subj
            virt.append(dec)
            dfs.append(dat)

        # merge all decoded output on subjid
        # save the output as csv
        decoded = self.merge_dat(dfs)
        decoded.to_csv(self.decoded_VP, index=False)

        ##################### GET LOG-LIKELIHOODS ###################################

        # get log likelihoods for R plot
        # read virtual patient decoding csv file
        # define empty list
        VPcodes = pd.read_csv(self.virtual_ppts)
        dfs = list()

        # for each variable group
        for f in files:
            # get best hyperparam config for the current variable group
            # add a n_batch column
            # set setting for model config
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            # get zcode and scode metadata
            zcodes = VPcodes['zcode_'+re.sub('.csv','',f)]
            scodes = VPcodes['scode_'+re.sub('.csv','',f)] if 'scode_'+re.sub('.csv','',f) in VPcodes.columns else np.zeros(zcodes.shape)
            
            # execute the decoder log-likelihood part of HIVAE
            # get mean of log-likelihood
            loglik = self.dec_network_loglik(settings, zcodes, scodes, VP=True)
            loglik = np.nanmean(np.array(loglik).T, axis=1)

            # load subject from csv
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv', '', f)}_subj.csv")['x']

            # convert mean of log-likelihood as a dataframe
            # append column names and subjectid to the dataframe
            # append the dataframe to variable dfs
            dat = pd.DataFrame(loglik)
            dat.columns = [f]
            dat['SUBJID'] = subj
            dfs.append(dat)

        # merge all decoded log-likelihood output on subjid
        # save the output as csv
        decoded = self.merge_dat(dfs)
        decoded.to_csv(self.virtual_logliks, index=False)

        print('[*] HI-VAE model virtual patients decoding script completed.')


    # originally from 1_ADNI_HIVAE_training.ipynb
    # note: modload doesnt do anything right now, hardcoded in helpers.py
    def set_settings(self, opts, nepochs=500, modload=False, save=True): 
        """
        Set configs for a single grid search experiment.
        Return settings as a string.
        """

        # set file name for data, missing and types files
        inputf = re.sub('.csv', '', opts['files'].iloc[0])
        missf = inputf + '_missing.csv'
        typef = inputf + '_types.csv'

        # create settings as a string variable
        settings = f"--epochs {nepochs} \
                    --restore {1 if modload else 0} \
                    --data_file {self.data_python}{inputf}.csv \
                    --types_file {self.data_python}{typef} \
                    --batch_size {opts['nbatch'].iloc[0]} \
                    --save {nepochs-1 if save else nepochs*2} \
                    --save_file {inputf}\
                    --dim_latent_s {opts['sdims'].iloc[0]} \
                    --dim_latent_z 1 \
                    --dim_latent_y {opts['ydims'].iloc[0]} \
                    --miss_percentage_train 0 \
                    --miss_percentage_test 0 \
                    --true_miss_file {self.data_python}{missf} \
                    --learning_rate {opts['lrates'].iloc[0]} \
                    --weight_decay {opts['wdecay'].iloc[0]} \
                    --adam_beta_1 {opts['beta1'].iloc[0]} \
                    --adam_beta_2 {opts['beta2'].iloc[0]}"
        
        # return all settings as a string
        return settings
    

    # originally from parser_arguments.py
    def getArgs(self, argv=None):
        """
        Convert string settings to arguments for a single experiment.
        Return all arguments.
        """

        # define all arguments with default values and descriptions
        parser = argparse.ArgumentParser(description='Default parameters of the models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--batch_size', type=int, default=200, help='Size of the batches')
        parser.add_argument('--epochs', type=int, default=5001, help='Number of epochs of the simulations')
        parser.add_argument('--perp', type=int, default=10, help='Perplexity for the t-SNE')
        parser.add_argument('--display', type=int, default=1, help='Display option flag')
        parser.add_argument('--save', type=int, default=1000, help='Save variables every save iterations')
        parser.add_argument('--restore', type=int, default=0, help='To restore session, to keep training or evaluation') 
        parser.add_argument('--plot', type=int, default=1, help='Plot results flag')
        parser.add_argument('--dim_latent_s', type=int, default=10, help='Dimension of the categorical space')
        parser.add_argument('--dim_latent_z', type=int, default=2, help='Dimension of the Z latent space')
        parser.add_argument('--dim_latent_y', type=int, default=10, help='Dimension of the Y latent space')
        parser.add_argument('--dim_latent_y_partition', type=int, nargs='+', help='Partition of the Y latent space')
        parser.add_argument('--miss_percentage_train', type=float, default=0.0, help='Percentage of missing data in training')
        parser.add_argument('--miss_percentage_test', type=float, default=0.0, help='Percentage of missing data in test')
        parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
        parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
        parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
        parser.add_argument('--miss_file', type=str, default='Missing_test.csv', help='File with the missing indexes mask')
        parser.add_argument('--true_miss_file', type=str, help='File with the missing indexes when there are NaN in the data')
        parser.add_argument('--learning_rate', type=float, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, help='L2: Weight decay')
        parser.add_argument('--adam_beta_1', type=float, default=0.9, help='Adam Beta 1')
        parser.add_argument('--adam_beta_2', type=float, default=0.999, help='Adam Beta 2')
    
        # convert string variable as arguments and return all arguments
        return parser.parse_args(argv)


    # originally from read_functions.py
    def read_data(self, data_file, types_file, miss_file, true_miss_file):
        """
        Load and impute missing data.
        Return data without missing values, types dict, miss_mask, true_miss_mask, and n_samples.  
        """
        
        # load types file csv as dict
        with open(types_file) as f:
            types_dict = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
        
        # load data file csv as float values and convert as an np array
        with open(data_file, 'r') as f:
            data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            data = np.array(data)
        
        # if true missing file is supplied
        if true_miss_file:
            # load true missing file csv as int values and convert as a list
            with open(true_miss_file, 'r') as f:
                missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
                missing_positions = np.array(missing_positions)
            
            # create an array mask of 1s
            true_miss_mask = np.ones([np.shape(data)[0], len(types_dict)])
            
            # replace the mask with value 0 where there are missing values
            # indices in the csv start at 1, so -1 to make the indices compatible in Python (which start at 0)
            true_miss_mask[missing_positions[:,0]-1, missing_positions[:,1]-1] = 0 
            
            # create another mask where data values isnan=1, not nan=0
            data_masked = np.ma.masked_where(np.isnan(data), data) 

            # for each item/row in types dict
            data_filler = []
            for i in range(len(types_dict)):
                # if the type is categorical or ordinal
                if types_dict[i]['type'] == 'cat' or types_dict[i]['type'] == 'ordinal':
                    
                    # get aux data (unique data values of current data column)
                    aux = np.unique(data[:,i])
                    
                    # if the first value of aux is not nan,
                    # fill with the first element of the cat (0, 1, or whatever)
                    # if the first value of aux is nan,
                    # append value 0 to data_filler list
                    if not np.isnan(aux[0]):
                        data_filler.append(aux[0])  
                    else:
                        data_filler.append(int(0))
                
                # if the type is other than categorical or ordinal, 
                # append the data_filler list with value 0.0
                else:
                    data_filler.append(0.0)
            
            # create a filled data array where all 1s of data_masked is filled with values of data_filler
            # this will impute all missing values with the first value of data column
            data = data_masked.filled(data_filler)
        
        # if no true missing file is supplied
        else:
            # create a placeholder true_miss_mask with all 1s as values
            # it doesn't affect our data
            true_miss_mask = np.ones([np.shape(data)[0], len(types_dict)]) 
        
        # create data with no missing values
        # for each column of the data array
        data_complete = []
        for i in range(np.shape(data)[1]):
            # if the type of the data column is categorical
            if types_dict[i]['type'] == 'cat':
                # get categories
                cat_data = [int(x) for x in data[:,i]]
                categories, indexes = np.unique(cat_data, return_inverse=True)
                # transform categories to a vector of 0:n_categories
                new_categories = np.arange(int(types_dict[i]['dim']))
                cat_data = new_categories[indexes]
                # create one hot encoding for the categories
                aux = np.zeros([np.shape(data)[0], len(new_categories)])
                aux[np.arange(np.shape(data)[0]), cat_data] = 1
                # append the one-hot encoding to data_complete
                data_complete.append(aux)
            
            # if the type of the data column is ordinal
            elif types_dict[i]['type'] == 'ordinal':
                # get categories
                cat_data = [int(x) for x in data[:,i]]
                categories, indexes = np.unique(cat_data, return_inverse=True)
                # transform categories to a vector of 0:n_categories
                new_categories = np.arange(int(types_dict[i]['dim']))
                cat_data = new_categories[indexes]
                # create thermometer encoding for the categories
                aux = np.zeros([np.shape(data)[0], 1+len(new_categories)])
                aux[:,0] = 1
                aux[np.arange(np.shape(data)[0]), 1+cat_data] = -1
                aux = np.cumsum(aux, 1)
                # append the thermometer encoding to data_complete
                data_complete.append(aux[:, :-1])
            
            # if the type of the data column is other than categorical or ordinal
            else:
                # append the transposed data to data_complete
                data_complete.append(np.transpose([data[:, i]]))

        # add value 1 at the end of data_complete, assign as variable data                
        data = np.concatenate(data_complete, 1)
        
        # create a miss_mask with all values 1s
        n_samples = np.shape(data)[0]
        n_variables = len(types_dict)
        miss_mask = np.ones([np.shape(data)[0], n_variables])

        # read missing mask from csv (contains positions of missing values)
        # get all missing positions as an array
        # replace the mask with value 0 where there are missing values
        # indices in the csv start at 1, so -1 to make the indices compatible in Python (which start at 0)
        # if there is no mask, assume all data is observed
        if os.path.isfile(miss_file):
            with open(miss_file, 'r') as f:
                missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
                missing_positions = np.array(missing_positions)
            miss_mask[missing_positions[:,0]-1, missing_positions[:,1]-1] = 0 
        
        # return data without missing values, types dict, miss_mask, true_miss_mask, and n_samples
        return data, types_dict, miss_mask, true_miss_mask, n_samples
    

    # originally from graph_new.py
    def HVAE_graph(self, types_file, batch_size, learning_rate=1e-3, beta1=0.9, beta2=0.999,z_dim=2, y_dim=1, s_dim=2, weight_decay=0, y_dim_partition=[]):
        """
        Create TensorFlow graph nodes for HIVAE.
        Return TF nodes (HIVAE model architecture).
        """

        # load placeholders
        batch_data_list, batch_data_list_observed, miss_list, miss_list_VP, tau, types_list, zcodes, scodes = self.place_holder_types(types_file, batch_size)
        
        # perform batch normalization
        X_list, normalization_params = self.batch_normalization(batch_data_list_observed, types_list, miss_list)
        
        # set the dimensionality of Y
        # if y_dim_partition list is defined, sum all the elements up 
        if y_dim_partition:
            y_dim_output = np.sum(y_dim_partition)
        
        # if y_dim_partition list is not defined, create y_dim_partition of all ones
        # then sum all the elements up
        else:
            y_dim_partition = y_dim * np.ones(len(types_list), dtype=int)
            y_dim_output = np.sum(y_dim_partition)
        
        # define the encoder of HIVAE
        print("[*] Defining Encoder...")
        samples, q_params = self.encoder(X_list, batch_size, z_dim, s_dim, tau, weight_decay)
        
        # define the decoder of HIVAE
        print("[*] Defining Decoder...")
        theta, samples, p_params, log_p_x, log_p_x_missing = self.decoder(batch_data_list, miss_list, types_list, samples, normalization_params, batch_size, z_dim, y_dim_output, y_dim_partition, weight_decay)
        
        # define the cost function and optimizer of HIVAE
        print("[*] Defining Cost function...")
        ELBO, loss_reconstruction, KL_z, KL_s = self.cost_function(log_p_x, p_params, q_params, z_dim, s_dim)
        optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(-ELBO, var_list=None)

        # define the fixed decoder of HIVAE 
        # for getting samples based on fixed code inputs (from virtual patient data. if not VP run, just set to ones)
        samples_zgen, test_params_zgen, log_p_x_zgen, log_p_x_missing_zgen = self.fixed_decoder(batch_data_list, X_list, miss_list_VP, miss_list, types_list, batch_size, y_dim_output, y_dim_partition, s_dim, normalization_params, zcodes, scodes, weight_decay)

        # packing all defined nodes as a dict
        tf_nodes = {'ground_batch' : batch_data_list,
                    'ground_batch_observed' : batch_data_list_observed,
                    'miss_list': miss_list,
                    'miss_list_VP': miss_list_VP,
                    'tau_GS': tau,
                    'zcodes': zcodes,
                    'scodes': scodes,
                    'samples': samples,
                    'log_p_x': log_p_x,
                    'log_p_x_missing': log_p_x_missing,
                    'loss_re' : loss_reconstruction,
                    'loss': -ELBO,
                    'optim': optim,
                    'KL_s': KL_s,
                    'KL_z': KL_z,
                    'p_params': p_params,
                    'q_params': q_params,
                    'samples_zgen': samples_zgen,
                    'test_params_zgen': test_params_zgen,
                    'log_p_x_zgen': log_p_x_zgen,
                    'log_p_x_missing_zgen': log_p_x_missing_zgen}

        # return HIVAE model architecture
        return tf_nodes
    

    # originally from helpers.py
    def train_network(self, settings):
        """
        Run HIVAE model training.
        Create a TensorFlow graph, run train and test batches for a predefined n_epoch.
        """

        # split the setting string and convert it into a series of arguments
        argvals = settings.split()
        args = self.getArgs(argvals)

        # create a directory for the files to be saved 
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name = f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
        load_file_name = f"{self.saved_networks}{re.sub('_BNet', '', args.save_file)}/{re.sub('_BNet', '', args.save_file)}.ckpt" 

        # create a TF graph for HIVAE
        sess_HVAE = tf.Graph()
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, beta1=args.adam_beta_1, beta2=args.adam_beta_2, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)

        ################### Running the VAE Training #################################

        # load data, types, missing and true missing masks as variables
        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        
        # get an integer number of train batches
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))

        # compute a final missing mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)

        # run a HIVAE session
        with tf.compat.v1.Session(graph=sess_HVAE) as session:
            # option to restore all the variables
            saver = tf.compat.v1.train.Saver()
            if args.restore == 1:
                saver.restore(session, load_file_name)
                print(f"[*] Model restored: {load_file_name}")
            else:
                print("[*] Initizalizing Variables ...")
                tf.compat.v1.global_variables_initializer().run()

            # define variables with default values
            loss_epoch = []

            # for every epoch
            for epoch in range(args.epochs):
                # define variables with default values
                avg_loss = 0.
                avg_KL_s = 0.
                avg_KL_z = 0.
                samples_list = []
                p_params_list = []
                q_params_list = []
                log_p_x_total = []
                log_p_x_missing_total = []

                # annealing of Gumbel-Softmax parameter
                tau = np.max([1.0 - (0.999/(args.epochs-50+0.0001))*epoch, 1e-3]) # add 0.0001 to avoid zero division error

                # randomize the data in the mini-batches
                random_perm = np.random.RandomState(seed=self.seed).permutation(range(np.shape(train_data)[0]))
                train_data_aux = train_data[random_perm, :]
                miss_mask_aux = miss_mask[random_perm, :]

                # for each data batch
                for i in range(n_batches):
                    # prepares batch data and miss list for the feed dictionary
                    data_list, miss_list = self.next_batch(train_data_aux, types_dict, miss_mask_aux, args.batch_size, index_batch=i) 

                    # delete unknown data (input zeros)
                    data_list_observed = [data_list[i] * np.reshape(miss_list[:,i], [args.batch_size,1]) for i in range(len(data_list))] 

                    # create feed dictionary
                    feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                    feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                    feedDict[tf_nodes['miss_list']] = miss_list 
                    feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape) # only works when running all 1 batch 1 epoch
                    feedDict[tf_nodes['tau_GS']] = tau
                    feedDict[tf_nodes['zcodes']] = np.ones(args.batch_size).reshape((args.batch_size, 1)) # just for placeholder
                    feedDict[tf_nodes['scodes']] = np.ones(args.batch_size).reshape((args.batch_size, 1)) # just for placeholder

                    # run HIVAE (with training data)
                    _, loss, KL_z, KL_s, samples, log_p_x, log_p_x_missing, p_params, q_params = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'], tf_nodes['samples'], tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['p_params'], tf_nodes['q_params']], feed_dict=feedDict)

                    # collect all samples, distirbution parameters and logliks in lists
                    samples_list.append(samples)
                    p_params_list.append(p_params)
                    q_params_list.append(q_params)
                    log_p_x_total.append(log_p_x)
                    log_p_x_missing_total.append(log_p_x_missing)

                    # compute average loss, average KL s, and average KL z
                    avg_loss += np.mean(loss)
                    avg_KL_s += np.mean(KL_s)
                    avg_KL_z += np.mean(KL_z)

                # append -avg_loss for all epochs
                print(f"[*] Epoch: {epoch}, Reconstruction Loss: {avg_loss/n_batches}, KL s: {avg_KL_s/n_batches}, KL z: {avg_KL_z/n_batches}")
                loss_epoch.append(-avg_loss/n_batches)

                # save session if predefined to save
                if epoch % args.save == 0:
                    print(f"[*] Saving Variables: {network_file_name}")  
                    save_path = saver.save(session, network_file_name)    

            # create a plot of reconstruction loss over all epochs and save the figure
            print("[*] Training Finished ...")
            plt.clf()
            plt.figure()
            plt.plot(loss_epoch)
            plt.xlabel('Epoch')
            plt.ylabel('Reconstruction loss')  # we already handled the x-label with ax1
            plt.title(args.save_file)
            plt.savefig(f"{self.train_stats}{args.save_file}.png", bbox_inches='tight')
    

    # originally from VAE_functions.py
    def place_holder_types(self, types_file, batch_size):
        """
        Define placeholders for types file.
        Return all placeholders.
        """

        # read the types of the data from types_file csv
        with open(types_file) as f:
            types_list = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
            
        # create placeholders for every data type, with appropriate dimensions
        batch_data_list = []
        for i in range(len(types_list)):
            batch_data_list.append(tf.compat.v1.placeholder(shape=[batch_size, int(types_list[i]['dim'])], dtype=tf.float32))
        tf.concat(batch_data_list, axis=1)
        
        # create placeholders for every missing data type, with appropriate dimensions
        batch_data_list_observed = []
        for i in range(len(types_list)):
            batch_data_list_observed.append(tf.compat.v1.placeholder(shape=[batch_size, int(types_list[i]['dim'])], dtype=tf.float32))
        tf.concat(batch_data_list_observed, axis=1)
            
        # create placeholders for the missing data indicator variable
        miss_list = tf.compat.v1.placeholder(shape=[batch_size, len(types_list)], dtype=tf.int32)
        miss_list_VP = tf.compat.v1.placeholder(shape=[batch_size, len(types_list)], dtype=tf.int32)
        
        # create placeholders for Gumbel-softmax parameters
        tau = tf.compat.v1.placeholder(shape=[], dtype=tf.float32)
        zcodes=tf.compat.v1.placeholder(shape=[batch_size,1], dtype=tf.float32)
        scodes=tf.compat.v1.placeholder(shape=[batch_size,1], dtype=tf.int32)
        
        # return all placeholders
        return batch_data_list, batch_data_list_observed, miss_list, miss_list_VP, tau, types_list, zcodes, scodes


    # originally from VAE_functions.py
    def batch_normalization(self, batch_data_list, types_list, miss_list):
        """
        Perform batch normalization with the data.
        Return batch-normed data and their mean and variance.
        """
        
        # define empty lists
        normalized_data = []
        normalization_parameters = []
        
        # for every item of the batch_data_list
        for i, d in enumerate(batch_data_list):
            
            # partition the data in missing data (0) and observed data n(1)
            missing_data, observed_data = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
            
            # if the data type = real
            if types_list[i]['type'] == 'real':
                
                # transform the data to a gaussian with mean 0 and std 1
                data_mean, data_var = tf.nn.moments(observed_data, 0)

                # clip data range to avoid zero values
                data_var = tf.clip_by_value(data_var, 1e-6, 1e20) 

                # batch norm data
                aux_X = tf.nn.batch_normalization(observed_data, data_mean, data_var, offset=0.0, scale=1.0, variance_epsilon=1e-6)
                
                # append batch normed data and its mean and variance 
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
                normalization_parameters.append([data_mean, data_var])
                
            # if the data type = positive real
            elif types_list[i]['type'] == 'pos':
                
                # use log-normal: transform the log of the data to a gaussian with mean 0 and std 1
                observed_data_log = tf.math.log(1 + observed_data)
                data_mean_log, data_var_log = tf.nn.moments(observed_data_log, 0)

                # clip data range to avoid zero values
                data_var_log = tf.clip_by_value(data_var_log, 1e-6, 1e20) 

                # batch norm data
                aux_X = tf.nn.batch_normalization(observed_data_log, data_mean_log, data_var_log, offset=0.0, scale=1.0, variance_epsilon=1e-6)
                
                # append batch normed data and its mean and variance 
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
                normalization_parameters.append([data_mean_log, data_var_log])
            
            # if the data type = count
            elif types_list[i]['type'] == 'count':
                
                # use the log of the data
                aux_X = tf.math.log(observed_data)
                
                # append batch normed data and its mean and variance 
                # mean and variance are 0.0 and 1.0
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
                normalization_parameters.append([0.0, 1.0])
            
            # if the data type is other than real, pos real and count (expect categorical and ordinal)
            else:
                # don't normalize the categorical and ordinal variables, append as is
                # no normalization here
                normalized_data.append(d)
                normalization_parameters.append([0.0, 1.0]) #No normalization here
        
        # return batch-normed data and their mean and variance
        return normalized_data, normalization_parameters


    # originally from model_HIVAE_inputDropout.py
    def encoder(self, X_list, batch_size, z_dim, s_dim, tau, weight_decay):
        """
        Create the encoder part of HIVAE model architecture.
        Return encoder features.
        """
        
        # define samples and q_params variables
        samples = dict.fromkeys(['s','z','y','x'], [])
        q_params = dict()

        # concatenate X_list with 1
        X = tf.concat(X_list, 1)
        
        # create the proposal of q(s|x^o)
        samples['s'], q_params['s'] = self.s_proposal_multinomial(X, batch_size, s_dim, tau, weight_decay, reuse=None)
        
        # create the proposal of q(z|s,x^o)
        samples['z'], q_params['z'] = self.z_proposal_GMM(X, samples['s'], batch_size, z_dim, weight_decay, reuse=None)
        
        # return encoder features
        return samples, q_params


    # originally from model_HIVAE_inputDropout.py
    def decoder(self, batch_data_list, miss_list, types_list, samples, normalization_params, batch_size, z_dim, y_dim, y_dim_partition, weight_decay):
        """
        Create the decoder part of HIVAE model architecture.
        Return decoder features.
        """

        # define p_params
        p_params = dict()
        
        # create the distribution of p(z|s)
        p_params['z'] = self.z_distribution_GMM(samples['s'], z_dim, weight_decay, reuse=None)
        
        # create deterministic layer y
        samples['y'] = tf.compat.v1.layers.dense(inputs=samples['z'], units=y_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name= 'layer_h1_', reuse=None)
 
        grouped_samples_y = self.y_partition(samples['y'], types_list, y_dim_partition)

        # compute the parameters h_y
        theta = self.theta_estimation_from_y(grouped_samples_y, types_list, miss_list, batch_size, weight_decay, reuse=None)
        
        # compute loglik and output of the VAE
        log_p_x, log_p_x_missing, samples['x'], p_params['x'] = self.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params)
        
        # return decoder features
        return theta, samples, p_params, log_p_x, log_p_x_missing


    # originally from model_HIVAE_inputDropout.py
    def cost_function(self, log_p_x, p_params, q_params, z_dim, s_dim):
        """
        Define the cost function of HIVAE model architecture.
        Return ELBO, reconstruction loss, KL z, and KL s.
        """

        # KL(q(s|x)|p(s))
        log_pi = q_params['s']
        pi_param = tf.nn.softmax(log_pi)
        KL_s = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=pi_param) + tf.math.log(float(s_dim))
        
        # KL(q(z|s,x)|p(z|s))
        mean_pz, log_var_pz = p_params['z']
        mean_qz, log_var_qz = q_params['z']
        KL_z = -0.5*z_dim + 0.5*tf.reduce_sum(tf.exp(log_var_qz-log_var_pz) + tf.square(mean_pz-mean_qz)/tf.exp(log_var_pz)-log_var_qz+log_var_pz, 1)
        
        # Eq[log_p(x|y)]
        loss_reconstruction = tf.reduce_sum(log_p_x, 0)
        
        # complete ELBO
        ELBO = tf.reduce_mean(loss_reconstruction - KL_z - KL_s, 0)
        
        # return ELBO, reconstruction loss, KL z, and KL s
        return ELBO, loss_reconstruction, KL_z, KL_s
    

    # originally from model_HIVAE_inputDropout.py
    def fixed_decoder(self, batch_data_list, X_list, miss_list_VP, miss_list, types_list, batch_size, y_dim, y_dim_partition, s_dim, normalization_params, zcodes, scodes, weight_decay):
        """
        Create the decoder part of HIVAE model architecture (with fixed z).
        Return decoder features.
        """

        # define test samples and test params
        samples_test = dict.fromkeys(['s','z','y','x'], [])
        test_params = dict()

        # concatenate X_list with value 1
        X = tf.concat(X_list, 1)
        
        # create the proposal of q(s|x^o)
        samples_test['s'] = tf.one_hot(scodes, depth=s_dim)
        
        # set fixed z
        samples_test['z'] = zcodes
        
        # create deterministic layer y
        samples_test['y'] = tf.compat.v1.layers.dense(inputs=samples_test['z'], units=y_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name= 'layer_h1_', reuse=True)

        grouped_samples_y = self.y_partition(samples_test['y'], types_list, y_dim_partition)
        
        # compute the parameters h_y
        theta = self.theta_estimation_from_y(grouped_samples_y, types_list, miss_list_VP, batch_size, weight_decay, reuse=True)
        
        # compute loglik and output of the VAE
        log_p_x, log_p_x_missing, samples_test['x'], test_params['x'] = self.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params)
        
        # return decoder features
        return samples_test, test_params, log_p_x, log_p_x_missing


    # originally from read_functions.py
    def next_batch(self, data, types_dict, miss_mask, batch_size, index_batch):
        """
        Transform data into mini-batches.
        Return batch data and miss list.
        """
        
        # create minibatch
        batch_xs = data[index_batch*batch_size:(index_batch+1)*batch_size, :]
        
        # slipt variables of the batches
        data_list = []
        initial_index = 0
        for d in types_dict:
            dim = int(d['dim'])
            data_list.append(batch_xs[:, initial_index:initial_index+dim])
            initial_index += dim
        
        # create batch missing data
        miss_list = miss_mask[index_batch*batch_size:(index_batch+1)*batch_size, :]

        # return batch data and miss list
        return data_list, miss_list
    

    # originally from helpers.py
    def enc_network(self, settings):
        """
        Execute the encoder part of HIVAE 1 time.
        Get s and z samples as embeddings, as well as the original dataframe (with relevelled factors & NA's=0).
        Return the deterministic and sampled s and z codes and the reconstructed dataframe (now imputed).
        """
        
        # split the setting string and convert it into a series of arguments
        argvals = settings.split()
        args = self.getArgs(argvals)

        # create a directory for the files to be saved 
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name = f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
        
        # create a TF graph for HIVAE
        sess_HVAE = tf.Graph()
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, beta1=args.adam_beta_1, beta2=args.adam_beta_2, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)
        
        ################### Running the VAE Training #################################

        # load data, types, missing and true missing masks as variables
        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        
        # get an integer number of batches
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
        
        # compute a final missing mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)

        # run a HIVAE session
        with tf.compat.v1.Session(graph=sess_HVAE) as session:
            # restore all the variables
            saver = tf.compat.v1.train.Saver()
            saver.restore(session, network_file_name)
            print(f"[*] Model restored: {network_file_name}")
            print('::::::ENCODING:::::::::')

            # define variables with default values
            avg_loss = 0.
            avg_KL_s = 0.
            avg_KL_z = 0.
            samples_list = []
            q_params_list = []

            # constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3

            # for each data batch
            for i in range(n_batches):      
                
                # prepares batch data and miss list for the feed dictionary
                data_list, miss_list = self.next_batch(train_data, types_dict, miss_mask, args.batch_size, index_batch=i) 
                
                # delete unknown data (input zeros)
                data_list_observed = [data_list[i] * np.reshape(miss_list[:,i], [args.batch_size,1]) for i in range(len(data_list))]

                # create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape) # unused
                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['zcodes']] = np.ones(args.batch_size).reshape((args.batch_size, 1))
                feedDict[tf_nodes['scodes']] = np.ones(args.batch_size).reshape((args.batch_size, 1))

                # run HIVAE (with training data)
                KL_s, loss, samples, log_p_x, log_p_x_missing, loss_total, KL_z, p_params, q_params = session.run([tf_nodes['KL_s'], tf_nodes['loss_re'], tf_nodes['samples'], tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'], tf_nodes['loss'], tf_nodes['KL_z'], tf_nodes['p_params'], tf_nodes['q_params']], feed_dict=feedDict)

                # append encoded outputs
                samples_list.append(samples)
                q_params_list.append(q_params)

                # compute average loss
                avg_loss += np.mean(loss)
                avg_KL_s += np.mean(KL_s)
                avg_KL_z += np.mean(KL_z)

            # transform discrete variables to original values (this is for getting the original data frame)
            train_data_transformed = self.discrete_variables_transformation(train_data, types_dict)

            # create global dictionary of the distribution parameters
            q_params_complete = self.q_distribution_params_concatenation(q_params_list)

            # return the deterministic and sampled s and z codes and the reconstructed dataframe (now imputed)
            encs = np.argmax(q_params_complete['s'], 1)
            encz = q_params_complete['z'][0, :, :]
            return [encs, encz, train_data_transformed]
    

    # originally from helpers.py
    def dec_network(self, settings, zcodes, scodes, VP=False):
        """
        Execute the decoder part of HIVAE 1 time.
        Decode using set s and z values (if generated provide a generated miss_list). 
        Return decoded data.
        """
        
        # split the setting string and convert it into a series of arguments
        argvals = settings.split()
        args = self.getArgs(argvals)

        # create a directory for the save file
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name=f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
        
        # create a TF graph for HIVAE
        sess_HVAE = tf.Graph()
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, beta1=args.adam_beta_1, beta2=args.adam_beta_2, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)
        
        ################### Running the VAE Training #################################

        # load data, types, missing and true missing masks as variables
        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        
        # get an integer number of batches
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
        
        # compute a final missing mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)
        
        # run a HIVAE session
        with tf.compat.v1.Session(graph=sess_HVAE) as session:
            # restore all the variables
            saver = tf.compat.v1.train.Saver()
            saver.restore(session, network_file_name)
            print(f"[*] Model restored: {network_file_name}")
            print('::::::DECODING:::::::::')

            # define variable with default value
            samples_list = []

            # constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3

            # for each data batch
            for i in range(n_batches):  

                # prepares batch data and miss list for the feed dictionary
                data_list, miss_list = self.next_batch(train_data, types_dict, miss_mask, args.batch_size, index_batch=i) 
                
                # delete unknown data (input zeros)
                data_list_observed = [data_list[i] * np.reshape(miss_list[:,i], [args.batch_size,1]) for i in range(len(data_list))]

                # create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list

                if VP:
                    sub = re.sub(f"{self.data_python}|.csv", "", args.data_file)
                    vpfile = f"{self.VP_misslist}{sub}_vpmiss.csv"
                    print(f"[*] ::::::::::::{vpfile}")
                    feedDict[tf_nodes['miss_list_VP']] = pd.read_csv(vpfile, header=None)
                elif VP == "nomiss":
                    print("[*] :::::::::::: ones for miss list VP")
                    feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape)
                else:
                    feedDict[tf_nodes['miss_list_VP']] = miss_list

                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['zcodes']] = np.array(zcodes).reshape((len(zcodes), 1))
                feedDict[tf_nodes['scodes']] = np.array(scodes).reshape((len(scodes), 1))

                # get samples from the fixed decoder function
                samples_zgen, log_p_x_test, log_p_x_missing_test, test_params  = session.run([tf_nodes['samples_zgen'], tf_nodes['log_p_x_zgen'], tf_nodes['log_p_x_missing_zgen'], tf_nodes['test_params_zgen']], feed_dict=feedDict)
                
                # append encoded outputs
                samples_list.append(samples_zgen)

            # separate the samples from the batch list
            s_aux, z_aux, y_total, est_data = self.samples_concatenation(samples_list)

            # transform discrete variables to original values
            est_data_transformed = self.discrete_variables_transformation(est_data, types_dict)
            print(f"[*] est data: {est_data}")
            print(f"[*] types_dict: {types_dict}")

            # return decoded data
            return est_data_transformed
    

    # originally from helpers.py
    def dec_network_loglik(self, settings, zcodes, scodes, VP=False):
        """
        Execute the decoder part of HIVAE with log-likelihood.
        Decode using set s and z values (if generated provide a generated miss_list). 
        Return decoded data.
        """
        
        # split the setting string and convert it into a series of arguments
        argvals = settings.split()
        args = self.getArgs(argvals)

        # create a directory for the save file
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name = f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
    
        # create a TF graph for HIVAE
        sess_HVAE = tf.Graph()
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, beta1=args.adam_beta_1, beta2=args.adam_beta_2, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)
        
        ################### Running the VAE Training #################################

        # load data, types, missing and true missing masks as variables
        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        
        # get an integer number of batches
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
        
        # compute a final missing mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)
        
        # run a HIVAE session
        with tf.compat.v1.Session(graph=sess_HVAE) as session:
            # restore all the variables
            saver = tf.compat.v1.train.Saver()
            saver.restore(session, network_file_name)
            print(f"[*] Model restored: {network_file_name}")
            print('[*] ::::::DECODING:::::::::')
            
            # define variable with default value
            samples_list = []

            # constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3

            # for each data batch
            for i in range(n_batches):      

                # prepares batch data and miss list for the feed dictionary
                data_list, miss_list = self.next_batch(train_data, types_dict, miss_mask, args.batch_size, index_batch=i) 
                
                # delete unknown data (input zeros)
                data_list_observed = [data_list[i] * np.reshape(miss_list[:,i], [args.batch_size,1]) for i in range(len(data_list))]

                # create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list

                if VP:
                    sub = re.sub(f"{self.data_python}|.csv", "", args.data_file)
                    vpfile = f"{self.VP_misslist}{sub}_vpmiss.csv"
                    print(f"[*] ::::::::::::{vpfile}")
                    feedDict[tf_nodes['miss_list_VP']] = pd.read_csv(vpfile, header=None)
                elif VP == "nomiss":
                    print(":::::::::::: ones for miss list VP")
                    feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape)
                else:
                    feedDict[tf_nodes['miss_list_VP']] = miss_list

                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['zcodes']] = np.array(zcodes).reshape((len(zcodes), 1))
                feedDict[tf_nodes['scodes']] = np.array(scodes).reshape((len(scodes), 1))

                # get samples from the fixed decoder function
                samples_zgen, log_p_x_test, log_p_x_missing_test, test_params  = session.run([tf_nodes['samples_zgen'], tf_nodes['log_p_x_zgen'], tf_nodes['log_p_x_missing_zgen'], tf_nodes['test_params_zgen']], feed_dict=feedDict)
                
                # append encoded outputs
                samples_list.append(samples_zgen)

            # return decoded data
            return log_p_x_test
    

    # originally from helpers.py
    def merge_dat(self, lis):
        """
        Merge all dataframes in a list on SUBJID.
        Return merged dataframe.
        """
        
        df = lis[0]
        for x in lis[1:]:
            df = pd.merge(df, x, on = 'SUBJID')
        return df
    

    # originally from read_functions.py
    def discrete_variables_transformation(self, data, types_dict):
        """
        Transform variables with 'cat' and 'ordinal' type.
        Return transformed data output.
        """
        
        # define empty variable and list
        ind_ini = 0
        output = []

        # for every item in types_dict
        for d in range(len(types_dict)):

            # determine end index
            ind_end = ind_ini + int(types_dict[d]['dim'])

            # if type is categorical, do argmax on the data chunk and reshape
            if types_dict[d]['type'] == 'cat':
                output.append(np.reshape(np.argmax(data[:,ind_ini:ind_end],1), [-1,1]))
            
            # if type is ordinal, do sum on the data chunk and reshape
            elif types_dict[d]['type'] == 'ordinal':
                output.append(np.reshape(np.sum(data[:,ind_ini:ind_end],1)-1, [-1,1]))
            
            # if type is neither categorical or ordinal, process the data chunk as is
            else:
                output.append(data[:, ind_ini:ind_end])
            
            # update initial index accordingly
            ind_ini = ind_end
        
        # return transformed data output
        return np.concatenate(output, 1)
    

    # originally from read_functions.py
    def q_distribution_params_concatenation(self, params):
        """
        Concatenate z and s params.
        Return z and s params as dict.
        """
        
        # define empty variable and dict
        keys = params[0].keys()
        out_dict = {key: [] for key in keys}
        
        # add key and batch data as items into the dict
        for i, batch in enumerate(params):
            for d, k in enumerate(keys):
                out_dict[k].append(batch[k])

        # concatenate z and s values
        out_dict['z'] = np.concatenate(out_dict['z'], 1)
        out_dict['s'] = np.concatenate(out_dict['s'], 0)
        
        # return z and s params as dict
        return out_dict
    

    # originally from read_functions.py
    def samples_concatenation(self, samples):
        """
        Concatenate x, y, z and s samples.
        Return samples s, z, y and x.
        """
        
        # for each sample
        for i, batch in enumerate(samples):

            # for the first loop, add values
            if i == 0:
                samples_x = np.concatenate(batch['x'], 1)
                samples_y = batch['y']
                samples_z = batch['z']
                samples_s = batch['s']
            
            # for the second loop onwards, concatenate values
            else:
                samples_x = np.concatenate([samples_x,np.concatenate(batch['x'],1)], 0)
                samples_y = np.concatenate([samples_y,batch['y']], 0)
                samples_z = np.concatenate([samples_z,batch['z']], 0)
                samples_s = np.concatenate([samples_s,batch['s']], 0)
        
        # return samples s, z, y and x
        return samples_s, samples_z, samples_y, samples_x


    # originally from VAE_functions.py
    def s_proposal_multinomial(self, X, batch_size, s_dim, tau, weight_decay, reuse):
        """
        Create the proposal of q(s|x^o).
        Return samples_s and log_pi.
        """
        
        # propose a categorical distribution to create a GMM for the latent space z
        log_pi = tf.compat.v1.layers.dense(inputs=X, units=s_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='layer_1_enc_s', reuse=reuse)

        # gumbel-softmax trick
        U = -tf.math.log(-tf.math.log(tf.random.uniform([batch_size,s_dim], seed=self.seed)))
        samples_s = tf.nn.softmax((log_pi + U)/tau)
        
        # return samples_s and log_pi
        return samples_s, log_pi


    # originally from VAE_functions.py
    def z_proposal_GMM(self, X, samples_s, batch_size, z_dim, weight_decay, reuse):
        """
        Create the proposal of q(z|s,x^o).
        Return samples_z and [mean_qz, log_var_qz].
        """
        
        # propose a GMM for z
        mean_qz = tf.compat.v1.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='layer_1_mean_enc_z', reuse=reuse)

        log_var_qz = tf.compat.v1.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay),  name='layer_1_logvar_enc_z', reuse=reuse)
        
        # avoid numerical problems by clipping value range
        log_var_qz = tf.clip_by_value(log_var_qz, -15.0, 15.0)
        
        # reparameterization trick
        eps = tf.random.normal((batch_size, z_dim), 0, 1, dtype=tf.float32, seed=self.seed)
        samples_z = mean_qz + tf.multiply(tf.exp(log_var_qz/2), eps)
        
        # return samples_z and [mean_qz, log_var_qz]
        return samples_z, [mean_qz, log_var_qz]


    # originally from VAE_functions.py
    def z_distribution_GMM(self, samples_s, z_dim, weight_decay, reuse):
        """
        Create the distribution of p(z|s).
        Return mean_pz and log_var_pz.
        """
        
        # propose a GMM for z
        mean_pz = tf.compat.v1.layers.dense(inputs=samples_s, units=z_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name= 'layer_1_mean_dec_z', reuse=reuse)

        log_var_pz = tf.zeros([tf.shape(samples_s)[0], z_dim])
        
        # avoid numerical problems by clipping value range
        log_var_pz = tf.clip_by_value(log_var_pz, -15.0, 15.0)
        
        # return mean_pz and log_var_pz
        return mean_pz, log_var_pz


    # originally from VAE_functions.py
    def y_partition(self, samples_y, types_list, y_dim_partition):
        """
        Perform data partitioning of variable y.
        Return partitioned y data.
        """
        
        # define empty list
        grouped_samples_y = []
        # the first element must be 0 and the length of the partition vector must be len(types_dict)+1
        if len(y_dim_partition) != len(types_list):
            raise Exception("[*] The length of the partition vector must match the number of variables in the data + 1")
            
        # insert a 0 at the beginning of the cumsum vector
        # perform data partitioning
        partition_vector_cumsum = np.insert(np.cumsum(y_dim_partition), 0, 0)
        for i in range(len(types_list)):
            grouped_samples_y.append(samples_y[:, partition_vector_cumsum[i]:partition_vector_cumsum[i+1]])
        
        # return partitioned y data
        return grouped_samples_y


    # originally from VAE_functions.py
    def theta_estimation_from_y(self, samples_y, types_list, miss_list, batch_size, weight_decay, reuse):
        """
        Compute the parameters h_y.
        Return the parameter h(y).
        """

        # define empty list
        theta = []

        # independent yd -> compute p(xd|yd)
        for i, d in enumerate(samples_y):
            
            # partition the data in missing data (0) and observed data (1)
            missing_y, observed_y = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
            
            # different layer models for each type of variable
            if types_list[i]['type'] == 'real':
                params = self.theta_real(observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse)
            
            elif types_list[i]['type'] == 'pos':
                params = self.theta_pos(observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse)
                
            elif types_list[i]['type'] == 'count':
                params = self.theta_count(observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse)
            
            elif types_list[i]['type'] == 'cat':
                params = self.theta_cat(observed_y, missing_y, condition_indices, types_list, batch_size, i, weight_decay, reuse)
                
            elif types_list[i]['type'] == 'ordinal':
                params = self.theta_ordinal(observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse)

            # append all model layers  
            theta.append(params)

        # return the parameter h(y)
        return theta


    # originally from VAE_functions.py
    def loglik_evaluation(self, batch_data_list, types_list, miss_list, theta, normalization_params):
        """
        Compute log-likelihood log(p(xd|yd)).
        Return all log-likelihood values.
        """

        # define empty lists
        log_p_x = []
        log_p_x_missing = []
        samples_x = []
        params_x = []
        
        # independent yd -> compute log(p(xd|yd))
        for i, d in enumerate(batch_data_list):

            # select the likelihood for the types of variables
            loglik_models = c_loglik_models_missing_normalize([d, miss_list[:,i]], types_list[i], theta[i], normalization_params[i])
            loglik_function = getattr(loglik_models, 'loglik_' + types_list[i]['type'])
            out = loglik_function()
            
            # append all log-likelihood calculation
            log_p_x.append(out['log_p_x'])
            # test loglik element
            log_p_x_missing.append(out['log_p_x_missing']) 
            samples_x.append(out['samples'])
            params_x.append(out['params'])
        
        # return all log-likelihood values
        return log_p_x, log_p_x_missing, samples_x, params_x
    

    # originally from VAE_functions.py
    def theta_real(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        """
        Create a mean layer and a sigma layer for 'real' type data.
        Return the mean layer and the sigma layer.
        """
        
        # mean layer
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), weight_decay=weight_decay, reuse=reuse)
        
        # sigma Layer
        h2_sigma = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i), weight_decay=weight_decay, reuse=reuse)
        
        # return the mean and sigma layer
        return [h2_mean, h2_sigma]


    # originally from VAE_functions.py
    def theta_pos(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        """
        Create a mean layer and a sigma layer for 'positive real' type data.
        Return the mean layer and the sigma layer.
        """
        
        # mean layer
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), weight_decay=weight_decay, reuse=reuse)
        
        # sigma Layer
        h2_sigma = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i), weight_decay=weight_decay, reuse=reuse)
        
        # return the mean and sigma layer
        return [h2_mean, h2_sigma]


    # originally from VAE_functions.py
    def theta_count(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        """
        Create a mean layer and a sigma layer for 'count' type data.
        Return the lambda layer.
        """
        
        # lambda Layer
        h2_lambda = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), weight_decay=weight_decay, reuse=reuse)
        
        # return the lambda layer
        return h2_lambda


    # originally from VAE_functions.py
    def theta_cat(self, observed_y, missing_y, condition_indices, types_list, batch_size, i, weight_decay, reuse):
        """
        Create a mean layer and a sigma layer for 'categorical' type data.
        Return the log_pi layer.
        """

        # log pi layer, with zeros in the first value to avoid the identificability problem
        h2_log_pi_partial = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, name='layer_h2' + str(i), weight_decay=weight_decay, reuse=reuse)
        h2_log_pi = tf.concat([tf.zeros([batch_size,1]), h2_log_pi_partial], 1)
        
        # return the log_pi layer
        return h2_log_pi
    

    # originally from VAE_functions.py
    def theta_ordinal(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        """
        Create a mean layer and a sigma layer for 'ordinal' type data.
        Return the theta layer and the mean layer.
        """

        # theta layer, dimension of ordinal - 1
        h2_theta = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, name='layer_h2' + str(i), weight_decay=weight_decay, reuse=reuse)
        
        # mean layer, a single value
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=1, name='layer_h2_sigma' + str(i), weight_decay=weight_decay, reuse=reuse)
        
        # return the theta layer and the mean layer
        return [h2_theta, h2_mean]
    

    # originally from VAE_functions.py
    def observed_data_layer(self, observed_data, missing_data, condition_indices, output_dim, name, weight_decay, reuse):
        """
        Create a dense layer of observed data and a dense layer of missing data.
        Return a combined dense layer.
        """

        # train a layer with the observed data and reuse it for the missing data
        obs_output = tf.compat.v1.layers.dense(inputs=observed_data, units=output_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=name, reuse=reuse, trainable=True)

        miss_output = tf.compat.v1.layers.dense(inputs=missing_data, units=output_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=name, reuse=True, trainable=False)

        # join back the data
        output = tf.dynamic_stitch(condition_indices, [miss_output,obs_output])
        
        # return a combined dense layer
        return output
