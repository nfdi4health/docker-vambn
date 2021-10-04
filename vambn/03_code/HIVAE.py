
import os
import re
import time
import csv
import argparse
import itertools
from operator import itemgetter 
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import tensorflow
import tensorflow.compat.v1 as tf
import configparser


# originally from loglik_models_missing_normalize
class c_loglik_models_missing_normalize:

    def __init__(self, batch_data, list_type, theta, normalization_params):
        self.batch_data = batch_data
        self.list_type = list_type
        self.theta = theta
        self.normalization_params = normalization_params

    def loglik_real(self):
        output = dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        #Data outputs
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask, tf.float32)
        
        data_mean, data_var = self.normalization_params
        data_var = tf.clip_by_value(data_var, epsilon, np.inf)
        
        est_mean, est_var = self.theta
        est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1.0) #Must be positive
        
        # Affine transformation of the parameters
        est_mean = tf.sqrt(data_var)*est_mean + data_mean
        est_var = data_var*est_var
        
        #Compute loglik
        log_p_x = -0.5 * tf.reduce_sum(tf.squared_difference(data,est_mean)/est_var,1) \
            - int(self.list_type['dim'])*0.5*tf.log(2*np.pi) - 0.5*tf.reduce_sum(tf.log(est_var),1)
        
        #Outputs
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = [est_mean, est_var]
        output['samples'] = tf.compat.v1.distributions.Normal(est_mean,tf.sqrt(est_var)).sample()
            
        return output

    def loglik_pos(self):
        #Log-normal distribution
        output = dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        #Data outputs
        data_mean_log, data_var_log = self.normalization_params
        data_var_log = tf.clip_by_value(data_var_log, epsilon, np.inf)
        
        data, missing_mask = self.batch_data
        data_log = tf.log(1.0 + data)
        missing_mask = tf.cast(missing_mask, tf.float32)
        
        est_mean, est_var = self.theta
        est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1.0)
        
        # Affine transformation of the parameters
        est_mean = tf.sqrt(data_var_log)*est_mean + data_mean_log
        est_var = data_var_log*est_var
        
        #Compute loglik
        log_p_x = -0.5 * tf.reduce_sum(tf.squared_difference(data_log,est_mean)/est_var,1) \
            - 0.5*tf.reduce_sum(tf.log(2*np.pi*est_var),1) - tf.reduce_sum(data_log,1)
        
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = [est_mean, est_var]
        output['samples'] = tf.exp(tf.compat.v1.distributions.Normal(est_mean,tf.sqrt(est_var)).sample()) - 1.0
            
        return output

    def loglik_cat(self):
        output=dict()
        
        #Data outputs
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask, tf.float32)
        
        log_pi = self.theta
        
        #Compute loglik
        log_p_x = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=data)
        
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = log_pi
        output['samples'] = tf.one_hot(tf.compat.v1.distributions.Categorical(probs=tf.nn.softmax(log_pi)).sample(),depth=int(self.list_type['dim']))
        
        return output

    def loglik_ordinal(self):
        output=dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        #Data outputs
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask, tf.float32)
        batch_size = tf.shape(data)[0]
        
        #We need to force that the outputs of the network increase with the categories
        partition_param, mean_param = self.theta
        mean_value = tf.reshape(mean_param,[-1,1])
        theta_values = tf.cumsum(tf.clip_by_value(tf.nn.softplus(partition_param), epsilon, 1e20), 1)
        sigmoid_est_mean = tf.nn.sigmoid(theta_values - mean_value)
        mean_probs = tf.concat([sigmoid_est_mean,tf.ones([batch_size,1],tf.float32)],1) - tf.concat([tf.zeros([batch_size,1],tf.float32),sigmoid_est_mean], 1)
        
        #Code needed to compute samples from an ordinal distribution
        true_values = tf.one_hot(tf.reduce_sum(tf.cast(data,tf.int32),1)-1, int(self.list_type['dim']))
        
        #Compute loglik
        log_p_x = tf.log(tf.clip_by_value(tf.reduce_sum(mean_probs*true_values,1),epsilon,1e20))
        
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = mean_probs
        output['samples'] = tf.sequence_mask(1+tf.compat.v1.distributions.Categorical(logits=tf.log(tf.clip_by_value(mean_probs,epsilon,1e20))).sample(), int(self.list_type['dim']),dtype=tf.float32)
        
        return output

    def loglik_count(self):
        output=dict()
        epsilon = tf.constant(1e-6, dtype=tf.float32)
        
        #Data outputs
        data, missing_mask = self.batch_data
        missing_mask = tf.cast(missing_mask,tf.float32)
        
        est_lambda = self.theta
        est_lambda = tf.clip_by_value(tf.nn.softplus(est_lambda),epsilon,1e20)
        
        log_p_x = -tf.reduce_sum(tf.nn.log_poisson_loss(targets=data,log_input=tf.log(est_lambda),compute_full_loss=True),1)
        
        output['log_p_x'] = tf.multiply(log_p_x, missing_mask)
        output['log_p_x_missing'] = tf.multiply(log_p_x, 1.0-missing_mask)
        output['params'] = est_lambda
        output['samples'] = tf.compat.v1.distributions.Poisson(est_lambda).sample()
            
        return output


class c_HIVAE_Modelling:

    def __init__(self):
        plt.ioff()
        tf.disable_v2_behavior()
        cp = configparser.RawConfigParser()
        config_file = r'/vambn/02_config/config_python.txt'
        assert os.path.exists(config_file)
        cp.read(config_file)
        print(f'[*] Config file sections: {cp.sections()}')

        self.sample_size = int(cp.get('Modelling', 'sample_size')) 
        self.sds = [int(i) for i in cp.get('Modelling', 'sds').split(',')]
        self.data_python = cp.get('GridSearch', 'path_data_python') 
        self.python_names = cp.get('Modelling', 'path_python_names') 
        self.saved_networks = cp.get('Modelling', 'path_saved_networks') 
        self.train_stats = cp.get('Modelling', 'path_train_stats') 
        self.VP_misslist = cp.get('Modelling', 'path_VP_misslist') 
        self.results = cp.get('GridSearch', 'path_results') 
        self.metaenc = cp.get('Modelling', 'path_metaenc') 
        self.embedding_plot = cp.get('Modelling', 'path_embedding_plot') 
        self.reconRP = cp.get('Modelling', 'path_reconRP') 
        self.training_logliks = cp.get('Modelling', 'path_training_logliks') 
        self.virtual_ppts = cp.get('Modelling', 'path_virtual_ppts') 
        self.decoded_VP = cp.get('Modelling', 'path_decoded_VP') 
        self.virtual_logliks = cp.get('Modelling', 'path_virtual_logliks') 
        
    # originally from 1_ADNI_HIVAE_training.ipynb
    def train_HIVAE(self):
        # get file list
        files = [i for i in os.listdir(self.data_python) if not '_type' in i and not '_missing' in i and i not in '.DS_Store']
        files.sort()
        print("[*] Files: ")
        print(files)

        best_hyper = pd.read_csv(self.results)
        ls_hyper_files = best_hyper['files'].tolist()
        ls_hyper_files.sort()
        print("[*] Best hyper (current file): ")
        print(ls_hyper_files)

        #if any(files != best_hyper['files']):
        if files != ls_hyper_files:
            print("[*] ERROR!!")
        else:
            best_hyper['sdims'] = self.sds
        
        print("[*] Best hyperparameters (all):")
        print(best_hyper)

        # HIVAE training for all modules
        for f in files:
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            settings = self.set_settings(opts, modload=False, save=True)
            self.train_network(settings)
        print("[*] Model training done for all modules.")

        # get network embeddings
        dat=list()
        dfs=list()
        for f in files:
            # replace placeholders in template
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            #run
            encs, encz, d = self.enc_network(settings)

            # make deterministic embeddings
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv', '', f)}_subj.csv")['x']
            sc = pd.DataFrame({'scode_'+re.sub('.csv','',f) : pd.Series(np.array([i for i in encs])), 'SUBJID' : subj})
            zc = pd.DataFrame({'zcode_'+re.sub('.csv','',f) : pd.Series(np.array([i[0] for i in encz])), 'SUBJID' : subj})
            enc = pd.merge(sc, zc, on = 'SUBJID')
            
            # save out individual file's metadata
            enc.to_csv(f"{self.saved_networks}{re.sub('.csv','',f)}_meta.csv", index=False)
            dfs.append(enc)
            dat.append(d)

        # join metadata
        enc_vars = [pd.read_csv(f"{self.saved_networks}{re.sub('.csv','',f)}_meta.csv") for f in files]
        meta = self.merge_dat(enc_vars)
        meta[meta.columns[['Unnamed' not in i for i in meta.columns]]].to_csv(self.metaenc, index=False)

        metadata = meta[meta.columns.drop(list(meta.filter(regex='SUBJID|scode_')))]
        print('[*] Metadata:')
        print(metadata)

        # display embedding scatter matrix
        plt.clf()
        plt.figure()
        fig = scatter_matrix(metadata, figsize=[15, 15], marker=".", s=10, diagonal="kde")
        for ax in fig.ravel():
            ax.set_xlabel(re.sub('_VIS|zcode_','',ax.get_xlabel()), fontsize = 20, rotation = 90)
            ax.set_ylabel(re.sub('_VIS|zcode_','',ax.get_ylabel()), fontsize = 20, rotation = 90)
            
        plt.suptitle('HI-VAE embeddings (deterministic)',fontsize=20)
        plt.savefig(self.embedding_plot, bbox_inches='tight')

        # RP decoding (reconstruction)
        meta = pd.read_csv(self.metaenc)

        recon=list()
        recdfs=list()
        for f in files:
            # replace placeholders in template
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            #run
            zcodes = meta['zcode_'+re.sub('.csv','',f)]
            scodes = meta['scode_'+re.sub('.csv','',f)]
            rec = self.dec_network(settings, zcodes, scodes)
            recon.append(rec)
            
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_subj.csv")['x']
            names = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_cols.csv")['x']
            recd = pd.DataFrame(rec)
            recd.columns = names
            recd['SUBJID'] = subj
            recdfs.append(recd)

        data_recon = self.merge_dat(recdfs)
        data_recon.to_csv(self.reconRP, index=False)

        # get likelihoods
        meta = pd.read_csv(self.metaenc)

        dfs = list()
        for f in files:
            # replace placeholders in template
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            #run
            zcodes = meta['zcode_'+re.sub('.csv','',f)]
            scodes = meta['scode_'+re.sub('.csv','',f)]
            
            loglik = self.dec_network_loglik(settings, zcodes, scodes)
            loglik = np.nanmean(np.array(loglik).T, axis=1)
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_subj.csv")['x']
            dat = pd.DataFrame(loglik)
            dat.columns = [f]
            dat['SUBJID'] = subj
            dfs.append(dat)

        decoded = self.merge_dat(dfs)
        decoded.to_csv(self.training_logliks, index=False)

        print('[*] HI-VAE model training script completed.')

    # originally from 2_ADNI_HIVAE_VP_decoding_and_counterfactuals.ipynb
    def decode_HIVAE(self):
        # get file list
        files = [i for i in os.listdir(self.data_python) if not '_type' in i and not '_missing' in i and not '.DS_Store' in i]
        print(f"[*] File type: {type(files)}")

        files = sorted(files)
        print("[*] Files: ")
        print(files)

        best_hyper = pd.read_csv(self.results)
        print("[*] Best hyper (current): ")
        print(best_hyper['files'])

        if any(files!=best_hyper['files']):
            print("[*] ERROR!!")
        else:
            best_hyper['sdims'] = self.sds
        
        print("[*] Best hyperparameters:")
        print(best_hyper)

        # VP decoding
        VPcodes = pd.read_csv(self.virtual_ppts)

        dfs = list()
        virt = list()
        for f in files:
            # replace placeholders in template
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            #run
            zcodes= VPcodes['zcode_'+re.sub('.csv','',f)]
            scodes = VPcodes['scode_'+re.sub('.csv','',f)] if 'scode_'+re.sub('.csv','',f) in VPcodes.columns else np.zeros(zcodes.shape)
            print(f"[*] scodes = {scodes}")
          
            dec = self.dec_network(settings, zcodes, scodes, VP=True)
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_subj.csv")['x']
            names = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_cols.csv")['x']
            dat = pd.DataFrame(dec)
            dat.columns = names
            dat['SUBJID'] = subj
            virt.append(dec)
            dfs.append(dat)

        decoded = self.merge_dat(dfs)
        decoded.to_csv(self.decoded_VP, index=False)

        # get log likelihoods for R plot
        VPcodes = pd.read_csv(self.virtual_ppts)

        dfs = list()
        for f in files:
            # replace placeholders in template
            opts = dict(best_hyper[best_hyper['files'].copy()==f])
            opts['nbatch'].iloc[0] = self.sample_size
            settings = self.set_settings(opts, nepochs=1, modload=True, save=False)
            
            #run
            zcodes = VPcodes['zcode_'+re.sub('.csv','',f)]
            scodes = VPcodes['scode_'+re.sub('.csv','',f)] if 'scode_'+re.sub('.csv','',f) in VPcodes.columns else np.zeros(zcodes.shape)
                
            loglik = self.dec_network_loglik(settings, zcodes, scodes, VP=True)
            loglik = np.nanmean(np.array(loglik).T, axis=1)
            subj = pd.read_csv(f"{self.python_names}{re.sub('.csv','',f)}_subj.csv")['x']
            dat = pd.DataFrame(loglik)
            dat.columns = [f]
            dat['SUBJID'] = subj
            dfs.append(dat)

        decoded = self.merge_dat(dfs)
        decoded.to_csv(self.virtual_logliks, index=False)

        print('[*] HI-VAE model virtual patients decoding script completed.')

    # originally from 1_ADNI_HIVAE_training.ipynb
    # note: modload doesnt do anything right now, hardcoded in helpers.py
    def set_settings(self, opts, nepochs=500, modload=False, save=True): 
        'replace setting template placeholders with file info'

        inputf = re.sub('.csv','',opts['files'].iloc[0])
        missf = inputf + '_missing.csv'
        typef = inputf + '_types.csv'
        
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
                    --learning_rate {opts['lrates'].iloc[0]}"
        
        return settings
    
    # originally from helpers.py
    def train_network(self, settings):
        'run training (no output)'

        argvals = settings.split()
        args = self.getArgs(argvals)

        #Create a directoy for the save file
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name = f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
        load_file_name = f"{self.saved_networks}{re.sub('_BNet','',args.save_file)}/{re.sub('_BNet','',args.save_file)}.ckpt" 

        #Creating graph
        sess_HVAE = tf.Graph()
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, z_dim=args.dim_latent_z, 
                                       y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, y_dim_partition=args.dim_latent_y_partition)

        ################### Running the VAE Training #################################
        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))#Get an integer number of batches
        miss_mask = np.multiply(miss_mask, true_miss_mask)#Compute the real miss_mask

        with tf.Session(graph=sess_HVAE) as session:
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            if args.restore==1:
                saver.restore(session, load_file_name)
                print(f"[*] Model restored: {load_file_name}")
            else:
                print("[*] Initizalizing Variables ...")
                tf.global_variables_initializer().run()

            # Training cycle
            loss_epoch=[]
            for epoch in range(args.epochs):
                avg_loss = 0.
                avg_loss_reg = 0.
                avg_KL_s = 0.
                avg_KL_z = 0.
                samples_list = []
                p_params_list = []
                q_params_list = []
                log_p_x_total = []
                log_p_x_missing_total = []

                # Annealing of Gumbel-Softmax parameter
                tau = np.max([1.0 - (0.999/(args.epochs-50))*epoch, 1e-3])
                print(f"[*] tau = {tau}")

                #Randomize the data in the mini-batches
                random_perm = np.random.RandomState(seed=42).permutation(range(np.shape(train_data)[0]))
                train_data_aux = train_data[random_perm,:]
                miss_mask_aux = miss_mask[random_perm,:]

                for i in range(n_batches):
                    data_list, miss_list = self.next_batch(train_data_aux, types_dict, miss_mask_aux, args.batch_size, index_batch=i) #Create inputs for the feed_dict
                    data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))] #Delete not known data (input zeros)

                    #Create feed dictionary
                    feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                    feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                    feedDict[tf_nodes['miss_list']] = miss_list 
                    feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape) # only works when running all 1 batch 1 epoch
                    feedDict[tf_nodes['tau_GS']] = tau
                    feedDict[tf_nodes['zcodes']] = np.ones(args.batch_size).reshape((args.batch_size,1)) # just for placeholder
                    feedDict[tf_nodes['scodes']] = np.ones(args.batch_size).reshape((args.batch_size,1)) # just for placeholder

                    #Running VAE
                    _,loss,KL_z,KL_s,samples,log_p_x,log_p_x_missing,p_params,q_params,loss_reg  = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], 
                                                                                                                tf_nodes['KL_s'], tf_nodes['samples'], tf_nodes['log_p_x'], 
                                                                                                                tf_nodes['log_p_x_missing'],tf_nodes['p_params'],
                                                                                                                tf_nodes['q_params'],tf_nodes['loss_reg']],feed_dict=feedDict)

                    #Collect all samples, distirbution parameters and logliks in lists
                    samples_list.append(samples)
                    p_params_list.append(p_params)
                    q_params_list.append(q_params)
                    log_p_x_total.append(log_p_x)
                    log_p_x_missing_total.append(log_p_x_missing)

                    # Compute average loss
                    avg_loss += np.mean(loss)
                    avg_KL_s += np.mean(KL_s)
                    avg_KL_z += np.mean(KL_z)
                    avg_loss_reg += np.mean(loss_reg)

                print(f"[*] Epoch: {epoch} Rec. Loss: {avg_loss/n_batches} KL s: {avg_KL_s/n_batches} KL z: {avg_KL_z/n_batches}")
                loss_epoch.append(-avg_loss/n_batches)

                if epoch % args.save == 0:
                    print(f"[*] Saving Variables: {network_file_name}")  
                    save_path = saver.save(session, network_file_name)    

            print("[*] Training Finished ...")
            plt.clf()
            plt.figure()
            plt.plot(loss_epoch)
            plt.xlabel('Epoch')
            plt.ylabel('Reconstruction loss')  # we already handled the x-label with ax1
            plt.title(args.save_file)
            plt.savefig(f"{self.train_stats}{args.save_file}.png", bbox_inches='tight')
    
    # originally from helpers.py
    def enc_network(self, settings):
        'get s and z samples as embeddings as well as the original dataframe (with relevelled factors & NA\'s=0!)'
        argvals = settings.split()
        args = self.getArgs(argvals)
        print(f"[*] args: {args}")

        #Create a directoy for the save file
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name = f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
        
        #Creating graph
        sess_HVAE = tf.Graph()

        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, z_dim=args.dim_latent_z, 
                                        y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, y_dim_partition=args.dim_latent_y_partition)

        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        #Get an integer number of batches
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
        #Compute the real miss_mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)

        with tf.Session(graph=sess_HVAE) as session:
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            saver.restore(session, network_file_name)
            print(f"[*] Model restored: {network_file_name}")

            # Training cycle
            avg_loss = 0.
            avg_loss_reg = 0.
            avg_KL_s = 0.
            avg_KL_z = 0.
            samples_list = []
            q_params_list = []

            # Constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3

            for i in range(n_batches):      

                data_list, miss_list = self.next_batch(train_data, types_dict, miss_mask, args.batch_size, index_batch=i)#Create train minibatch
                data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]#Delete not known data

                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape) # unused
                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['zcodes']] = np.ones(args.batch_size).reshape((args.batch_size,1))
                feedDict[tf_nodes['scodes']] = np.ones(args.batch_size).reshape((args.batch_size,1))

                #Get samples from the model
                KL_s,loss,samples,log_p_x,log_p_x_missing,loss_total,KL_z,p_params,q_params,loss_reg  = session.run([tf_nodes['KL_s'], tf_nodes['loss_re'],tf_nodes['samples'],
                                                        tf_nodes['log_p_x'], tf_nodes['log_p_x_missing'],tf_nodes['loss'],
                                                        tf_nodes['KL_z'],tf_nodes['p_params'],tf_nodes['q_params'],tf_nodes['loss_reg']], feed_dict=feedDict)

                samples_list.append(samples)
                q_params_list.append(q_params)

                # Compute average loss
                avg_loss += np.mean(loss)
                avg_loss_reg += np.mean(loss_reg)
                avg_KL_s += np.mean(KL_s)
                avg_KL_z += np.mean(KL_z)

            #Transform discrete variables to original values (this is for getting the original data frame)
            train_data_transformed = self.discrete_variables_transformation(train_data, types_dict)

            #Create global dictionary of the distribution parameters
            q_params_complete = self.q_distribution_params_concatenation(q_params_list)

            # return the deterministic and sampled s and z codes and the reconstructed dataframe (now imputed)
            encs = np.argmax(q_params_complete['s'],1)
            encz = q_params_complete['z'][0,:,:]
            return [encs, encz, train_data_transformed]
    
    # originally from helpers.py
    def merge_dat(self, lis):
        'merge all dataframes in a list on SUBJID'
        df = lis[0]
        for x in lis[1:]:
            df=pd.merge(df, x, on = 'SUBJID')
        return df
    
    # originally from helpers.py
    def dec_network(self, settings, zcodes, scodes, VP=False):
        'decode using set s and z values (if generated provide a generated miss_list) and return decoded data'
        argvals = settings.split()
        args = self.getArgs(argvals)

        #Create a directoy for the save file
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name=f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
        
        #Creating graph
        sess_HVAE = tf.Graph()
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, 
                                        z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, 
                                        y_dim_partition=args.dim_latent_y_partition)

        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        
        #Get an integer number of batches
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
        
        ######Compute the real miss_mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)
            
        with tf.Session(graph=sess_HVAE) as session:
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            saver.restore(session, network_file_name)
            print(f"[*] Model restored: {network_file_name}")

            print('::::::DECODING:::::::::')
            # Training cycle
            samples_list = []

            # Constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3

            for i in range(n_batches):  
                data_list, miss_list = self.next_batch(train_data, types_dict, miss_mask, args.batch_size, index_batch=i) #Create inputs for the feed_dict
                data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]#Delete not known data

                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list

                if VP:
                    sub = re.sub(f"{self.data_python}|.csv", "", args.data_file)
                    vpfile=f"{self.VP_misslist}{sub}_vpmiss.csv"
                    print(f"[*] ::::::::::::{vpfile}")
                    feedDict[tf_nodes['miss_list_VP']] = pd.read_csv(vpfile, header=None)
                elif VP=='nomiss':
                    print("[*] :::::::::::: ones for miss list VP")
                    feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape)
                else:
                    feedDict[tf_nodes['miss_list_VP']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['zcodes']] = np.array(zcodes).reshape((len(zcodes),1))
                feedDict[tf_nodes['scodes']] = np.array(scodes).reshape((len(scodes),1))

                #Get samples from the fixed decoder function
                samples_zgen,log_p_x_test,log_p_x_missing_test,test_params  = session.run([tf_nodes['samples_zgen'],tf_nodes['log_p_x_zgen'],
                                                                                        tf_nodes['log_p_x_missing_zgen'],tf_nodes['test_params_zgen']], feed_dict=feedDict)
                samples_list.append(samples_zgen)

            #Separate the samples from the batch list
            s_aux, z_aux, y_total, est_data = self.samples_concatenation(samples_list)

            #Transform discrete variables to original values
            print(f"[*] est data: {est_data}")
            print(f"[*] types_dict: {types_dict}")
            
            est_data_transformed = self.discrete_variables_transformation(est_data, types_dict)

            return est_data_transformed
    
    # originally from helpers.py
    def dec_network_loglik(self, settings, zcodes, scodes, VP=False):
        'decode using set s and z values (if generated provide a generated miss_list) and return decoded data'
        argvals = settings.split()
        args = self.getArgs(argvals)
        print(f"[*] args: {args}")

        #Create a directoy for the save file
        if not os.path.exists(f"{self.saved_networks}{args.save_file}"):
            os.makedirs(f"{self.saved_networks}{args.save_file}")
        network_file_name = f"{self.saved_networks}{args.save_file}/{args.save_file}.ckpt"
    
        #Creating graph
        sess_HVAE = tf.Graph()
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, z_dim=args.dim_latent_z, 
                                        y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, y_dim_partition=args.dim_latent_y_partition)

        train_data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        
        #Get an integer number of batches
        n_batches = int(np.floor(np.shape(train_data)[0]/args.batch_size))
        
        ######Compute the real miss_mask
        miss_mask = np.multiply(miss_mask, true_miss_mask)
            
        with tf.Session(graph=sess_HVAE) as session:
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            saver.restore(session, network_file_name)
            print(f"[*] Model restored: {network_file_name}")

            print('[*] ::::::DECODING:::::::::')
            # Training cycle
            samples_list = []

            # Constant Gumbel-Softmax parameter (where we have finished the annealing)
            tau = 1e-3

            for i in range(n_batches):      

                data_list, miss_list = self.next_batch(train_data, types_dict, miss_mask, args.batch_size, index_batch=i) #Create inputs for the feed_dict
                data_list_observed = [data_list[i]*np.reshape(miss_list[:,i],[args.batch_size,1]) for i in range(len(data_list))]#Delete not known data

                #Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
                feedDict[tf_nodes['miss_list']] = miss_list
                if VP:
                    sub = re.sub(f"{self.data_python}|.csv", '', args.data_file)
                    vpfile = f"{self.VP_misslist}{sub}_vpmiss.csv"
                    print(f"[*] ::::::::::::{vpfile}")
                    feedDict[tf_nodes['miss_list_VP']] = pd.read_csv(vpfile,header=None)
                elif VP=='nomiss':
                    print(":::::::::::: ones for miss list VP")
                    feedDict[tf_nodes['miss_list_VP']] = np.ones(miss_list.shape)
                else:
                    feedDict[tf_nodes['miss_list_VP']] = miss_list
                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['zcodes']] = np.array(zcodes).reshape((len(zcodes),1))
                feedDict[tf_nodes['scodes']] = np.array(scodes).reshape((len(scodes),1))

                #Get samples from the fixed decoder function
                samples_zgen, log_p_x_test, log_p_x_missing_test, test_params  = session.run([tf_nodes['samples_zgen'],tf_nodes['log_p_x_zgen'],
                                                                                            tf_nodes['log_p_x_missing_zgen'],tf_nodes['test_params_zgen']],feed_dict=feedDict)
                samples_list.append(samples_zgen)

            return log_p_x_test
    
    # originally from parser_arguments.py
    def getArgs(self, argv=None):
        parser = argparse.ArgumentParser(description='Default parameters of the models',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--batch_size', type=int, default=200, help='Size of the batches')
        parser.add_argument('--epochs',type=int,default=5001, help='Number of epochs of the simulations')
        parser.add_argument('--perp',type=int,default=10, help='Perplexity for the t-SNE')
        parser.add_argument('--display', type=int,default=1, help='Display option flag')
        parser.add_argument('--save', type=int,default=1000, help='Save variables every save iterations')
        parser.add_argument('--restore', type=int,default=0, help='To restore session, to keep training or evaluation') 
        parser.add_argument('--plot', type=int,default=1, help='Plot results flag')
        parser.add_argument('--dim_latent_s',type=int,default=10, help='Dimension of the categorical space')
        parser.add_argument('--dim_latent_z',type=int,default=2, help='Dimension of the Z latent space')
        parser.add_argument('--dim_latent_y',type=int,default=10, help='Dimension of the Y latent space')
        parser.add_argument('--dim_latent_y_partition',type=int, nargs='+', help='Partition of the Y latent space')
        parser.add_argument('--miss_percentage_train',type=float,default=0.0, help='Percentage of missing data in training')
        parser.add_argument('--miss_percentage_test',type=float,default=0.0, help='Percentage of missing data in test')
        parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
        parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
        parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
        parser.add_argument('--miss_file', type=str, default='Missing_test.csv', help='File with the missing indexes mask')
        parser.add_argument('--true_miss_file', type=str, help='File with the missing indexes when there are NaN in the data')
        parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
        return parser.parse_args(argv)

    # originally from graph_new.py
    def HVAE_graph(self, types_file, batch_size, learning_rate=1e-3, z_dim=2, y_dim=1, s_dim=2, y_dim_partition=[]):
        print(f"[*] types_file = {types_file}")
        
        #Load placeholders
        print("[*] Defining placeholders")
        batch_data_list, batch_data_list_observed, miss_list,miss_list_VP, tau, types_list, zcodes, scodes = self.place_holder_types(types_file, batch_size)
        
        #Batch normalization of the data
        X_list, normalization_params = self.batch_normalization(batch_data_list_observed, types_list, miss_list)
        
        #Set dimensionality of Y
        if y_dim_partition:
            y_dim_output = np.sum(y_dim_partition)
        else:
            y_dim_partition = y_dim*np.ones(len(types_list), dtype=int)
            y_dim_output = np.sum(y_dim_partition)
        
        #Encoder definition
        print("[*] Defining Encoder...")
        samples, q_params = self.encoder(X_list, batch_size, z_dim, s_dim, tau)
        
        print("[*] Defining Decoder...")
        theta, samples, p_params, log_p_x, log_p_x_missing = self.decoder(batch_data_list, miss_list, types_list, samples, normalization_params, batch_size, z_dim, y_dim_output, y_dim_partition)
        
        print("[*] Defining Cost function...")
        ELBO, loss_reconstruction, KL_z, KL_s = self.cost_function(log_p_x, p_params, q_params, z_dim, s_dim)
        
        loss_reg = tf.losses.get_regularization_loss() # not using this at the moment (set weight_decay to 0 to be safe)
        optim = tf.train.AdamOptimizer(learning_rate).minimize(-ELBO)# + loss_reg)

        # fixed decoder for passing s/z codes and miss_list of VPs generated in the BNet
        samples_zgen, test_params_zgen, log_p_x_zgen, log_p_x_missing_zgen = self.fixed_decoder(batch_data_list, X_list, miss_list_VP,miss_list, types_list, batch_size, y_dim_output, y_dim_partition, s_dim, normalization_params, zcodes, scodes)

        #Packing results
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
                    'loss_reg':loss_reg,
                    'optim': optim,
                    'KL_s': KL_s,
                    'KL_z': KL_z,
                    'p_params': p_params,
                    'q_params': q_params,
                    'samples_zgen': samples_zgen,
                    'test_params_zgen': test_params_zgen,
                    'log_p_x_zgen': log_p_x_zgen,
                    'log_p_x_missing_zgen': log_p_x_missing_zgen}

        return tf_nodes
    
    # originally from read_functions.py
    def read_data(self, data_file, types_file, miss_file, true_miss_file):
        
        #Read types of data from data file
        with open(types_file) as f:
            types_dict = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
        
        #Read data from input file
        with open(data_file, 'r') as f:
            data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            data = np.array(data)
        
        #Sustitute NaN values by something (we assume we have the real missing value mask)
        if true_miss_file:
            with open(true_miss_file, 'r') as f:
                missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
                missing_positions = np.array(missing_positions)
                
            true_miss_mask = np.ones([np.shape(data)[0],len(types_dict)])
            true_miss_mask[missing_positions[:,0]-1,missing_positions[:,1]-1] = 0 #Indexes in the csv start at 1
            data_masked = np.ma.masked_where(np.isnan(data),data) 

            #We need to fill the data depending on the given data...
            data_filler = []
            for i in range(len(types_dict)):
                if types_dict[i]['type'] == 'cat' or types_dict[i]['type'] == 'ordinal':
                    aux = np.unique(data[:,i])
                    if not np.isnan(aux[0]):
                        data_filler.append(aux[0])  #Fill with the first element of the cat (0, 1, or whatever)
                    else:
                        data_filler.append(int(0))
                else:
                    data_filler.append(0.0)
                
            data = data_masked.filled(data_filler)
        else:
            true_miss_mask = np.ones([np.shape(data)[0],len(types_dict)]) #It doesn't affect our data
        
        #Construct the data matrices
        data_complete = []
        for i in range(np.shape(data)[1]):
            
            if types_dict[i]['type'] == 'cat':
                #Get categories
                cat_data = [int(x) for x in data[:,i]]
                categories, indexes = np.unique(cat_data,return_inverse=True)
                #Transform categories to a vector of 0:n_categories
                new_categories = np.arange(int(types_dict[i]['dim']))
                cat_data = new_categories[indexes]
                #Create one hot encoding for the categories
                aux = np.zeros([np.shape(data)[0],len(new_categories)])
                aux[np.arange(np.shape(data)[0]),cat_data] = 1
                data_complete.append(aux)
                
            elif types_dict[i]['type'] == 'ordinal':
                #Get categories
                cat_data = [int(x) for x in data[:,i]]
                categories, indexes = np.unique(cat_data,return_inverse=True)
                #Transform categories to a vector of 0:n_categories
                new_categories = np.arange(int(types_dict[i]['dim']))
                cat_data = new_categories[indexes]
                #Create thermometer encoding for the categories
                aux = np.zeros([np.shape(data)[0],1+len(new_categories)])
                aux[:,0] = 1
                aux[np.arange(np.shape(data)[0]),1+cat_data] = -1
                aux = np.cumsum(aux,1)
                data_complete.append(aux[:,:-1])
                
            else:
                data_complete.append(np.transpose([data[:,i]]))
                        
        data = np.concatenate(data_complete,1)
        
        #Read Missing mask from csv (contains positions of missing values)
        n_samples = np.shape(data)[0]
        n_variables = len(types_dict)
        miss_mask = np.ones([np.shape(data)[0],n_variables])
        #If there is no mask, assume all data is observed
        if os.path.isfile(miss_file):
            with open(miss_file, 'r') as f:
                missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
                missing_positions = np.array(missing_positions)
            miss_mask[missing_positions[:,0]-1,missing_positions[:,1]-1] = 0 #Indexes in the csv start at 1
            
        return data, types_dict, miss_mask, true_miss_mask, n_samples
    
    # originally from read_functions.py
    def next_batch(self, data, types_dict, miss_mask, batch_size, index_batch):
        #Create minibath
        batch_xs = data[index_batch*batch_size:(index_batch+1)*batch_size, :]
        
        #Slipt variables of the batches
        data_list = []
        initial_index = 0
        for d in types_dict:
            dim = int(d['dim'])
            data_list.append(batch_xs[:, initial_index:initial_index+dim])
            initial_index += dim
        
        #Missing data
        miss_list = miss_mask[index_batch*batch_size:(index_batch+1)*batch_size, :]

        return data_list, miss_list
    
    # originally from read_functions.py
    def discrete_variables_transformation(self, data, types_dict):
        
        ind_ini = 0
        output = []
        for d in range(len(types_dict)):
            ind_end = ind_ini + int(types_dict[d]['dim'])
            if types_dict[d]['type'] == 'cat':
                output.append(np.reshape(np.argmax(data[:,ind_ini:ind_end],1),[-1,1]))
            elif types_dict[d]['type'] == 'ordinal':
                output.append(np.reshape(np.sum(data[:,ind_ini:ind_end],1) - 1,[-1,1]))
            else:
                output.append(data[:,ind_ini:ind_end])
            ind_ini = ind_end
        
        return np.concatenate(output,1)
    
    # originally from read_functions.py
    def q_distribution_params_concatenation(self, params):
        
        keys = params[0].keys()
        out_dict = {key: [] for key in keys}
        
        for i,batch in enumerate(params):
            for d,k in enumerate(keys):
                out_dict[k].append(batch[k])
                
        out_dict['z'] = np.concatenate(out_dict['z'],1)
        out_dict['s'] = np.concatenate(out_dict['s'],0)
            
        return out_dict
    
    # originally from read_functions.py
    def samples_concatenation(self, samples):
        
        for i,batch in enumerate(samples):
            if i == 0:
                samples_x = np.concatenate(batch['x'], 1)
                samples_y = batch['y']
                samples_z = batch['z']
                samples_s = batch['s']
            else:
                samples_x = np.concatenate([samples_x,np.concatenate(batch['x'],1)], 0)
                samples_y = np.concatenate([samples_y,batch['y']], 0)
                samples_z = np.concatenate([samples_z,batch['z']], 0)
                samples_s = np.concatenate([samples_s,batch['s']], 0)
            
        return samples_s, samples_z, samples_y, samples_x

    # originally from VAE_functions.py
    def place_holder_types(self, types_file, batch_size):
        #Read the types of the data from the files
        with open(types_file) as f:
            types_list = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
            
        #Create placeholders for every data type, with appropriate dimensions
        batch_data_list = []
        print(f"[*] types_list = {types_list}")

        for i in range(len(types_list)):
            print(i)
            batch_data_list.append(tf.placeholder(tf.float32, shape=(batch_size,int(types_list[i]['dim']))))
        tf.concat(batch_data_list, axis=1)
        
        #Create placeholders for every missing data type, with appropriate dimensions
        batch_data_list_observed = []
        for i in range(len(types_list)):
            batch_data_list_observed.append(tf.placeholder(tf.float32, shape=(batch_size,int(types_list[i]['dim']))))
        tf.concat(batch_data_list_observed, axis=1)
            
        #Create placeholders for the missing data indicator variable
        miss_list = tf.placeholder(tf.int32, shape=(batch_size,len(types_list)))
        miss_list_VP = tf.placeholder(tf.int32, shape=(batch_size,len(types_list)))
        
        #Placeholder for Gumbel-softmax parameter
        tau = tf.placeholder(tf.float32,shape=())
        
        zcodes=tf.placeholder(tf.float32, shape=(batch_size,1))
        scodes=tf.placeholder(tf.int32, shape=(batch_size,1))
        
        return batch_data_list, batch_data_list_observed, miss_list, miss_list_VP, tau, types_list, zcodes, scodes

    # originally from VAE_functions.py
    def batch_normalization(self, batch_data_list, types_list, miss_list):
        
        normalized_data = []
        normalization_parameters = []
        
        for i,d in enumerate(batch_data_list):
            #Partition the data in missing data (0) and observed data n(1)
            missing_data, observed_data = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
            
            if types_list[i]['type'] == 'real':
                #We transform the data to a gaussian with mean 0 and std 1
                data_mean, data_var = tf.nn.moments(observed_data, 0)
                data_var = tf.clip_by_value(data_var,1e-6,1e20) #Avoid zero values
                aux_X = tf.nn.batch_normalization(observed_data, data_mean, data_var, offset=0.0, scale=1.0, variance_epsilon=1e-6)
                
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
                normalization_parameters.append([data_mean, data_var])
                
            #When using log-normal
            elif types_list[i]['type'] == 'pos':
            #We transform the log of the data to a gaussian with mean 0 and std 1
                observed_data_log = tf.log(1 + observed_data)
                data_mean_log, data_var_log = tf.nn.moments(observed_data_log,0)
                data_var_log = tf.clip_by_value(data_var_log, 1e-6, 1e20) #Avoid zero values
                aux_X = tf.nn.batch_normalization(observed_data_log, data_mean_log, data_var_log, offset=0.0, scale=1.0, variance_epsilon=1e-6)
                
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
                normalization_parameters.append([data_mean_log, data_var_log])
                
            elif types_list[i]['type'] == 'count':
                
                #Input log of the data
                aux_X = tf.log(observed_data)
                
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data, aux_X]))
                normalization_parameters.append([0.0, 1.0])
            else:
                #Don't normalize the categorical and ordinal variables
                normalized_data.append(d)
                normalization_parameters.append([0.0, 1.0]) #No normalization here
        
        return normalized_data, normalization_parameters

    # originally from model_HIVAE_inputDropout.py
    def encoder(self, X_list, batch_size, z_dim, s_dim, tau):
        
        samples = dict.fromkeys(['s','z','y','x'],[])
        q_params = dict()
        X = tf.concat(X_list,1)
        
        #Create the proposal of q(s|x^o)
        samples['s'], q_params['s'] = self.s_proposal_multinomial(X, batch_size, s_dim, tau, reuse=None)
        
        #Create the proposal of q(z|s,x^o)
        samples['z'], q_params['z'] = self.z_proposal_GMM(X, samples['s'], batch_size, z_dim, reuse=None)
        
        return samples, q_params

    # originally from model_HIVAE_inputDropout.py
    def decoder(self, batch_data_list, miss_list, types_list, samples, normalization_params, batch_size, z_dim, y_dim, y_dim_partition):
        
        p_params = dict()
        
        #Create the distribution of p(z|s)
        p_params['z'] = self.z_distribution_GMM(samples['s'], z_dim, reuse=None)
        
        #Create deterministic layer y
        samples['y'] = tf.layers.dense(inputs=samples['z'], units=y_dim, activation=None,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42), name= 'layer_h1_', reuse=None)
        
        grouped_samples_y = self.y_partition(samples['y'], types_list, y_dim_partition)

        #Compute the parameters h_y
        theta = self.theta_estimation_from_y(grouped_samples_y, types_list, miss_list, batch_size, reuse=None)
        
        #Compute loglik and output of the VAE
        log_p_x, log_p_x_missing, samples['x'], p_params['x'] = self.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params)
            
        return theta, samples, p_params, log_p_x, log_p_x_missing

    # originally from model_HIVAE_inputDropout.py
    def cost_function(self, log_p_x, p_params, q_params, z_dim, s_dim):
        #KL(q(s|x)|p(s))
        log_pi = q_params['s']
        pi_param = tf.nn.softmax(log_pi)
        KL_s = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=pi_param) + tf.log(float(s_dim))
        
        #KL(q(z|s,x)|p(z|s))
        # to implement: if flagged iteration, take pred z instead
        mean_pz, log_var_pz = p_params['z']
        mean_qz, log_var_qz = q_params['z']
        KL_z = -0.5*z_dim +0.5*tf.reduce_sum(tf.exp(log_var_qz - log_var_pz) +tf.square(mean_pz - mean_qz)/tf.exp(log_var_pz) -log_var_qz + log_var_pz,1)
        
        #Eq[log_p(x|y)]
        loss_reconstruction = tf.reduce_sum(log_p_x, 0)
        
        #Complete ELBO
        ELBO = tf.reduce_mean(loss_reconstruction - KL_z - KL_s, 0)
        
        return ELBO, loss_reconstruction, KL_z, KL_s
    
    # originally from model_HIVAE_inputDropout.py
    def fixed_decoder(self, batch_data_list, X_list, miss_list_VP, miss_list, types_list, batch_size, y_dim, y_dim_partition, s_dim, normalization_params, zcodes, scodes):
        
        samples_test = dict.fromkeys(['s','z','y','x'],[])
        test_params = dict()
        X = tf.concat(X_list, 1)
        
        #Create the proposal of q(s|x^o)
        samples_test['s'] = tf.one_hot(scodes,depth=s_dim)
        
        # set fixed z
        samples_test['z'] = zcodes
        
        #Create deterministic layer y
        samples_test['y'] = tf.layers.dense(inputs=samples_test['z'], units=y_dim, activation=None,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42), name= 'layer_h1_', reuse=True)
        
        grouped_samples_y = self.y_partition(samples_test['y'], types_list, y_dim_partition)
        
        #Compute the parameters h_y
        theta = self.theta_estimation_from_y(grouped_samples_y, types_list, miss_list_VP, batch_size, reuse=True)
        
        #Compute loglik and output of the VAE
        log_p_x, log_p_x_missing, samples_test['x'], test_params['x'] = self.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params)
        
        return samples_test, test_params, log_p_x, log_p_x_missing

    # originally from VAE_functions.py
    def s_proposal_multinomial(self, X, batch_size, s_dim, tau, reuse):
        
        #We propose a categorical distribution to create a GMM for the latent space z
        log_pi = tf.layers.dense(inputs=X, units=s_dim, activation=None,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42), name='layer_1_enc_s', reuse=reuse)
        
        #Gumbel-softmax trick
        U = -tf.log(-tf.log(tf.random_uniform([batch_size,s_dim], seed=42)))
        samples_s = tf.nn.softmax((log_pi + U)/tau)
        
        return samples_s, log_pi

    # originally from VAE_functions.py
    def z_proposal_GMM(self, X, samples_s, batch_size, z_dim, reuse):
        
        #We propose a GMM for z
        mean_qz = tf.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42), name='layer_1_mean_enc_z', reuse=reuse)
        log_var_qz = tf.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42), name='layer_1_logvar_enc_z', reuse=reuse)
        
        # Avoid numerical problems
        log_var_qz = tf.clip_by_value(log_var_qz,-15.0,15.0)
        # Rep-trick
        eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32, seed=42)
        samples_z = mean_qz+tf.multiply(tf.exp(log_var_qz/2), eps)
        
        return samples_z, [mean_qz, log_var_qz]

    # originally from VAE_functions.py
    def z_distribution_GMM(self, samples_s, z_dim, reuse):
        
        #We propose a GMM for z
        mean_pz = tf.layers.dense(inputs=samples_s, units=z_dim, activation=None,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42), name= 'layer_1_mean_dec_z', reuse=reuse)
        log_var_pz = tf.zeros([tf.shape(samples_s)[0], z_dim])
        
        # Avoid numerical problems
        log_var_pz = tf.clip_by_value(log_var_pz, -15.0, 15.0)
        
        return mean_pz, log_var_pz

    # originally from VAE_functions.py
    def y_partition(self, samples_y, types_list, y_dim_partition):
        
        grouped_samples_y = []
        #First element must be 0 and the length of the partition vector must be len(types_dict)+1
        if len(y_dim_partition) != len(types_list):
            raise Exception("[*] The length of the partition vector must match the number of variables in the data + 1")
            
        #Insert a 0 at the beginning of the cumsum vector
        partition_vector_cumsum = np.insert(np.cumsum(y_dim_partition), 0, 0)
        for i in range(len(types_list)):
            grouped_samples_y.append(samples_y[:, partition_vector_cumsum[i]:partition_vector_cumsum[i+1]])
        
        return grouped_samples_y

    # originally from VAE_functions.py
    def theta_estimation_from_y(self, samples_y, types_list, miss_list, batch_size, reuse):
        theta = []
        #Independet yd -> Compute p(xd|yd)
        for i,d in enumerate(samples_y):
            
            #Partition the data in missing data (0) and observed data (1)
            missing_y, observed_y = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
            nObs = tf.shape(observed_y)[0]
            
            #Different layer models for each type of variable
            if types_list[i]['type'] == 'real':
                params = self.theta_real(observed_y, missing_y, condition_indices, types_list, i, reuse)
            
            elif types_list[i]['type'] == 'pos':
                params = self.theta_pos(observed_y, missing_y, condition_indices, types_list, i, reuse)
                
            elif types_list[i]['type'] == 'count':
                params = self.theta_count(observed_y, missing_y, condition_indices, types_list, i, reuse)
            
            elif types_list[i]['type'] == 'cat':
                params = self.theta_cat(observed_y, missing_y, condition_indices, types_list, batch_size, i, reuse)
                
            elif types_list[i]['type'] == 'ordinal':
                params = self.theta_ordinal(observed_y, missing_y, condition_indices, types_list, i, reuse)
                
            theta.append(params)
                
        return theta

    # originally from VAE_functions.py
    def loglik_evaluation(self, batch_data_list, types_list, miss_list, theta, normalization_params):
        log_p_x = []
        log_p_x_missing = []
        samples_x = []
        params_x = []
        
        #Independet yd -> Compute log(p(xd|yd))
        for i, d in enumerate(batch_data_list):
            # Select the likelihood for the types of variables
            loglik_models = c_loglik_models_missing_normalize([d,miss_list[:,i]], types_list[i], theta[i], normalization_params[i])
            loglik_function = getattr(loglik_models, 'loglik_' + types_list[i]['type'])

            out = loglik_function()
                
            log_p_x.append(out['log_p_x'])
            log_p_x_missing.append(out['log_p_x_missing']) #Test-loglik element
            samples_x.append(out['samples'])
            params_x.append(out['params'])
            
        return log_p_x, log_p_x_missing, samples_x, params_x
    
    # originally from VAE_functions.py
    def theta_real(self, observed_y, missing_y, condition_indices, types_list, i, reuse):
        
        #Mean layer
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
        #Sigma Layer
        h2_sigma = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i), reuse=reuse)
        
        return [h2_mean, h2_sigma]

    # originally from VAE_functions.py
    def theta_pos(self, observed_y, missing_y, condition_indices, types_list, i, reuse):
        
        #Mean layer
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
        #Sigma Layer
        h2_sigma = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i), reuse=reuse)
        
        return [h2_mean, h2_sigma]

    # originally from VAE_functions.py
    def theta_count(self, observed_y, missing_y, condition_indices, types_list, i, reuse):
        
        #Lambda Layer
        h2_lambda = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
        
        return h2_lambda

    # originally from VAE_functions.py
    def theta_cat(self, observed_y, missing_y, condition_indices, types_list, batch_size, i, reuse):
        
        #Log pi layer, with zeros in the first value to avoid the identificability problem
        h2_log_pi_partial = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, name='layer_h2' + str(i), reuse=reuse)
        h2_log_pi = tf.concat([tf.zeros([batch_size,1]), h2_log_pi_partial],1)
        
        return h2_log_pi
    
    # originally from VAE_functions.py
    def theta_ordinal(self, observed_y, missing_y, condition_indices, types_list, i, reuse):
        
        #Theta layer, Dimension of ordinal - 1
        h2_theta = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, name='layer_h2' + str(i), reuse=reuse)
        #Mean layer, a single value
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=1, name='layer_h2_sigma' + str(i), reuse=reuse)
        
        return [h2_theta, h2_mean]
    
    # originally from VAE_functions.py
    def observed_data_layer(self, observed_data, missing_data, condition_indices, output_dim, name, reuse):
        #Train a layer with the observed data and reuse it for the missing data
        obs_output = tf.layers.dense(inputs=observed_data, units=output_dim, activation=None,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),name=name,reuse=reuse,trainable=True)
        miss_output = tf.layers.dense(inputs=missing_data, units=output_dim, activation=None,
                        kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),name=name,reuse=True,trainable=False)
        #Join back the data
        output = tf.dynamic_stitch(condition_indices, [miss_output,obs_output])
        
        return output


class c_HIVAE_GridSearch:

    def __init__(self):
        plt.ioff()
        tf.disable_v2_behavior()

        cp = configparser.RawConfigParser()
        config_file = r'/vambn/02_config/config_python.txt'
        assert os.path.exists(r'/vambn/02_config/config_python.txt')
        cp.read(config_file)
        print(f'[*] Config file sections: {cp.sections()}')
        
        self.data_python = cp.get('GridSearch', 'path_data_python')
        self.results = cp.get('GridSearch', 'path_results') 
        self.search_options = {
            'ydims': [int(cp.get('GridSearch', 'hyperparam_options_y_dimensions'))],
            'lrates': [float(i) for i in cp.get('GridSearch', 'hyperparam_options_learning_rates').split(',')],
            'wdecay': [int(cp.get('GridSearch', 'hyperparam_options_weight_decay'))],
            'nbatch': [int(i) for i in cp.get('GridSearch', 'hyperparam_options_batch_size').split(',')]
        }
        print(f'[*] Config file search options: {self.search_options}')

    # originally from GridSearch_ADNI.ipynb
    def hyperopt_HIVAE(self):
        files=[i for i in os.listdir(self.data_python) if not '_type' in i and not '_missing' in i]
        
        # cross val rec loss
        keys, values = zip(*self.search_options.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        n_splits=3

        l_files=[]
        for f in files:
            print(f'\n[*] Currently performing grid search for: {f}. Please wait...')
            l_exp=[]
            for opt in experiments:
                settings=self.set_settings(f, opt)
                l_exp.append(self.run_network(settings,
                                        'YD'+str(opt['ydims'])+'_LR'+str(opt['lrates'])+'_WD'+str(opt['wdecay'])+'_NB'+str(opt['nbatch']),
                                        n_splits=n_splits))
            l_files.append(l_exp)
            
        losses=list(zip(files,l_files))
        selectexp=[np.argmin(losses[f][1]) for f in range(len(files))]
        minloss=[np.nanmin(losses[f][1]) for f in range(len(files))]
        output=itemgetter(*selectexp)(experiments)
        output=pd.DataFrame(list(output))
        output['files']=files
        output['loss']=minloss
        output.to_csv(self.results, index=True)
        print(output)
        print('[*] Grid search hyperopt script completed.')
    
    # originally from GridSearch_ADNI.ipynb
    def set_settings(self, f, opts):
        'replace setting template placeholders with file info'
        
        inputf = re.sub('.csv', '', f)
        missf = inputf + '_missing.csv'
        typef = inputf + '_types.csv'
        n_epochs = 500
        
        settings = f"--epochs {n_epochs} \
                    --restore 0 \
                    --train 1 \
                    --data_file {self.data_python}{inputf}.csv \
                    --types_file {self.data_python}{typef} \
                    --batch_size {opts['nbatch']} \
                    --save 499 \
                    --save_file {inputf} \
                    --dim_latent_s 1 \
                    --dim_latent_z 1 \
                    --dim_latent_y {opts['ydims']} \
                    --miss_percentage_train 0 \
                    --miss_percentage_test 0 \
                    --true_miss_file {self.data_python}{missf} \
                    --learning_rate {opts['lrates']} \
                    --weight_decay {opts['wdecay']}"
        
        return settings
    
    # originally from helpers.py
    def run_network(self, settings, name, n_splits=3):
        'run training (no output)'

        argvals = settings.split()
        args = self.getArgs(argvals)
        print(f"[*] run_network name: {name}")
        
        # get full data
        data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)
        miss_mask = np.multiply(miss_mask, true_miss_mask) #Compute the real miss_mask
        
        # split data and run training/test per fold
        kf = KFold(n_splits=n_splits, shuffle=True)
        score_keep = []
        fold = 0

        for train_idx, test_idx in kf.split(data):
            fold += 1
            score = self.run_epochs(args, data, train_idx, test_idx, types_dict, miss_mask, true_miss_mask) # returns final train and test score after # epochs
            score_keep.append(score[1]) # keep test score
            print(f"[*] Score for fold {fold}: Train - {score[0]:.3f} :: Test - {score[1]:.3f}")
            
        return np.mean(score_keep) # return the mean
    
    # originally from parser_arguments.py
    def getArgs(self, argv=None):

        parser = argparse.ArgumentParser(description='Default parameters of the models',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--batch_size', type=int, default=200, help='Size of the batches')
        parser.add_argument('--epochs',type=int,default=5001, help='Number of epochs of the simulations')
        parser.add_argument('--perp',type=int,default=10, help='Perplexity for the t-SNE')
        parser.add_argument('--train', type=int,default=1, help='Training model flag')
        parser.add_argument('--display', type=int,default=1, help='Display option flag')
        parser.add_argument('--save', type=int,default=1000, help='Save variables every save iterations')
        parser.add_argument('--restore', type=int,default=0, help='To restore session, to keep training or evaluation') 
        parser.add_argument('--plot', type=int,default=1, help='Plot results flag')
        parser.add_argument('--dim_latent_s',type=int,default=10, help='Dimension of the categorical space')
        parser.add_argument('--dim_latent_z',type=int,default=2, help='Dimension of the Z latent space')
        parser.add_argument('--dim_latent_y',type=int,default=10, help='Dimension of the Y latent space')
        parser.add_argument('--dim_latent_y_partition',type=int, nargs='+', help='Partition of the Y latent space')
        parser.add_argument('--miss_percentage_train',type=float,default=0.0, help='Percentage of missing data in training')
        parser.add_argument('--miss_percentage_test',type=float,default=0.0, help='Percentage of missing data in test')
        parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
        parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
        parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
        parser.add_argument('--miss_file', type=str, default='Missing_test.csv', help='File with the missing indexes mask')
        parser.add_argument('--true_miss_file', type=str, help='File with the missing indexes when there are NaN in the data')
        parser.add_argument('--learning_rate', type=float, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, help='L2: Weight decay')
        parser.add_argument('--activation', type=str, default='none', help='Activation function')
    
        return parser.parse_args(argv)
    
    # originally from read_functions.py
    def read_data(self, data_file, types_file, miss_file, true_miss_file):
        
        #Read types of data from data file
        with open(types_file) as f:
            types_dict = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
        
        #Read data from input file
        with open(data_file, 'r') as f:
            data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            data = np.array(data)
        
        #Sustitute NaN values by something (we assume we have the real missing value mask)
        if true_miss_file:
            with open(true_miss_file, 'r') as f:
                missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
                missing_positions = np.array(missing_positions)
                
            true_miss_mask = np.ones([np.shape(data)[0], len(types_dict)])
            true_miss_mask[missing_positions[:,0]-1, missing_positions[:,1]-1] = 0 #Indexes in the csv start at 1
            data_masked = np.ma.masked_where(np.isnan(data),data) 

            #We need to fill the data depending on the given data...
            data_filler = []
            for i in range(len(types_dict)):
                if types_dict[i]['type'] == 'cat' or types_dict[i]['type'] == 'ordinal':
                    aux = np.unique(data[:,i])
                    if not np.isnan(aux[0]):
                        data_filler.append(aux[0])  #Fill with the first element of the cat (0, 1, or whatever)
                    else:
                        data_filler.append(int(0))
                else:
                    data_filler.append(0.0)
                
            data = data_masked.filled(data_filler)
        else:
            true_miss_mask = np.ones([np.shape(data)[0], len(types_dict)]) #It doesn't affect our data
        
        #Construct the data matrices
        data_complete = []
        for i in range(np.shape(data)[1]):
            
            if types_dict[i]['type'] == 'cat':
                #Get categories
                cat_data = [int(x) for x in data[:,i]]
                categories, indexes = np.unique(cat_data, return_inverse=True)
                #Transform categories to a vector of 0:n_categories
                new_categories = np.arange(int(types_dict[i]['dim']))
                cat_data = new_categories[indexes]
                #Create one hot encoding for the categories
                aux = np.zeros([np.shape(data)[0],len(new_categories)])
                aux[np.arange(np.shape(data)[0]),cat_data] = 1
                data_complete.append(aux)
                
            elif types_dict[i]['type'] == 'ordinal':
                #Get categories
                cat_data = [int(x) for x in data[:,i]]
                categories, indexes = np.unique(cat_data, return_inverse=True)
                #Transform categories to a vector of 0:n_categories
                new_categories = np.arange(int(types_dict[i]['dim']))
                cat_data = new_categories[indexes]
                #Create thermometer encoding for the categories
                aux = np.zeros([np.shape(data)[0], 1+len(new_categories)])
                aux[:,0] = 1
                aux[np.arange(np.shape(data)[0]), 1+cat_data] = -1
                aux = np.cumsum(aux, 1)
                data_complete.append(aux[:,:-1])
                
            else:
                data_complete.append(np.transpose([data[:,i]]))
                        
        data = np.concatenate(data_complete, 1)
        
        #Read Missing mask from csv (contains positions of missing values)
        n_samples = np.shape(data)[0]
        n_variables = len(types_dict)
        miss_mask = np.ones([np.shape(data)[0], n_variables])
        #If there is no mask, assume all data is observed
        if os.path.isfile(miss_file):
            with open(miss_file, 'r') as f:
                missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
                missing_positions = np.array(missing_positions)
            miss_mask[missing_positions[:,0]-1, missing_positions[:,1]-1] = 0 #Indexes in the csv start at 1
            
        return data, types_dict, miss_mask, true_miss_mask, n_samples
    
    # originally from helpers.py
    def run_epochs(self, args, data, train_idx, test_idx, types_dict, miss_mask, true_miss_mask):
        'this creates the graph and runs train and test batches for this epoch'
        
        #Creating graph
        sess_HVAE = tf.Graph() 
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, 
                                        z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s,
                                        weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)

        n_batches_train = int(np.floor(len(train_idx)/args.batch_size)) #Get an integer number of batches
        n_batches_test = int(np.floor(len(test_idx)/args.batch_size)) #Get an integer number of batches
        
        config = self.gpu_assignment([0,1,2])
        with tf.Session(graph=sess_HVAE, config=config) as session: 
            print("[*] Initizalizing Variables ...")
            print(f"[*] Train size: {len(train_idx)} :: Test size: {len(test_idx)}")
            
            tf.global_variables_initializer().run() 
            
            # Training cycle
            train_loss_epoch=[]
            train_KL_s_epoch = []
            train_KL_z_epoch = []
            train_loss_reg_epoch=[]
            test_loss_epoch=[]
            test_KL_s_epoch = []
            test_KL_z_epoch = []
            test_loss_reg_epoch=[]

            for epoch in range(args.epochs):
                #training
                losses_train = self.run_batches(session, tf_nodes, data[train_idx], types_dict, miss_mask[train_idx], true_miss_mask[train_idx], 
                                                n_batches_train, args.batch_size, args.epochs ,epoch, train=True)
                train_loss_epoch.append(losses_train[0])
                train_KL_s_epoch.append(losses_train[1])
                train_KL_z_epoch.append(losses_train[2])
                train_loss_reg_epoch.append(losses_train[3])

                # testing
                losses_test = self.run_batches(session, tf_nodes, data[test_idx], types_dict, miss_mask[test_idx], true_miss_mask[test_idx],
                                               n_batches_test, args.batch_size, args.epochs, epoch, train=False)
                test_loss_epoch.append(losses_test[0])
                test_KL_s_epoch.append(losses_test[1])
                test_KL_z_epoch.append(losses_test[2])
                test_loss_reg_epoch.append(losses_test[3])
            
        return [train_loss_epoch[-1], test_loss_epoch[-1]]
    
    # originally from graph_new.py
    def HVAE_graph(self, types_file, batch_size, learning_rate=1e-3, z_dim=2, y_dim=1, s_dim=2, weight_decay=0, y_dim_partition=[]):
    
        #Load placeholders
        batch_data_list, batch_data_list_observed, miss_list, tau, types_list,zcodes,scodes = self.place_holder_types(types_file)
        
        #Batch normalization of the data
        X_list, normalization_params = self.batch_normalization(batch_data_list_observed, types_list, miss_list)
        
        #Set dimensionality of Y
        if y_dim_partition:
            y_dim_output = np.sum(y_dim_partition)
        else:
            y_dim_partition = y_dim*np.ones(len(types_list),dtype=int)
            y_dim_output = np.sum(y_dim_partition)
        
        #Encoder definition
        print("[*] Defining Encoder...")
        samples, q_params = self.encoder(X_list, batch_size, z_dim, s_dim, tau, weight_decay)
        
        print("[*] Defining Decoder...")
        theta, samples, p_params, log_p_x, log_p_x_missing = self.decoder(batch_data_list, miss_list, types_list, samples, normalization_params, 
                                                                    batch_size, z_dim, y_dim_output, y_dim_partition, weight_decay)

        print("[*] Defining Cost function...")
        ELBO, loss_reconstruction, KL_z, KL_s = self.cost_function(log_p_x, p_params, q_params, z_dim, s_dim)
        
        loss_reg = tf.losses.get_regularization_loss()
        optim = tf.train.AdamOptimizer(learning_rate).minimize(-ELBO + loss_reg)
        
        # fixed decoder for getting samples based on fixed code inputs (from virtual patient data! if not VP run, just set to ones)
        samples_zgen, test_params_zgen, log_p_x_zgen, log_p_x_missing_zgen = self.fixed_decoder(batch_data_list, X_list, miss_list, types_list, batch_size, 
                                                                                            y_dim_output, y_dim_partition, s_dim, normalization_params,
                                                                                            zcodes, scodes, weight_decay)

        #Packing results
        tf_nodes = {'ground_batch': batch_data_list,
                    'ground_batch_observed': batch_data_list_observed,
                    'miss_list': miss_list,
                    'tau_GS': tau,
                    'zcodes': zcodes,
                    'scodes': scodes,
                    'samples': samples,
                    'log_p_x': log_p_x,
                    'log_p_x_missing': log_p_x_missing,
                    'loss_re': loss_reconstruction,
                    'loss': -ELBO, 
                    'loss_reg': loss_reg,
                    'optim': optim,
                    'KL_s': KL_s,
                    'KL_z': KL_z,
                    'p_params': p_params,
                    'q_params': q_params,
                    'samples_zgen': samples_zgen,
                    'log_p_x_zgen': log_p_x_zgen,
                    'log_p_x_missing_zgen': log_p_x_missing_zgen}

        return tf_nodes
    
    # originally from GridSearch/gpu_assignment.py
    def gpu_assignment(self, gpus, allow_growth=True, per_process_gpu_memory_fraction=0.95):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpus_string = ""
        for gpu in gpus:
            gpus_string += "," + str(gpu)
        gpus_string = gpus_string[1:] # drop first comma
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_string

        config = tf.compat.v1.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = allow_growth 
        # Only allow a total fraction the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction 
        # Create a session with the above options specified.
        #k.tensorflow_backend.set_session(tf.Session(config=config))
        
        return config
    
    # originally from helpers.py
    def run_batches(self, session, tf_nodes, data, types_dict, miss_mask, true_miss_mask, n_batches, batch_size, n_epochs, epoch, train):
        'This runs the batch training for a single epoch and returns performance'
        
        avg_loss = 0.
        avg_KL_s = 0.
        avg_KL_z = 0.
        avg_loss_reg = 0.

        # Annealing of Gumbel-Softmax parameter
        tau = np.max([1.0 - (0.999/(n_epochs-50))*epoch, 1e-3])

        #Randomize the data in the mini-batches
        random_perm = np.random.RandomState(seed=42).permutation(range(np.shape(data)[0]))
        data_aux = data[random_perm,:]
        miss_mask_aux = miss_mask[random_perm,:]
        #true_miss_mask_aux = true_miss_mask[random_perm,:]
    
        for i in range(n_batches):
            data_list, miss_list = self.next_batch(data_aux, types_dict, miss_mask_aux, batch_size, index_batch=i) #Create inputs for the feed_dict
            data_list_observed = [data_list[i]*np.reshape(miss_list[:,i], [batch_size,1]) for i in range(len(data_list))] #Delete not known data (input zeros)

            #Create feed dictionary
            feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
            feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
            feedDict[tf_nodes['miss_list']] = miss_list
            feedDict[tf_nodes['tau_GS']] = tau
            feedDict[tf_nodes['zcodes']] = np.ones(batch_size).reshape((batch_size, 1))
            feedDict[tf_nodes['scodes']] = np.ones(batch_size).reshape((batch_size, 1))

            #Running VAE
            if train:
                _, loss, KL_z, KL_s, loss_reg  = session.run([tf_nodes['optim'],tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'],
                                                              tf_nodes['loss_reg']], feed_dict=feedDict)
            else:
                loss, KL_z, KL_s, loss_reg  = session.run([tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s'],tf_nodes['loss_reg']], feed_dict=feedDict)

            # Compute average loss
            avg_loss += np.mean(loss)
            avg_KL_s += np.mean(KL_s)
            avg_KL_z += np.mean(KL_z)
            avg_loss_reg += np.mean(loss_reg)
            
        return [-avg_loss/n_batches, avg_KL_s/n_batches, avg_KL_z/n_batches, avg_loss_reg/n_batches]
    
    # originally from VAE_functions.py
    def place_holder_types(self, types_file):

        #Read the types of the data from the files
        with open(types_file) as f:
            types_list = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
            
        #Create placeholders for every data type, with appropriate dimensions
        batch_data_list = []
        for i in range(len(types_list)):
            batch_data_list.append(tf.placeholder(tf.float32, shape=(None, int(types_list[i]['dim']))))
        tf.concat(batch_data_list, axis=1)
        
        #Create placeholders for every missing data type, with appropriate dimensions
        batch_data_list_observed = []
        for i in range(len(types_list)):
            batch_data_list_observed.append(tf.placeholder(tf.float32, shape=(None, int(types_list[i]['dim']))))
        tf.concat(batch_data_list_observed, axis=1)
            
        #Create placeholders for the missing data indicator variable
        miss_list = tf.placeholder(tf.int32, shape=(None, len(types_list)))
        
        #Placeholder for Gumbel-softmax parameter
        tau = tf.placeholder(tf.float32,shape=())
        zcodes=tf.placeholder(tf.float32, shape=(None,1))
        scodes=tf.placeholder(tf.int32, shape=(None,1))
        
        return batch_data_list, batch_data_list_observed, miss_list, tau, types_list, zcodes, scodes
    
    # originally from VAE_functions.py
    def batch_normalization(self, batch_data_list, types_list, miss_list): 
        normalized_data = []
        normalization_parameters = []
        
        for i,d in enumerate(batch_data_list):
            #Partition the data in missing data (0) and observed data n(1)
            missing_data, observed_data = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
            
            if types_list[i]['type'] == 'real':
                #We transform the data to a gaussian with mean 0 and std 1
                data_mean, data_var = tf.nn.moments(observed_data, 0)
                data_var = tf.clip_by_value(data_var, 1e-6, 1e20) #Avoid zero values
                aux_X = tf.nn.batch_normalization(observed_data, data_mean, data_var, offset=0.0, scale=1.0, variance_epsilon=1e-6)
                
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data,aux_X]))
                normalization_parameters.append([data_mean,data_var])
                
            #When using log-normal
            elif types_list[i]['type'] == 'pos':
                #We transform the log of the data to a gaussian with mean 0 and std 1
                observed_data_log = tf.log(1+observed_data)
                data_mean_log, data_var_log = tf.nn.moments(observed_data_log, 0)
                data_var_log = tf.clip_by_value(data_var_log, 1e-6, 1e20) #Avoid zero values
                aux_X = tf.nn.batch_normalization(observed_data_log, data_mean_log, data_var_log, offset=0.0, scale=1.0, variance_epsilon=1e-6)
                
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data,aux_X]))
                normalization_parameters.append([data_mean_log, data_var_log])
                
            elif types_list[i]['type'] == 'count':
                #Input log of the data
                aux_X = tf.log(observed_data)
                
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data,aux_X]))
                normalization_parameters.append([0.0, 1.0])
            else:
                #Don't normalize the categorical and ordinal variables
                normalized_data.append(d)
                normalization_parameters.append([0.0, 1.0]) #No normalization here
        
        return normalized_data, normalization_parameters
    
    # originally from model_HIVAE_inputDropout.py
    def encoder(self, X_list, batch_size, z_dim, s_dim, tau, weight_decay):
        samples = dict.fromkeys(['s','z','y','x'],[])
        q_params = dict()
        X = tf.concat(X_list, 1)
        
        #Create the proposal of q(s|x^o)
        samples['s'], q_params['s'] = self.s_proposal_multinomial(X, batch_size, s_dim, tau, weight_decay, reuse=None)
        
        #Create the proposal of q(z|s,x^o)
        samples['z'], q_params['z'] = self.z_proposal_GMM(X, samples['s'], batch_size, z_dim, weight_decay, reuse=None)
        
        return samples, q_params
    
    # originally from model_HIVAE_inputDropout.py
    def decoder(self, batch_data_list, miss_list, types_list, samples, normalization_params, batch_size, z_dim, y_dim, y_dim_partition, weight_decay):
        p_params = dict()
        
        #Create the distribution of p(z|s)
        p_params['z'] = self.z_distribution_GMM(samples['s'], z_dim, weight_decay, reuse=None)
        
        #Create deterministic layer y
        samples['y'] = tf.layers.dense(inputs=samples['z'], units=y_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name= 'layer_h1_', reuse=None)
        
        grouped_samples_y = self.y_partition(samples['y'], types_list, y_dim_partition)

        #Compute the parameters h_y
        theta = self.theta_estimation_from_y(grouped_samples_y, types_list, miss_list, batch_size, weight_decay, reuse=None)
        
        #Compute loglik and output of the VAE
        log_p_x, log_p_x_missing, samples['x'], p_params['x'] = self.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params)
            
        return theta, samples, p_params, log_p_x, log_p_x_missing
    
    # originally from model_HIVAE_inputDropout.py
    def cost_function(self, log_p_x, p_params, q_params, z_dim, s_dim):
        #KL(q(s|x)|p(s))
        log_pi = q_params['s']
        pi_param = tf.nn.softmax(log_pi)
        KL_s = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=pi_param) + tf.log(float(s_dim))
        
        #KL(q(z|s,x)|p(z|s))
        # to implement: if flagged iteration, take pred z instead
        mean_pz, log_var_pz = p_params['z']
        mean_qz, log_var_qz = q_params['z']
        KL_z = -0.5*z_dim + 0.5*tf.reduce_sum(tf.exp(log_var_qz-log_var_pz)+tf.square(mean_pz-mean_qz)/tf.exp(log_var_pz)-log_var_qz+log_var_pz, 1)
        
        #Eq[log_p(x|y)]
        loss_reconstruction = tf.reduce_sum(log_p_x, 0)
        
        #Complete ELBO
        ELBO = tf.reduce_mean(loss_reconstruction - KL_z - KL_s, 0)
        
        return ELBO, loss_reconstruction, KL_z, KL_s
    
    # originally from model_HIVAE_inputDropout.py
    def fixed_decoder(self, batch_data_list, X_list, miss_list, types_list, batch_size, y_dim, y_dim_partition, s_dim, normalization_params, zcodes, scodes, weight_decay):
        
        samples_test = dict.fromkeys(['s','z','y','x'],[])
        test_params = dict()
        #X = tf.concat(X_list,1)
        
        #Create the proposal of q(s|x^o)
        samples_test['s'] = tf.one_hot(scodes, depth=s_dim)
        
        # set fixed z
        samples_test['z'] = zcodes
        
        #Create deterministic layer y
        samples_test['y'] = tf.layers.dense(inputs=samples_test['z'], units=y_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name= 'layer_h1_', reuse=True)
        
        grouped_samples_y = self.y_partition(samples_test['y'], types_list, y_dim_partition)
        
        #Compute the parameters h_y
        theta = self.theta_estimation_from_y(grouped_samples_y, types_list, miss_list, batch_size, weight_decay, reuse=True)
        
        #Compute loglik and output of the VAE
        log_p_x, log_p_x_missing, samples_test['x'], test_params['x'] = self.loglik_evaluation(batch_data_list, types_list, miss_list, theta, normalization_params)
        
        return samples_test, test_params, log_p_x, log_p_x_missing
    
    # originally from read_functions.py
    def next_batch(self, data, types_dict, miss_mask, batch_size, index_batch):
        #Create minibath
        batch_xs = data[index_batch*batch_size:(index_batch+1)*batch_size, :]
        
        #Slipt variables of the batches
        data_list = []
        initial_index = 0
        for d in types_dict:
            dim = int(d['dim'])
            data_list.append(batch_xs[:, initial_index:initial_index+dim])
            initial_index += dim
        
        #Missing data
        miss_list = miss_mask[index_batch*batch_size:(index_batch+1)*batch_size, :]

        return data_list, miss_list
    
    # originally from VAE_functions.py
    def s_proposal_multinomial(self, X, batch_size, s_dim, tau, weight_decay, reuse):
        
        #We propose a categorical distribution to create a GMM for the latent space z
        log_pi = tf.layers.dense(inputs=X, units=s_dim, activation=None,kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='layer_1_enc_s', reuse=reuse)
        
        #Gumbel-softmax trick
        U = -tf.log(-tf.log(tf.random_uniform([batch_size,s_dim], seed=42)))
        samples_s = tf.nn.softmax((log_pi + U)/tau)
        
        return samples_s, log_pi
    
    # originally from VAE_functions.py
    def z_proposal_GMM(self, X, samples_s, batch_size, z_dim, weight_decay, reuse):
        
        #We propose a GMM for z
        mean_qz = tf.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='layer_1_mean_enc_z', reuse=reuse)
        
        log_var_qz = tf.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='layer_1_logvar_enc_z', reuse=reuse)
        
        # Avoid numerical problems
        log_var_qz = tf.clip_by_value(log_var_qz, -15.0, 15.0)
        # Rep-trick
        eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32, seed=42)
        samples_z = mean_qz + tf.multiply(tf.exp(log_var_qz/2), eps)
        
        return samples_z, [mean_qz, log_var_qz]
    
    # originally from VAE_functions.py
    def z_distribution_GMM(self, samples_s, z_dim, weight_decay, reuse):
        
        #We propose a GMM for z
        mean_pz = tf.layers.dense(inputs=samples_s, units=z_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name= 'layer_1_mean_dec_z', reuse=reuse)
        
        log_var_pz = tf.zeros([tf.shape(samples_s)[0], z_dim])
        
        # Avoid numerical problems
        log_var_pz = tf.clip_by_value(log_var_pz, -15.0, 15.0)
        
        return mean_pz, log_var_pz
    
    # originally from VAE_functions.py
    def y_partition(self, samples_y, types_list, y_dim_partition):
        
        grouped_samples_y = []
        # First element must be 0 and the length of the partition vector must be len(types_dict)+1
        if len(y_dim_partition) != len(types_list):
            raise Exception("[*] The length of the partition vector must match the number of variables in the data + 1")
            
        # Insert a 0 at the beginning of the cumsum vector
        partition_vector_cumsum = np.insert(np.cumsum(y_dim_partition), 0, 0)
        for i in range(len(types_list)):
            grouped_samples_y.append(samples_y[:, partition_vector_cumsum[i]:partition_vector_cumsum[i+1]])
        
        return grouped_samples_y
    
    # originally from VAE_functions.py
    def theta_estimation_from_y(self, samples_y, types_list, miss_list, batch_size, weight_decay, reuse):
        theta = []
        # Independet yd -> Compute p(xd|yd)
        for i, d in enumerate(samples_y):
            #Partition the data in missing data (0) and observed data (1)
            missing_y, observed_y = tf.dynamic_partition(d, miss_list[:,i], num_partitions=2)
            condition_indices = tf.dynamic_partition(tf.range(tf.shape(d)[0]), miss_list[:,i], num_partitions=2)
            
            # Different layer models for each type of variable
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
            
            theta.append(params)
                
        return theta
    
    # originally from VAE_functions.py
    def loglik_evaluation(self, batch_data_list, types_list, miss_list, theta, normalization_params):
        log_p_x = []
        log_p_x_missing = []
        samples_x = []
        params_x = []
        
        # Independet yd -> Compute log(p(xd|yd))
        for i, d in enumerate(batch_data_list):
            # Select the likelihood for the types of variables
            loglik_models = c_loglik_models_missing_normalize([d,miss_list[:,i]], types_list[i], theta[i], normalization_params[i])
            loglik_function = getattr(loglik_models, 'loglik_' + types_list[i]['type'])

            out = loglik_function()
                
            log_p_x.append(out['log_p_x'])
            log_p_x_missing.append(out['log_p_x_missing']) #Test-loglik element
            samples_x.append(out['samples'])
            params_x.append(out['params'])
            
        return log_p_x, log_p_x_missing, samples_x, params_x
    
    # originally from VAE_functions.py
    def theta_real(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        
        # Mean layer
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), 
                                        weight_decay=weight_decay, reuse=reuse)
        # Sigma Layer
        h2_sigma = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i),
                                        weight_decay=weight_decay, reuse=reuse)
        
        return [h2_mean, h2_sigma]
    
    # originally from VAE_functions.py
    def theta_pos(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        
        # Mean layer
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), 
                                        weight_decay=weight_decay,reuse=reuse)
        # Sigma Layer
        h2_sigma = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i),
                                        weight_decay=weight_decay, reuse=reuse)
        
        return [h2_mean, h2_sigma]
    
    # originally from VAE_functions.py
    def theta_count(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        
        # Lambda Layer
        h2_lambda = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=types_list[i]['dim'], name='layer_h2' + str(i),
                                        weight_decay=weight_decay, reuse=reuse)
        
        return h2_lambda
    
    # originally from VAE_functions.py
    def theta_cat(self, observed_y, missing_y, condition_indices, types_list, batch_size, i, weight_decay, reuse):
        
        # Log pi layer, with zeros in the first value to avoid the identificability problem
        h2_log_pi_partial = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, 
                                                    name='layer_h2' + str(i), weight_decay=weight_decay,reuse=reuse)
        h2_log_pi = tf.concat([tf.zeros([batch_size,1]), h2_log_pi_partial], 1)
        
        return h2_log_pi
    
    # originally from VAE_functions.py
    def theta_ordinal(self, observed_y, missing_y, condition_indices, types_list, i, weight_decay, reuse):
        
        # Theta layer, Dimension of ordinal - 1
        h2_theta = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=int(types_list[i]['dim'])-1, 
                                            name='layer_h2' + str(i), weight_decay=weight_decay, reuse=reuse)
        # Mean layer, a single value
        h2_mean = self.observed_data_layer(observed_y, missing_y, condition_indices, output_dim=1, name='layer_h2_sigma' + str(i),
                                            weight_decay=weight_decay,reuse=reuse)
        
        return [h2_theta, h2_mean]
    
    # originally from VAE_functions.py
    def observed_data_layer(self, observed_data, missing_data, condition_indices, output_dim, name, weight_decay, reuse): 
        # Train a layer with the observed data and reuse it for the missing data
        obs_output = tf.layers.dense(inputs=observed_data, units=output_dim, activation=None, 
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                    name=name, reuse=reuse, trainable=True)
        miss_output = tf.layers.dense(inputs=missing_data, units=output_dim, activation=None, 
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=42),
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                    name=name, reuse=True, trainable=False)
        # Join back the data
        output = tf.dynamic_stitch(condition_indices, [miss_output,obs_output])
        
        return output


#if __name__ == '__main__':
    #grid_search = c_HIVAE_GridSearch()
    #grid_search.hyperopt_HIVAE()

    #modelling = c_HIVAE_Modelling()
    #modelling.train_HIVAE()
    #modelling.decode_HIVAE()
