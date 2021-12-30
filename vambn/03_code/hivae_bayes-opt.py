import os
import re
import csv
import argparse
import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import configparser
import optuna 
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


class c_HIVAE_BayesianOptimization:
    """
    Perform hyperparameter optimization on HIVAE using the bayesian optimization method. 
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
        
        # assign configs as variables and dict
        self.bayes_opt_n_trial = int(cp.get('HyperparameterOptimization', 'bo_n_trial'))
        self.data_python = cp.get('HyperparameterOptimization', 'path_data_python')
        self.results = cp.get('HyperparameterOptimization', 'path_results') 
        self.max_epochs = int(cp.get('HyperparameterOptimization', 'bo_max_epochs'))
        self.final_n_epoch = 0
        self.cross_val_k_fold = int(cp.get('HyperparameterOptimization', 'bo_cross_val_k_fold'))
        self.early_stop_patience = int(cp.get('HyperparameterOptimization', 'bo_early_stop_patience'))
        self.seed = int(cp.get('HyperparameterOptimization', 'bo_seed_n'))
        self.search_options = {
            'ydims': [int(i) for i in cp.get('HyperparameterOptimization', 'bo_y_dimensions').split(',')],
            'nbatch': [int(i) for i in cp.get('HyperparameterOptimization', 'bo_batch_size').split(',')],
            'lrates': [float(i) for i in cp.get('HyperparameterOptimization', 'bo_learning_rates').split(',')],
            'wdecay': [float(i) for i in cp.get('HyperparameterOptimization', 'bo_weight_decay').split(',')],
            'beta1': [float(i) for i in cp.get('HyperparameterOptimization', 'bo_adam_beta_1').split(',')],
            'beta2': [float(i) for i in cp.get('HyperparameterOptimization', 'bo_adam_beta_2').split(',')]
        }
        self.current_data_file = ''
        print(f'[*] Config file search options: {self.search_options}')

    
    def objective(self, trial):
        """
        Perform hyperparameter optimization using the Bayesian optimization method.
        """

        # define search space
        ydims = trial.suggest_int("ydims", self.search_options['ydims'][0], self.search_options['ydims'][1], self.search_options['ydims'][2])
        nbatch = trial.suggest_int("nbatch", self.search_options['nbatch'][0], self.search_options['nbatch'][1], self.search_options['nbatch'][2])
        lrates = trial.suggest_float("lrates", self.search_options['lrates'][0], self.search_options['lrates'][1], log=True)
        wdecay = trial.suggest_float("wdecay", self.search_options['wdecay'][0], self.search_options['wdecay'][1], log=True)
        beta1 = trial.suggest_float("beta1", self.search_options['beta1'][0], self.search_options['beta1'][1], log=True)
        beta2 = trial.suggest_float("beta2", self.search_options['beta2'][0], self.search_options['beta2'][1], log=True)
        opt = dict({"ydims": ydims, "lrates": lrates, "wdecay": wdecay, "nbatch": nbatch, "beta1": beta1, "beta2": beta2})

        # set model config
        settings = self.set_settings(self.current_data_file, opt)
        name = f"YD{opt['ydims']}_NB{opt['nbatch']}_LR{opt['lrates']}_WD{opt['wdecay']}_B1{opt['beta1']}_B2_{opt['beta2']}"

        # train model
        loss = self.run_network(settings, name, n_splits=self.cross_val_k_fold)
        
        # return the loss of this training
        return loss


    # originally from GridSearch_ADNI.ipynb
    def hyperopt_HIVAE(self):
        """
        Perform hyperparameter optimization on all search option combinations.
        """

        # read data files for every variable group
        files = [i for i in os.listdir(self.data_python) if not '_type' in i and not '_missing' in i]

        # for each variable group data file,
        # for each search option combination,
        # set respective configs for grid search and perform hyperopt,
        # append all grid search losses in a list
        l_files = []
        l_params = []
        l_epochs = []
        for f in files:
            print(f'\n[*] Currently performing bayesian hyperopt for: {f}. Please wait...')

            self.current_data_file = f
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective, n_trials=self.bayes_opt_n_trial)

            trial = study.best_trial
            print(f"[*] Best loss for module {self.current_data_file}: {trial.value}")
            print(f"[*] Best hyperparameters for module {self.current_data_file}: {trial.params}")
        
            l_files.append(trial.value)
            l_params.append(trial.params)
            l_epochs.append(self.final_n_epoch)

        # list of losses and best params of all variable groups
        losses = list(zip(files, l_files)) 

        # list of indices of the minimum loss per variable group, ignoring NaNs
        minloss = [np.nanmin(losses[f][1]) for f in range(len(files))]

        # create a dataframe containing the best hyperparams in each variable group
        # append columns to the dataframe: data file name, losses, final used epochs
        output = pd.DataFrame(l_params)
        output['files'] = files
        output['loss'] = minloss
        output['epochs'] = l_epochs

        # save dataframe as a csv file and print on terminal
        output.to_csv(self.results, index=True)
        print(output)
        print('[*] Bayesian optimization hyperopt script completed.')
    

    # originally from GridSearch_ADNI.ipynb
    def set_settings(self, f, opts):
        """
        Set configs for a single bayes opt experiment.
        Return settings as a string.
        """
        
        # set file name for data, missing and types files
        inputf = re.sub('.csv', '', f)
        missf = inputf + '_missing.csv'
        typef = inputf + '_types.csv'
        
        # create settings as a string variable
        settings = f"--epochs {self.max_epochs} \
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
                    --weight_decay {opts['wdecay']} \
                    --adam_beta_1 {opts['beta1']} \
                    --adam_beta_2 {opts['beta2']}"
        
        # return all settings as a string
        return settings
    

    # originally from helpers.py
    def run_network(self, settings, name, n_splits=3):
        """
        Run Bayesian Optimization hyperparameter optimization.
        Return the nanmean test loss of all K-Fold outcomes.
        """

        # split the setting string and convert it into a series of arguments
        argvals = settings.split()
        args = self.getArgs(argvals)
        print(f"[*] Current hyperopt config: {name}")
        
        # load data, types, missing and true missing masks as variables
        data, types_dict, miss_mask, true_miss_mask, n_samples = self.read_data(args.data_file, args.types_file, args.miss_file, args.true_miss_file)

        # compute a final missing mask
        miss_mask = np.multiply(miss_mask, true_miss_mask) 
        
        # split data with K-Fold
        # for each fold, run HIVAE, return train and test score
        # append all test score
        kf = KFold(n_splits=n_splits, shuffle=True)
        score_keep = []
        n_epochs = []
        fold = 0
        for train_idx, test_idx in kf.split(data):
            fold += 1
            score = self.run_epochs(args, data, train_idx, test_idx, types_dict, miss_mask, true_miss_mask) 

            print(f"[*] Score for fold {fold}: Train - {score[0]:.3f} :: Test - {score[1]:.3f}")

            # append score if not nan
            if not np.isnan(score[1]):
                score_keep.append(score[1]) 
                # append the final n_epoch executed
                n_epochs.append(score[2])
        
        # if score list is not empty (all NaNs), 
        # return the correct final n_epoch used and mean score
        # else,
        # return large mean loss value (to not be picked as best hyperparam later) 
        if len(score_keep) >= 1:
            # record the n_epoch executed on the fold that yielded the lowest loss score
            self.final_n_epoch = n_epochs[score_keep.index(min(score_keep))]
            # return the mean of test scores across all folds    
            mean_score = np.mean(score_keep) 
        else:
            # return a filler number
            self.final_n_epoch = 0
            # return a large loss as a filler number
            mean_score = 99999
        return mean_score
    

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
        parser.add_argument('--train', type=int, default=1, help='Training model flag')
        parser.add_argument('--display', type=int, default=1, help='Display option flag')
        parser.add_argument('--save', type=int, default=1000, help='Save variables every save iterations')
        parser.add_argument('--restore', type=int, default=0, help='To restore session, to keep training or evaluation') 
        parser.add_argument('--plot', type=int, default=1, help='Plot results flag')
        parser.add_argument('--dim_latent_s', type=int, default=10, help='Dimension of the categorical space')
        parser.add_argument('--dim_latent_z', type=int, default=2, help='Dimension of the Z latent space')
        parser.add_argument('--dim_latent_y', type=int, default=10, help='Dimension of the Y latent space')
        parser.add_argument('--dim_latent_y_partition',type=int, nargs='+', help='Partition of the Y latent space')
        parser.add_argument('--miss_percentage_train', type=float, default=0.0, help='Percentage of missing data in training')
        parser.add_argument('--miss_percentage_test', type=float, default=0.0, help='Percentage of missing data in test')
        parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
        parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
        parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
        parser.add_argument('--miss_file', type=str, default='Missing_test.csv', help='File with the missing indexes mask')
        parser.add_argument('--true_miss_file', type=str, help='File with the missing indexes when there are NaN in the data')
        parser.add_argument('--learning_rate', type=float, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, help='L2: Weight decay')
        parser.add_argument('--activation', type=str, default='none', help='Activation function')
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
        miss_mask = np.ones([n_samples, n_variables])

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
    

    # originally from helpers.py
    def run_epochs(self, args, data, train_idx, test_idx, types_dict, miss_mask, true_miss_mask):
        """
        Create a TensorFlow graph, run train and test batches for a predefined n_epoch.
        Return the train loss and test loss of the very last epoch as a list tuple.
        """
        
        # create a TF graph for HIVAE
        sess_HVAE = tf.Graph() 
        with sess_HVAE.as_default():
            tf_nodes = self.HVAE_graph(args.types_file, args.batch_size, learning_rate=args.learning_rate, beta1=args.adam_beta_1, beta2=args.adam_beta_2, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, weight_decay=args.weight_decay, y_dim_partition=args.dim_latent_y_partition)

        # get an integer number of train and test batches
        n_batches_train = int(np.floor(len(train_idx)/args.batch_size)) 
        n_batches_test = int(np.floor(len(test_idx)/args.batch_size)) 
        
        # assign gpu for HIVAE training (if any)
        # run a HIVAE session
        config = self.gpu_assignment([0,1,2])
        with tf.compat.v1.Session(graph=sess_HVAE, config=config) as session: 
            print("[*] Initizalizing Variables ...")
            print(f"[*] Train size: {len(train_idx)} :: Test size: {len(test_idx)}")
            
            tf.compat.v1.global_variables_initializer().run() 
            
            train_loss_epoch = []
            train_KL_s_epoch = []
            train_KL_z_epoch = []
            test_loss_epoch = []
            test_KL_s_epoch = []
            test_KL_z_epoch = []
            no_improvement_counter = 0

            # for every epoch
            for epoch in range(args.epochs):

                # run a batch with training data
                # append train loss, train KL_s and KL_z of the epoch
                losses_train = self.run_batches(session, tf_nodes, data[train_idx], types_dict, miss_mask[train_idx], true_miss_mask[train_idx], n_batches_train, args.batch_size, args.epochs, epoch, train=True)
                train_loss_epoch.append(losses_train[0])
                train_KL_s_epoch.append(losses_train[1])
                train_KL_z_epoch.append(losses_train[2])

                # run a batch with test data
                # append test loss, test KL_s and KL_z of the epoch
                losses_test = self.run_batches(session, tf_nodes, data[test_idx], types_dict, miss_mask[test_idx], true_miss_mask[test_idx], n_batches_test, args.batch_size, args.epochs, epoch, train=False)
                test_loss_epoch.append(losses_test[0])
                test_KL_s_epoch.append(losses_test[1])
                test_KL_z_epoch.append(losses_test[2])

                # no improvement check: if the current train loss has no improvement to its previous loss, increase the counter
                if len(train_loss_epoch) >= 2:
                    if (train_loss_epoch[-1] >= train_loss_epoch[-2]):
                        no_improvement_counter += 1
                    else: 
                        no_improvement_counter = 0

                # early stopping check: if no_improvement counter is equal to early_stop_patience, end the training loop
                if (no_improvement_counter >= self.early_stop_patience):
                    print(f"[*] Train loss has not improved ({train_loss_epoch[-1]}) for {no_improvement_counter} times. Early stopping at epoch {self.final_n_epoch}/{args.epochs}.")
                    break
        
        # return the train loss and test loss of the very last epoch
        return [train_loss_epoch[-1], test_loss_epoch[-1], epoch+1]
    

    # originally from graph_new.py
    def HVAE_graph(self, types_file, batch_size, learning_rate=1e-3, beta1=0.9, beta2=0.999, z_dim=2, y_dim=1, s_dim=2, weight_decay=0, y_dim_partition=[]):
        """
        Create TensorFlow graph nodes for HIVAE.
        Return TF nodes (HIVAE model architecture).
        """
    
        # load placeholders
        batch_data_list, batch_data_list_observed, miss_list, tau, types_list, zcodes, scodes = self.place_holder_types(types_file)
        
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
        samples_zgen, test_params_zgen, log_p_x_zgen, log_p_x_missing_zgen = self.fixed_decoder(batch_data_list, X_list, miss_list, types_list, batch_size, y_dim_output, y_dim_partition, s_dim, normalization_params, zcodes, scodes, weight_decay)

        # packing all defined nodes as a dict
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
                    'optim': optim,
                    'KL_s': KL_s,
                    'KL_z': KL_z,
                    'p_params': p_params,
                    'q_params': q_params,
                    'samples_zgen': samples_zgen,
                    'log_p_x_zgen': log_p_x_zgen,
                    'log_p_x_missing_zgen': log_p_x_missing_zgen}

        # return HIVAE model architecture
        return tf_nodes
    

    # originally from GridSearch/gpu_assignment.py
    def gpu_assignment(self, gpus, allow_growth=True, per_process_gpu_memory_fraction=0.95):
        """
        Define the config for using GPU during model training.
        Return config.
        """
        
        # define CUDA device order
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpus_string = ""

        # for each available gpu, append values as string
        # assign the string as CUDA visible devices
        for gpu in gpus:
            gpus_string += "," + str(gpu)
        gpus_string = gpus_string[1:] # drop first comma
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_string

        # define GPU configs
        config = tf.compat.v1.ConfigProto()
        # don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = allow_growth 
        # only allow a total fraction the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction 

        # return gpu config
        return config
    

    # originally from helpers.py
    def run_batches(self, session, tf_nodes, data, types_dict, miss_mask, true_miss_mask, n_batches, batch_size, n_epochs, epoch, train):
        """
        Execute model training with batch data for a single epoch.
        Return -average loss/n_batch, avg KL s/n_batch, and avg KL z/n_batch.
        """
        
        # default values
        avg_loss = 0.
        avg_KL_s = 0.
        avg_KL_z = 0.

        # annealing of Gumbel-Softmax parameter
        tau = np.max([1.0 - (0.999/(n_epochs-50+0.0001))*epoch, 1e-3]) # add 0.0001 to avoid zero division error

        # randomize the order of data and miss_mask in the mini-batches
        random_perm = np.random.RandomState(seed=self.seed).permutation(range(np.shape(data)[0]))
        data_aux = data[random_perm, :]
        miss_mask_aux = miss_mask[random_perm, :]

        # for each data batch
        for i in range(n_batches):

            # prepares batch data and miss list for the feed dictionary
            data_list, miss_list = self.next_batch(data_aux, types_dict, miss_mask_aux, batch_size, index_batch=i) 

            # delete unknown data (input zeros)
            data_list_observed = [data_list[i] * np.reshape(miss_list[:,i], [batch_size,1]) for i in range(len(data_list))] 

            # create feed dictionary
            feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
            feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_observed'], data_list_observed)})
            feedDict[tf_nodes['miss_list']] = miss_list
            feedDict[tf_nodes['tau_GS']] = tau
            feedDict[tf_nodes['zcodes']] = np.ones(batch_size).reshape((batch_size, 1))
            feedDict[tf_nodes['scodes']] = np.ones(batch_size).reshape((batch_size, 1))

            # run HIVAE (either train or test)
            if train:
                _, loss, KL_z, KL_s = session.run([tf_nodes['optim'], tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s']], feed_dict=feedDict)
            else:
                loss, KL_z, KL_s = session.run([tf_nodes['loss_re'], tf_nodes['KL_z'], tf_nodes['KL_s']], feed_dict=feedDict)

            # compute average loss, average KL s, and average KL z
            avg_loss += np.mean(loss)
            avg_KL_s += np.mean(KL_s)
            avg_KL_z += np.mean(KL_z)

        # return -average loss, avg KL s, and avg KL z divided by the number of batches
        return [-avg_loss/n_batches, avg_KL_s/n_batches, avg_KL_z/n_batches]
    

    # originally from VAE_functions.py
    def place_holder_types(self, types_file):
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
            batch_data_list.append(tf.compat.v1.placeholder(shape=[None, int(types_list[i]['dim'])], dtype=tf.float32))
        tf.concat(batch_data_list, axis=1)
        
        # create placeholders for every missing data type, with appropriate dimensions
        batch_data_list_observed = []
        for i in range(len(types_list)):
            batch_data_list_observed.append(tf.compat.v1.placeholder(shape=[None, int(types_list[i]['dim'])], dtype=tf.float32))
        tf.concat(batch_data_list_observed, axis=1)
            
        # create placeholders for the missing data indicator variable
        miss_list = tf.compat.v1.placeholder(shape=[None, len(types_list)], dtype=tf.int32)
        
        # create placeholders for Gumbel-softmax parameters
        tau = tf.compat.v1.placeholder(shape=[], dtype=tf.float32)
        zcodes = tf.compat.v1.placeholder(shape=[None,1], dtype=tf.float32)
        scodes = tf.compat.v1.placeholder(shape=[None,1], dtype=tf.int32)
        
        # return all placeholders
        return batch_data_list, batch_data_list_observed, miss_list, tau, types_list, zcodes, scodes
    

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
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data,aux_X]))
                normalization_parameters.append([data_mean_log, data_var_log])
            
            # if the data type = count
            elif types_list[i]['type'] == 'count':
                
                # use the log of the data
                aux_X = tf.math.log(observed_data)
                
                # append batch normed data and its mean and variance 
                # mean and variance are 0.0 and 1.0
                normalized_data.append(tf.dynamic_stitch(condition_indices, [missing_data,aux_X]))
                normalization_parameters.append([0.0, 1.0])
            
            # if the data type is other than real, pos real and count (expect categorical and ordinal)
            else:
                # don't normalize the categorical and ordinal variables, append as is
                # no normalization here
                normalized_data.append(d)
                normalization_parameters.append([0.0, 1.0]) 
        
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
    def fixed_decoder(self, batch_data_list, X_list, miss_list, types_list, batch_size, y_dim, y_dim_partition, s_dim, normalization_params, zcodes, scodes, weight_decay):
        """
        Create the decoder part of HIVAE model architecture (with fixed z).
        Return decoder features.
        """
        
        # define test samples and test params
        samples_test = dict.fromkeys(['s','z','y','x'], [])
        test_params = dict()
        
        # create the proposal of q(s|x^o)
        samples_test['s'] = tf.one_hot(scodes, depth=s_dim)
        
        # set fixed z
        samples_test['z'] = zcodes
        
        # create deterministic layer y
        samples_test['y'] = tf.compat.v1.layers.dense(inputs=samples_test['z'], units=y_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name= 'layer_h1_', reuse=True)
        
        grouped_samples_y = self.y_partition(samples_test['y'], types_list, y_dim_partition)
        
        # compute the parameters h_y
        theta = self.theta_estimation_from_y(grouped_samples_y, types_list, miss_list, batch_size, weight_decay, reuse=True)
        
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
        
        log_var_qz = tf.compat.v1.layers.dense(inputs=tf.concat([X,samples_s],1), units=z_dim, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=0.05, seed=self.seed), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='layer_1_logvar_enc_z', reuse=reuse)
        
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
