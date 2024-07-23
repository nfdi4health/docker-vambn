### Snakemake Configs

- `output_dir`: Directory to store all outputs.
- `bn`: Bayesian network settings
  - `refactor`: Convert s-encodings and some demographic variables to factors
  - `discretize`: Should the data be discretized?
  - `aux`: Use auxiliary information (currently without use, and will not be needed in future)
  - `cv_runs`: Number of cross-validation runs to find the best algorithm.
  - `cv_restart`: Number of cross-validation restarts.
  - `fit`: Fitting method.
  - `maxp`: Maximum parents for a node.
  - `loss`: Loss function.
  - `score`: Scoring metric.
  - `n_bootstrap`: Number of bootstrap samples.
  - `seed`: Random seed.
- `variants`: Model variants. Can be *modular* and *modulary*. *modular* is the ModularHIVAE without shared y layer and serves only for validation purposes. *modulary* is sufficient in most scenarios.
- `excluded_datasets`: Datasets to ignore. Should be a list.
- `exclusive_dataset`: Use only one dataset although multiple might be present.
- `loss_modes`: Loss modes, like "equal" or "weighted". Specifies if the losses from the different HIVAE-modules are equally aggregated or weighted.
- `cluster_modules`: Cluster module settings, e.g., R version.

### General Configs

- `seed`: Random seed.
- `eval_batch_size`: Batch size for evaluations.
- `logging`: Logging settings.

  - `level`: Log level. Should be numeric. 10 for debug, 20 for info, 30 for warning, 40 for error, 50 for critical.
  - `mlflow`: MLFlow configs.
    - `use`: indicate whether mlflow should be used.
    - `tracking_uri`: set the url to your mlflow server
    - `experiment_name`: specify any name you want the experiments to be logged with
- `lightning`: PyTorch Lightning settings.

  - `log_every_n_steps`: Frequency of logging steps.
  - `fast_dev_run`: Quick model run for debugging.
  - `use_gpu`: Use GPU for training.
  - `devices`: Number of GPUs to use.
  - `gradient_clip_val`: Maximum gradient value.
  - `gradient_clip_algorithm`: Algorithm to use for gradient clipping.
- `callbacks`: Callback settings

  - `early_stopping`: Whether to use early stopping in CV. Currently the median best epoch from the different folds is taken for the final fit.

### Optimization Configs

- `folds`: Folds for cross-validation for finding optimal HI-VAE parameters.
- `n_modular_trials`: Number of modular trials.
- `n_traditional_trials`: Number of traditional trials.
- `dim_s_lower`, `dim_s_upper`, `dim_s_step`: Dimensions for 's' range.
- `equal_dim_s`: Use the same dimensions for all data modules (only applicable to modular HI-VAE)
- `dim_y_lower`, `dim_y_upper`, `dim_y_step`: Dimensions for 'y' range.
- `y_layer_dropout_lower`, `y_layer_dropout_upper`: Dropout for 'y'.
- `batch_size_options`: Available batch sizes.
- `learning_rate_lower`, `learning_rate_upper`: Learning rate range.
- `weight_decay_options`: Weight decay options.
- epoch_options

### LSTM Layer (For Longitudinal Data)

- `lstm_layer_options`: Number of LSTM layers in the Encoder.

### Training Configs

- `apply_minmax_scaling`: Apply MinMaxScaler to data
- `apply_batch_normalization`: Use batch normalization like in the orignal HI-VAE (unstable during recent tests)
