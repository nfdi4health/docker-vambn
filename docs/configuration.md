# Configuration Documentation

This document provides detailed information on the configuration options available in the provided configuration file. The configuration is divided into several sections: `snakemake`, `general`, `optimization`, and `training`. Each section and its respective options are described below.

## Snakemake Configuration

The `snakemake` section contains settings related to the Snakemake workflow management system.

### Options

- **use_slurm**: `false`
    - Type: Boolean
    - Description: Flag to indicate if SLURM is available (e.g., on Loewenburg).
- **with_gan**: `false`
    - Type: Boolean
    - Description: Flag to indicate if a GAN variant should be used in addition to the normal training.
- **with_mtl**: `false`
    - Type: Boolean
    - Description: Flag to indicate if multitask learning variants should be used in addition to the normal training.
- **output_dir**: `"reports"`
    - Type: String
    - Description: Output directory for all generated output files.
- **bn**: Bayesian network configuration; typically does not need to be changed.
    - **refactor**: `true`
        - Type: Boolean
    - **cv_runs**: `5`
        - Type: Integer
    - **cv_restart**: `5`
        - Type: Integer
    - **fit**: `"mle-cg"`
        - Type: String
    - **maxp**: `5`
        - Type: Integer
    - **loss**: `null`
        - Type: Null
    - **score**: `"bic-cg"`
        - Type: String
    - **folds**: `3`
        - Type: Integer
    - **n_bootstrap**: `500`
        - Type: Integer
    - **seed**: `42`
        - Type: Integer
- **excluded_datasets**: List of datasets to be excluded from the pipeline.
    - Type: List of Strings
    - Description: List of datasets to be excluded from the pipeline. The datasets are defined by their folder name in the `data/raw` directory.
    - **Example**: `["texas"]`
- **exclusive_dataset**: `null`
    - Type: Null or String
    - Description: Define this if you only want to run the pipeline for a single dataset for e.g. testing purposes.
- **cluster_modules**:
    - **R**: `null`
        - Type: Null or String
        - Description: Optional cluster module for R (e.g., `"R/4.0.3"`). Not required if R is available on the system or in a Conda environment.
- **r_env**: `"/path/to/R.yaml"`
    - Type: String
    - Description: Path to the R environment file. The file is used to setup an conda environment for R. If this is used snakemake needs to be run with the `--use-conda` flag.

## General Configuration

The `general` section contains general settings for the application.

### Options

- **seed**: `42`
    - Type: Integer
    - Description: Seed for reproducibility.
- **eval_batch_size**: `64`
    - Type: Integer
    - Description: Batch size for evaluation. Does not affect training and is only restricted by the available memory.
- **device**: `"cpu"`
    - Type: String
    - Description: Device to use for training. Use `"cuda"` for GPU training.
- **optuna_db**: `"postgresql://localhost/optuna"`
    - Type: String or Null
    - Description: Database connection for Optuna. If not available, set to `null` to use SQLite databases. PostgreSQL is recommended since all results would be stored in a single database.
- **logging**:
    - **level**: `20`
        - Type: Integer
        - Description: Logging level. Default is `20` (INFO). Other options are `10` (DEBUG), `30` (WARNING), `40` (ERROR), and `50` (CRITICAL).
    - **mlflow**:
        - **use**: `false`
            - Type: Boolean
            - Description: Flag to indicate if MLflow should be used for logging.
        - **tracking_uri**: `"http://localhost:5000"`
            - Type: String
            - Description: URI for the MLflow tracking server. Can be a local or remote server.
        - **experiment_name**: `"VAMBN2"`
            - Type: String
            - Description: Name of the MLflow experiment.

## Optimization Configuration

The `optimization` section contains settings related to the optimization process.

### Options

- **folds**: `3`
    - Type: Integer
- **n_traditional_trials**: `20`
    - Type: Integer
- **n_modular_trials**: `20`
    - Type: Integer
- **s_dim_lower**: `1`
    - Type: Integer
- **s_dim_upper**: `5`
    - Type: Integer
- **s_dim_step**: `1`
    - Type: Integer
- **fixed_s_dim**: `false`
    - Type: Boolean
- **y_dim_lower**: `1`
    - Type: Integer
- **y_dim_upper**: `5`
    - Type: Integer
- **y_dim_step**: `1`
    - Type: Integer
- **fixed_y_dim**: `false`
    - Type: Boolean
- **latent_dim_lower**: `1`
    - Type: Integer
- **latent_dim_upper**: `5`
    - Type: Integer
- **latent_dim_step**: `1`
    - Type: Integer
- **batch_size_lower_n**: `4`
    - Type: Integer
- **batch_size_upper_n**: `8`
    - Type: Integer
- **max_epochs**: `2500`
    - Type: Integer
    - Description: Maximum number of epochs. Currently, early stopping is used.
- **learning_rate_lower**: `0.0001`
    - Type: Float
- **learning_rate_upper**: `0.1`
    - Type: Float
- **fixed_learning_rate**: `true`
    - Type: Boolean
- **lstm_layers_lower**: `1`
    - Type: Integer
- **lstm_layers_upper**: `4`
    - Type: Integer
- **lstm_layers_step**: `1`
    - Type: Integer
- **use_relative_correlation_error_for_optimization**: `false`
    - Type: Boolean
    - Description: Flag to indicate if the relative correlation error should be used as optuna metric.
- **use_auc_for_optimization**: `false`
    - Type: Boolean
    - Description: Flag to indicate if the Area under the ROC curve should be used as optuna metric.

## Training Configuration

The `training` section contains settings related to the training process.

### Options

- **use_imputation_layer**: `true`
    - Type: Boolean
    - Description: Flag to indicate if the imputation layer should be used.
