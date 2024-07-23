# Setup

## Manual Installation

### Python Dependencies

To get started, ensure you have Python 3.11 and [Poetry](https://python-poetry.org/docs/) installed. If Python 3.11 isn't installed, consider using [Pyenv](https://github.com/pyenv/pyenv) or [Conda](https://github.com/conda-forge/miniforge) to get the correct version. Then, install dependencies with the following commands:

```bash
# If Python 3.11 isn't your default Python version:
poetry env use path/to/your/python3.11

# Next, run these commands:
poetry shell
poetry install
```

### R Dependencies

This pipeline is compatible with R 4.1.2. Other versions may work but haven't been tested. Normally, R dependencies will be automatically installed when you run the pipeline. However, if you do not have an internet connection on the compute node, you may want to install the dependencies manually. To do so, run:

```bash
Rscript r-dependencies.R
```

Alternatively, you can set up the dependencies with Snakemake:

```bash
# To only set up:
snakemake -s snakemake_modules/setup.snakefile -c1
```

### Optional: PostgreSQL Database

To set up a PostgreSQL database for Optuna, install the necessary packages in a Conda environment (psycopg2, postgresql, ncurses). Then initialize the database with the following commands:

```bash
pg_ctl init -D db_data
pg_ctl -D db_data -l logfile start

# Create users
createuser --superuser username

# Create databases
createdb -O username mlflow
createdb -O username optuna
```

You can then add the uri to the `config.yml`:  

`postgresql://username@localhost:5432/optuna`  

and start the mlflow server, if used, with the following command:

```bash
mlflow server --backend-store-uri postgresql://username@localhost:5432/mlflow --default-artifact-root file:/path/to/mlflow/artifacts
```

Besides PostgreSQL, you can also use any other database supported by Optuna.

## Easier Handling with Docker

If you prefer to use Docker for setting up the environment, you can use the provided `docker-compose.yml` for this process. Follow these steps:

### Prerequisites

Ensure you have Docker and Docker Compose installed on your machine. You can install them by following the instructions on the [Docker](https://docs.docker.com/get-docker/) website. If you do not want or cannot use docker, you can also use [Podman](https://podman.io/) as an alternative. Replace `docker` with `podman` in the commands below to use Podman.

### Steps

#### 1) Build and Run the Docker Containers

Navigate to the directory containing your `docker-compose.yml` file and run the following command to build and start the Docker containers:

```bash
docker compose up --build
```

This command will:

- Build the Docker images specified in the `Dockerfile`.
- Start the PostgreSQL container and initialize the `optuna` database automatically.
- Start the MLflow server and the corresponding PostgreSQL container.
- Start the Snakemake container with all dependencies installed.

You do not need to manually start PostgreSQL or MLflow. Docker Compose will handle that for you based on the service definitions. You only have to ensure that the vambn_config.yml file is correctly configured.

#### 2) Running Snakemake Commands

Once the containers are up and running, you can access the Snakemake container to execute your pipeline. Open a new terminal window and run the following command:

```bash
docker compose exec snakemake /bin/bash
```

This will open a bash shell inside the Snakemake container. From there, you can run Snakemake commands as needed, for example:

```bash
snake -s Snakefile # run the entire pipeline
# or
snake -s snakemake_modules/traditional-postprocessing.snakefile # run a specific module
```

**Note:** The `snake` command is an alias for `poetry run snakemake --use-conda` that is defined in the `Dockerfile`. You can use `poetry run snakemake --use-conda` directly if you prefer.

#### 3) Accessing PostgreSQL and MLflow

The PostgreSQL and MLflow services are accessible from within the Snakemake container using the following URIs:

- PostgreSQL: `postgresql://vambn:app@postgres:5432/optuna`
- MLflow: `http://mlflow:5000`

You can configure these URIs in your configuration files as needed.

#### 4) Shutting Down the Containers

Once you are done with your work, you can stop and remove the containers by running:

```bash
docker-compose down
```

If you plan on using GPUs with this setup, be aware that you might encounter issues. In our testing scenarios, using GPUs was not found to be beneficial. Therefore, it is recommended to use the CPU for running this pipeline unless you have specific requirements that necessitate GPU usage and are prepared to handle potential issues.

This Docker setup simplifies the installation and configuration process, ensuring all dependencies are correctly installed and services are running in isolated environments.
