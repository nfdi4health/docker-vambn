# Documentation
Here I will keep track of everything I do or change in the project.
## System
- Ubuntu (wsl)

## Setup
- cloned project
- removed pyproject.toml and poetry.lock file
- initialized an uv init
- during setup i will install all necessary dependencies
- dowloaded texas csv file and used ... to convert it to input
  - ```wget -P data/raw https://github.com/arx-deidentifier/arx/raw/shadow/data/texas.csv```
  -  ```python notebooks/convert_texas.py data/raw/texas.csv data/raw/input_texas.csv```
-  removed snakemake part from docker-compose.yml
   -  add ports: "5432:5432" to postgres because itherwise its not accessible with localhost
-  copy-past-rename vambn-config.yml
   -  change ```"http://mlflow:5000"``` to ```"http://localhost:5000"```
   -  change ```"postgresql://vambn:app@postgres:5432/optuna"``` to ```"postgresql://vambn:app@localhost:5432/optuna"```
   -  change ```/usr/src/app/R.yml``` to ```../R.yml```
-  change the R.yml file: swap conda-forge and default around (otherwise the ackages wont load)uv
   -  conda-forge
   -  default 
-  ```podman compose up -d```
- ```snakemake -s ./snakemake_modules/traditional-modelling.snakefile --use-conda --conda-frontend mamba -c8```
  - only use traditional modelling