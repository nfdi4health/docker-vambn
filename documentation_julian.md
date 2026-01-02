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


- Some error occured, first uv add dill
- Than not syndat.quality, get_auc 
  - --> changed trainer.py --> from syndat.metrics import discriminator_auc and changed code 
  - Happend because i use the up-to-date version of syndat not 0.0.2 or so

The run finished!!!

- Now i want to create the visualitsations with
  - ```snakemake -s ./snakemake_modules/traditional-postprocessing.snakefile --use-conda --conda-frontend mamba -c8```
- error in GenreateUmap and GenreateOptunaPlots
  - GenerateUmap, maybe because no tool umap --> installed umap-learn 
  - GenerateOptunaPlots -> kalaido package missing --> installed it

run finished


## Run with filtered Data
- Filter data based on split_Texas_Data-Exploration-Cleaning-Preprocessing.ipynb
- move input_texas to old folder
- try out ```snakemake -s ./snakemake_modules/traditional-modelling.snakefile --use-conda --conda-frontend mamba -c8```
- Error:
```bash
Traceback (most recent call last):

  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/src/app/vambn/data/make_data.py", line 522, in <module>
    app()

  File "/usr/src/app/vambn/data/make_data.py", line 187, in make
    processed_data = prepare_data(
                     ^^^^^^^^^^^^^

  File "/usr/src/app/vambn/data/helpers.py", line 433, in prepare_data
    raise Exception("No columns left after filtering.")

Exception: No columns left after filtering.
```

### Next try
- set config_texas.json, variance threshold to 0.
- needs very long. always "freezes" at Fitting the final BN using  rsmax2  with  500  bootstrap samples.
- manually changing the code to set method to "tabu" and only needs 2 min (instead of more than 18h!!!)
- looking at the real / synthetic data --> First look, very similar. But different BNs and slightly different outcomes. 
  
> Achtung: ```totalChargesAccomm``` in grouping und input_texas unterschiedlich !!!


Todos:
- Understand Code and how it filters input data!
- Do filtering or inspect intermediate steps
- Probably better to "exclude" filtering so that input and synthetic data can be compared better!
- Container of version 2.0.1 test -> does it work with "wrong" R.yml file? Why is in local repo the wrong R.yml file?

1. Understanding what Manuel changed compared to much easier understandable version 1.0.0 and Paper-VAMBN
   - tensorflow -> pytorch
   - what are all the different layers/modules? 
     - --> includes multiple HIVAE implementations with pytorch (lst-hivae, GAN-hivae)
     - What arte HEADS?
   - Why are there so many new .py files? What are they doing?
   - Where are the essential config files? 
   - Why is the code so much less documented ??? In version 1.0.0 the structure of what will be executed etc is clearer...

2. Use version 1.0.0 to execute texas file "by hand"
   - hopefully this will enhance my understanding of the pipeline and why manuel implemented it like this
  
3. Maybe I can "reduce" manuel's code to version 1.0.0 (?) and only "necessary" changes of vambn will be used to later to include DAGMA.
   - VAMBN would be closer to implementation in paper
   - VAMBN-DAGMA could be easier compared to VAMBN!
   - Maybe I can get the same data as in Paper to directly compare results?
  