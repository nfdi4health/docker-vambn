# import library
library("reticulate")
library(config)


# get argument
args <- commandArgs(trailingOnly = TRUE)
selection <- tryCatch({
                    as.integer(args[1])
                },
                error = function(cond) {
                    message("[**] Please pass in a valid argument (0/1/2/3/4/5/6).")
                    message(cond)
                    return(NA)
                })

print(paste0("[**] Argument received: ", selection))


# execute according to the received argument
if (selection == 0) {
    print("[**] Running all functions...")

    # get hyperopt method from config
    config <- config::get(file = "/vambn/02_config/config_r.yml")
    hyperopt_method <- config$hyperopt_method

    # hyperparameter optimization
    if (hyperopt_method == "BayesianOptimization"){
        print("[**] Executing Bayesian Optimization script...")
        source_python("./vambn/03_code/hivae_bayes-opt.py")
        bayes_opt <- c_HIVAE_BayesianOptimization()
        bayes_opt$hyperopt_HIVAE()
        print("[**] Bayesian Optimization script completed.")
    } else if (hyperopt_method == "GridSearch") {
        print("[**] Executing Grid Search script...")
        source_python("./vambn/03_code/hivae_grid-search.py")
        grid_search <- c_HIVAE_GridSearch()
        grid_search$hyperopt_HIVAE()
        print("[**] Grid Search script completed.")
    } else {
        print("[**] Skip hyperparameter optimization.")
    }

    # HIVAE training
    print("[**] Executing HIVAE Training script...")
    source_python("./vambn/03_code/hivae_modelling.py")
    hivae_modelling <- c_HIVAE_Modelling()
    hivae_modelling$train_HIVAE()
    print("[**] HIVAE Training script completed.")

    # bnet training
    print("[**] Executing Bnet script...")
    source("./vambn/03_code/bnet.R")
    run_bnet()
    print("[**] Bnet script completed.")

    # simulate virtual patients
    print("[**] Executing Virtual Patient Simulation script...")
    source("./vambn/03_code/bnet.R")
    simulate_virtual_patient()
    print("[**] Virtual Patient Simulation script completed.")

    # decode virtual patients
    print("[**] Executing HIVAE Decoding script...")
    source_python("./vambn/03_code/hivae_modelling.py")
    hivae_modelling <- c_HIVAE_Modelling()
    hivae_modelling$decode_HIVAE()
    print("[**] HIVAE Decoding script completed.")

    # generate plots
    print("[**] Executing Plotting script...")
    source("./vambn/03_code/plots_generation.R")
    create_all_plots()
    print("[**] Plotting script completed.")

} else if (selection == 1) {
    # get hyperopt method from config
    config <- config::get(file = "/vambn/02_config/config_r.yml")
    hyperopt_method <- config$hyperopt_method

    # hyperparameter optimization
    if (hyperopt_method == "BayesianOptimization"){
        print("[**] Executing Bayesian Optimization script...")
        source_python("./vambn/03_code/hivae_bayes-opt.py")
        bayes_opt <- c_HIVAE_BayesianOptimization()
        bayes_opt$hyperopt_HIVAE()
        print("[**] Bayesian Optimization script completed.")
    } else if (hyperopt_method == "GridSearch") {
        print("[**] Executing Grid Search script...")
        source_python("./vambn/03_code/hivae_grid-search.py")
        grid_search <- c_HIVAE_GridSearch()
        grid_search$hyperopt_HIVAE()
        print("[**] Grid Search script completed.")
    } else {
        print("[**] Skip hyperparameter optimization.")
    }

} else if (selection == 2) {
    print("[**] Running HIVAE Modelling...")

    # HIVAE training
    print("[**] Executing HIVAE Training script...")
    source_python("./vambn/03_code/hivae_modelling.py")
    hivae_modelling <- c_HIVAE_Modelling()
    hivae_modelling$train_HIVAE()
    print("[**] HIVAE Training script completed.")

} else if (selection == 3) {
    print("[**] Running BNet...")

    # bnet training
    print("[**] Executing Bnet script...")
    source("./vambn/03_code/bnet.R")
    run_bnet()
    print("[**] Bnet script completed.")

} else if (selection == 4) {
    print("[**] Running Virtual Patient Simulation...")

    # simulate virtual patients
    print("[**] Executing Virtual Patient Simulation script...")
    source("./vambn/03_code/bnet.R")
    simulate_virtual_patient()
    print("[**] Virtual Patient Simulation script completed.")

} else if (selection == 5) {
    print("[**] Running HIVAE Decoding...")

    # decode virtual patients
    print("[**] Executing HIVAE Decoding script...")
    source_python("./vambn/03_code/hivae_modelling.py")
    hivae_modelling <- c_HIVAE_Modelling()
    hivae_modelling$decode_HIVAE()
    print("[**] HIVAE Decoding script completed.")

} else if (selection == 6) {
    print("[**] Running Quality Assessment Plotting...")

    # generate plots
    print("[**] Executing Plotting script...")
    source("./vambn/03_code/plots_generation.R")
    create_all_plots()
    print("[**] Plotting script completed.")

} else {
   print("[*] Please pass in a valid argument (0/1/2/3/4/5/6).")
}