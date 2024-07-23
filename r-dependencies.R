# Install remotes package if needed
if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = 'http://cran.rstudio.com/')
}

# List of packages and versions
packages_to_install <- c(
    "igraph",
    "bnlearn",
    "ggraph",
    "tidygraph",
    "tidyverse",
    "arules",
    "mclust",
    "rpart",
    "stringr",
    "purrr",
    "readxl",
    "tidyr",
    "dplyr",
    "optparse",
    "svglite"
)

versions <- c(
    "1.5.1",
    "4.8.3",
    "2.1.0",
    "1.2.3",
    "2.0.0",
    "1.7.6",
    "6.0.0",
    "4.1.19",
    "1.5.0",
    "1.0.2",
    "1.4.3",
    "1.3.0",
    "1.1.3",
    "1.7.3",
    "2.1.1"
)

# Function to install package if not already installed or incorrect version
install_if_needed <- function(package, version) {
    if (!requireNamespace(package, quietly = TRUE) || 
        (packageVersion(package) != version)) {
        remotes::install_version(package, version, repos = 'http://cran.rstudio.com/')
    }
}

# Install packages
for (i in 1:length(packages_to_install)) {
    install_if_needed(packages_to_install[i], versions[i])
}


# Load the libraries
lapply(packages_to_install, require, character.only = TRUE)
