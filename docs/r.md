# R Reference

This project uses R for specific analyses alongside Python. This document provides an overview of the R functionality utilized in this project. All R scripts are located in the `vambn-r` directory and are invoked through Snakemake rules defined in the snakefiles.  

The key scripts and their roles are as follows: The Bayesian Network Generation script takes standalone data and module-specific encodings as input to generate a Bayesian Network. The Synthetic Data Generation script creates synthetic patient data based on the Bayesian Network. The Helper Functions script contains utility functions used across the other scripts. The Permutation Test script performs permutation tests on the generated Bayesian Network. The Visualization scripts include one for visualizing the Bayesian Network graph and another for visualizing the correlation matrices for real, decoded, and synthetic data.

The choice of R, specifically the bnlearn library, was driven by its advanced capabilities in handling Bayesian Networks, surpassing current Python libraries. Should an equivalent Python library become available in the future, the R scripts can be transitioned to Python.
