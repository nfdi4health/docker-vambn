# Python Reference

This project primarily uses Python for its main functionality, which is implemented as a Python package called `vambn`. This document provides an overview of the functionality that Python provides in this project. For detailed information, please refer to the API Reference page of this website documentation (Python Reference > API Reference).

The `vambn` package is structured into five main modules: data, metrics, modelling, utils, and visualization. The data module handles data loading, preprocessing, and transformation. The metrics module includes various evaluation metrics used to assess the performance of models and analyses. The modelling module primarily implements the Heterogeneous and Incomplete Variational Autoencoder (HIVAE) and the necessary code for it, including the trainer, MTL (multi-task learning), and GAN variants. The utils module provides utility functions that support various tasks across the package. Lastly, the visualization module contains functions and tools to create visual representations of the results, aiding in the interpretation and communication of the findings.

Like the R scripts, the Python functionality is handled from Snakemake itself.
