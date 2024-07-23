# VAMBN 2.0

## Overview

**VAMBN (Variational Autoencoder Modular Bayesian Network)** is a machine learning approach designed to simulate heterogeneous data from clinical studies and other research fields. It addresses data sharing restrictions due to legal and ethical constraints by combining Bayesian Networks and Variational Autoencoders. This combination allows VAMBN to handle complexities such as longitudinal data, limited sample sizes, diverse variables, and missing values. By creating realistic synthetic data, VAMBN facilitates data sharing and the simulation of counterfactual scenarios, which can be beneficial for various research purposes.

## New Features in VAMBN 2.0

VAMBN 2.0 introduces several enhancements and changes over the [original version](https://github.com/elg34/VAMBN):

1. **Implementation in Pytorch**: The HI-VAE component has been implemented in Pytorch, offering better performance and integration capabilities.
2. **Two Variants**:
    - **Traditional Variant**: Corresponds to the original implementation, using separate HI-VAEs for each module.
    - **Modular Variant**: Features a single modular HI-VAE that processes all modules with a shared structure. Despite the initial hypothesis that the modular approach would better preserve the correlation structure, results showed no real benefits over the traditional variant.
3. **Addition of LSTM HI-VAEs**: Long Short-Term Memory (LSTM) HI-VAEs have been introduced to handle longitudinal data more effectively, improving the model's ability to manage time-series data.
4. **Multi-task Learning Objectives**: Introduced to improve the performance and adaptability of the model.
5. **GAN Variant**: A new variant using Generative Adversarial Networks (GANs) to enhance the generation of synthetic data.
6. **Snakemake Implementation**: The Snakemake workflow management system has been integrated, making the application easy to use and scalable.
7. **Dockerized Version**: A Dockerized version of VAMBN 2.0 is available, which simplifies the handling of dependencies and ensures a consistent environment across different setups.

## Rationale of the Modular Approach

The development of VAMBN 2.0 was driven by the need to better preserve the correlation structure within the data. The modular approach was hypothesized to achieve this; however, performance tests indicated that while the modular approach was on par with the traditional method, it did not offer additional benefits. Thus, the traditional variant is recommended for its computational efficiency. Additionally, the introduction of LSTM HI-VAEs was aimed at enhancing the handling of longitudinal data.

## Use Cases

VAMBN 2.0 can be applied in various scenarios, including:

- **Generation of Synthetic Data**: Creating realistic virtual patients for data sharing and analysis.
- **Understanding Relationships Between Modules**: Analyzing the relationship of different modules with standalone data. For example, standalone data can be passed to the Bayesian Network (BN) without the HI-VAE, allowing researchers to understand the relationship between digital features (e.g., data from an app) and clinical scores.

## Further Information

For detailed setup instructions, including dependencies and installation, please refer to the [Setup Documentation](setup.md) and the [Walkthrough](walkthrough.md) section for a step-by-step guide on running VAMBN 2.0. The [Configuration](configuration.md) section provides information on how to configure the `vambn_config.yml` file according to your needs.
