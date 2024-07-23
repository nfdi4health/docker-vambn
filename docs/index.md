# Variational Autoencoders Modular Bayesian Networks (VAMBN)

Welcome to the documentation for Variational Autoencoders Modular Bayesian Networks (VAMBN) 2.0. This version features a PyTorch-based HI-VAE (refer to [Nazabal et al.'s paper](https://arxiv.org/abs/1807.03653)) and employs [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html) to manage the workflow of Python and R scripts. Have a look at the [VAMBN page](vambn.md) for an overview of the project.

---

## Getting Started

1. Follow the [installation instructions](setup.md).
2. Go through the example in the [walkthrough section](walkthrough.md).
3. Copy your input data into the `data/raw` folder according to the description in the [walkthrough section](walkthrough.md).
3. Configure the `vambn_config.yml` file according to your needs. Refer to the [configuration section](configuration.md) for details.
4. Execute your pipeline locally or on a cluster.
5. Analyze your results. Refer to the example for explanations.

---

## Development & Bug Tracking

This project is under active development. Expect changes and potential bugs. Please open an issue for any problems you encounter.

## License

This software is licensed under the GNU General Public License (GPL) v3 for non-commercial use. For commercial use, please contact [Holger Fröhlich](mailto:holger.froehlich@scai.fraunhofer.de) to obtain a commercial license.
