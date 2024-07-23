# Troubleshooting

## R-Related Issues

- **Error about 'module load' when R scripts are being executed:** Modify the configuration at `snakemake > cluster_modules > R` and set this to `null`.
- **Errors when running R scripts on a SLURM cluster:** Ensure the configuration at `snakemake > cluster_modules > R` is set correctly. For example, set it to `R/4.1.2-foss-2021b`.
- **Unable to install R dependencies:** Ensure your cluster/computer has internet access. If necessary, you can install the dependencies manually. Refer to the [setup instructions](setup.md) for details on how to install R dependencies manually.

## General Issues

- **There is no vambn_config.yml file in the repository**: Copy the `default_vambn_config.yml` file and rename it to `vambn_config.yml`. Modify the configuration as needed.
