from glob import glob
import re
from typing import Set
import random
from itertools import combinations
from snakemake.utils import min_version


min_version("6.0")
random.seed(42)


configfile: "vambn_config.yml"


################################################################################
# Module dependencies
################################################################################


ruleorder: modular_preprocessing_setupEnv_setupR > traditional_preprocessing_setupEnv_setupR > modular_preprocessing_Preprocessing > traditional_preprocessing_Preprocessing > modular_preprocessing_ConcatImputedFiles > traditional_preprocessing_ConcatImputedFiles > modular_preprocessing_ConcatStaloneFiles > traditional_preprocessing_ConcatStaloneFiles


if config["snakemake"]["modules"]["use_modular"]:

    module modular:
        snakefile:
            "modular-modelling.snakefile"
        config:
            config

    use rule * from modular as modular_*


if config["snakemake"]["modules"]["use_traditional"]:

    module traditional:
        snakefile:
            "traditional-modelling.snakefile"
        config:
            config

    use rule * from traditional as traditional_*


################################################################################
# Modular Modelling
################################################################################


# Rule to generate all possible outputs
# Collect targets from active modules
all_inputs = []
if config["snakemake"]["modules"]["use_modular"]:
    all_inputs.extend(expand("modular_out/{sample}.processed", sample=config["samples"]))
if config["snakemake"]["modules"]["use_traditional"]:
    all_inputs.extend(expand("traditional_out/{sample}.analyzed", sample=config["samples"]))


rule all:
    input:
        all_inputs,
    default_target: True  # Explicitly mark as default
