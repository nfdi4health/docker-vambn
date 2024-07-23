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


module modular:
    snakefile:
        "modular-modelling.snakefile"
    config:
        config


module traditional:
    snakefile:
        "traditional-modelling.snakefile"
    config:
        config


use rule * from modular as modular_*


use rule * from traditional as traditional_*


################################################################################
# Modular Modelling
################################################################################


# Rule to generate all possible outputs
rule all:
    input:
        rules.modular_all.input,
        rules.traditional_all.input,
    default_target: True
