"""
Global parameters and normalization configurations.
"""
from typing import Dict

# Parameter list for normalization
PARAMS = ['sig0', 'dsig', 'mf_CH4', 'mf_H2O', 'baseline0', 'baseline1', 'baseline2', 'P', 'T']

# Mapping from parameter name to index
PARAM_TO_IDX = {n: i for i, n in enumerate(PARAMS)}

# Parameters that use log-scale normalization
LOG_SCALE_PARAMS = {'mf_CH4', 'mf_H2O'}

# Minimum value for log scaling
LOG_FLOOR = 1e-7

# Normalization ranges (min, max) per parameter
# Defaults mirror the expanded training ranges from ``physae.py`` so the
# package remains usable even before YAML configuration files are loaded.
NORM_PARAMS: Dict[str, tuple] = {
    "sig0": (3085.37, 3085.52),
    "dsig": (0.001502, 0.001559),
    "mf_CH4": (1.0e-7, 2.9e-5),
    "mf_H2O": (1.0e-7, 4.25e-4),
    "baseline0": (0.999999, 1.00001),
    "baseline1": (-5.0e-4, -2.0e-4),
    "baseline2": (-7.505155e-8, 3.77485e-9),
    "P": (400.0, 600.0),
    "T": (302.65, 312.65),
}
