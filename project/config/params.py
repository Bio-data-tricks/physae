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
# This will be populated at runtime
NORM_PARAMS: Dict[str, tuple] = {}
