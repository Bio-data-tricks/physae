"""
Physical constants (CGS units) and molecular parameters.
"""
import math

# Physical constants (CGS)
C = 2.99792458e10  # Speed of light (cm/s)
NA = 6.02214129e23  # Avogadro's number
KB = 1.380649e-16  # Boltzmann constant (erg/K)
R = NA * KB  # Gas constant (erg/(mol.K))
P0 = 1013.25  # Standard pressure (mbar)
T0 = 273.15  # Standard temperature (K)
TREF = 296.0  # HITRAN reference temperature (K)
L0 = 2.6867773e19  # Loschmidt constant (cm^-3)
C2 = 1.438776877  # Second radiation constant (cm.K)

# Mathematical constants for spectral calculations
SQRT_LN2 = math.sqrt(math.log(2.0))
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

# Molecular parameters (g/mol for molecular weight M, partial line width PL in m)
MOLECULE_PARAMS = {
    'CH4': {'M': 16.04, 'PL': 15.12},
    'H2O': {'M': 18.01528, 'PL': 15.12},
}
