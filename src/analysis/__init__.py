"""Analysis modules: spectral, regime, rolling, statistical tests, non-equilibrium."""
from .spectral import SpectralAnalyzer
from .regime import RegimeDetector
from .rolling import RollingSpectralAnalyzer
from .chapman_kolmogorov import chapman_kolmogorov_test
from .statistics import StatisticalTests
from .nonequilibrium import (
    detailed_balance_violation,
    gallavotti_cohen_symmetry,
    fluctuation_theorem_ratio,
    onsager_regression_test,
    eigenvalue_complex_plane_statistics,
)
