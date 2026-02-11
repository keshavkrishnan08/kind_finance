"""Core model components: VAMPnet, Koopman analysis, thermodynamic quantities."""
from .vampnet import NonEquilibriumVAMPNet, VAMPNetLobe
from .koopman import KoopmanAnalyzer
from .entropy import EntropyDecomposer, estimate_empirical_entropy_production
from .irreversibility import IrreversibilityAnalyzer
from .losses import total_loss, vamp2_loss
