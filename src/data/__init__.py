"""Data pipeline: download, preprocessing, loading."""
from .loader import TimeLaggedDataset, RollingWindowDataset
from .preprocessing import compute_log_returns, standardize_returns, time_delay_embedding
