from setuptools import setup, find_packages

setup(
    name="ktnd_finance",
    version="1.0.0",
    description="Non-Equilibrium Koopman-Thermodynamic Neural Decomposition of Financial Market Dynamics",
    author="Keshav Krishnan",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.28",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "hmmlearn>=0.3.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "arch>=6.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
)
