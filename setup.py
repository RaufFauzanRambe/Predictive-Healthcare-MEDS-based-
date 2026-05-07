"""
setup.py - Package installation for predictive-healthcare-meds.
"""

from setuptools import setup, find_packages

setup(
    name="predictive-healthcare-meds",
    version="1.0.0",
    description="Predictive healthcare analytics using MEDS-formatted data",
    author="Healthcare ML Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.1.0",
        "pyyaml>=6.0",
        "mlflow>=2.14.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "healthcare-train=scripts.train_model:main",
            "healthcare-predict=scripts.run_inference:main",
        ],
    },
)
