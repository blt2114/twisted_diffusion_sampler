from setuptools import setup

setup(
    name="se3_tds",
    packages=[
        'data',
        'analysis',
        'model',
        'experiments',
        'openfold',
        'motif_scaffolding',
        'smc_utils_prot',
    ],
    package_dir={
        'data': './data',
        'analysis': './analysis',
        'model': './model',
        'experiments': './experiments',
        'openfold': './openfold',
        'smc_utils_prot': './smc_utils_prot',
    },
)
