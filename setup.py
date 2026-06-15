from setuptools import setup, find_packages

setup(
    name='desta',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "whisper_normalizer",
        "huggingface_hub",
        "lulutils @ git+https://github.com/kehanlu/lulutils.git",
        "transformers==4.49.0",
    ],
    entry_points={
        'console_scripts': [
            'desta-pull-model=desta.cli.pull_model:main',
            'desta-pull-audios=desta.cli.pull_audios:main',
        ],
    },
    description='A brief description of your project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)