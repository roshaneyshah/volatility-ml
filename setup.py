from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='forecasting-models',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Time series forecasting library for volatility and price prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/forecasting-models',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'scikit-learn>=0.24.0',
        'xgboost>=1.0.0',
        'lightgbm>=3.0.0',
        'arch>=4.0.0',
        'statsmodels>=0.12.0',
        'prophet>=1.0.0',
        'tensorflow>=2.5.0',
    ],
)
