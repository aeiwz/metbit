import io
from os.path import abspath, dirname, join
from setuptools import find_packages, setup


HERE = dirname(abspath(__file__))
LOAD_TEXT = lambda name: io.open(join(HERE, name), encoding='UTF-8').read()
DESCRIPTION = '\n\n'.join(LOAD_TEXT(_) for _ in [
    'README.rst'
])

setup(
  name = 'metbit',      
  packages = ['metbit'], 
  version = '0.0.1',  
  license='MIT', 
  description = 'Metabolomics data analysis and visualization tools.',
  long_description=DESCRIPTION,
  author = 'aeiwz',                 
  author_email = 'theerayut_aeiw_123@hotmail.com',     
  url = 'github link',  
  download_url = 'download link',  
  keywords = ['Omics', 'Chemometrics', 'Visualization', 'Data Analysis'],
  install_requires=[            
          'scikit-learn',
          'pandas',
          'numpy',
          'matplotlib',
          'seaborn',
          'scipy',
          'statsmodels',
          'plotly',
          'dash',
          'pyChemometrics'],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Education',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',        
    'Programming Language :: Python :: 3.11',
  ],
)