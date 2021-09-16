
from distutils.core import setup
setup(
  name = 'cleu',         # How you named your package folder (MyLib)
  packages = ['cleu','cleu.plot','cleu.utils'],   # Chose the same as "name"
  version = '0.6',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'CLEU is a a high-level easy to use python library that can be used to analyze cross-lingual word embeddings on multiple languages, In addition CLEU provide some interactive visualization, where users can explore the cross-lingual word embeddings on two or more languages.',   # Give a short description about your library
  author = 'Lungavile',                   # Type in your name
  author_email = 'lungavile@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/lungavile/cleu',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/lungavile/cleu/archive/refs/tags/v0.5.tar.gz',    # I explain this later on
  keywords = ['nlp', 'word embeddings', 'cross-lingual word embeddings', 'multilingual word embeddings','visualization'],   # Keywords that define your package best
  install_requires=[
        'altair',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'seaborn',
        'umap-learn'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
)