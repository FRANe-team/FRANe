from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='frane',
    version='0.1.0',
    description='Unsupervised Feature Ranking via Attribute Networks',
    url='https://github.com/FRANe-team/FRANe',
    author='Urh Primožič, Blaž Šrklj, Sašo Džeroski, Matej Petkovič',
    author_email='urh.primozic@student.fmf.uni-lj.si',
    license='BSD 2-clause',
    packages=['frane'],
    install_requires=[
        'numpy>=1.19.5',
        'scipy>=1.6.0',
    ],
    project_urls={
        "Bug Tracker": "https://github.com/FRANe-team/FRANe/issues",
        "Article": "not-yet-published",
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
