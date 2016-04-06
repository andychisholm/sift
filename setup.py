from setuptools import setup, find_packages

__version__ = '0.2.0'
__pkg_name__ = 'textsift'

setup(
    name = __pkg_name__,
    version = __version__,
    description = 'Text modelling framework',
    author='Andrew Chisholm',
    packages = find_packages(),
    license = 'MIT',
    url = 'https://github.com/wikilinks/sift',
    scripts = [
        'scripts/sift',
        'scripts/download-wikipedia'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic'
    ],
    install_requires = [
        "ujson",
        "numpy",
        "pattern",
        "gensim",
        "msgpack-python"
    ],
    test_suite = __pkg_name__ + '.test'
)
