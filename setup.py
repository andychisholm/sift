from setuptools import setup, find_packages

__version__ = '0.1.0'
__pkg_name__ = 'sift'

setup(
    name = __pkg_name__,
    version = __version__,
    description = 'Text modelling framework',
    author='Andrew Chisholm',
    packages = find_packages(),
    license = 'MIT',
    url = 'https://github.com/wikilinks/sift',
    scripts = {
        'scripts/sift'
    },
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
        "pattern"
    ],
    test_suite = __pkg_name__ + '.test'
)
