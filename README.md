# eazy-pype
Python pipeline to automate the calculation of photo-z's with [EAzY](http://github.com/gbrammer/eazy-photoz/) for large fields.

Steps include:
 * Zeropoint offsets from spectroscopic training set, including cross validation of multiple spec-z subsets.
 * Parallelisation of fitting into multiple sub-sets for speed improvements
 * Calibration of P(z) errors using spectroscopic training sets (again including cross validation)
 * Hierarchical Bayesian combination of estimates using multiple template sets



## Requirements
 * Working [EAzY](http://github.com/gbrammer/eazy-photoz/) installation
 * Python 2.7:
     * numpy==1.13.0
     * scipy==0.19.1
     * matplotlib==2.0.2
     * h5py==2.8.0
     * astropy==1.3.3
     * emcee==2.2.1
     * scikit_learn==0.19.1
     * smpy==1.0.3
