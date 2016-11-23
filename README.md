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
    * Astropy v1.+
    * scikit-learn v?
    

