# eazy-pype
Python pipeline to automate the calculation of photo-z's with [EAzY](http://github.com/gbrammer/eazy-photoz/) for large fields. Including calculation of zeropoint offsets from a spectroscopic training set, parallelisation of fitting into multiple subsets for speed improvements, calibration of redshift posteriors errors using spectroscopic training sets and Hierarchical Bayesian combination of estimates using multiple template sets.

Optionally, additional machine-learning estimates using the [GPz](https://github.com/OxfordML/GPz/tree/695a83aa3959d1c849046dd2bad25044603f2c78) Gaussian Process redshift code can also be folded into the Hierarchical Bayesian combination procedure to significantly improve the accuracy of the estimates in fields with good training samples available.

The full procedure and the motivations behind the methodology are presented in [Duncan et al. (2018a)](https://ui.adsabs.harvard.edu/link_gateway/2018MNRAS.473.2655D/doi:10.1093/mnras/stx2536) and [Duncan et al. (2018b)](https://ui.adsabs.harvard.edu/link_gateway/2018MNRAS.477.5177D/doi:10.1093/mnras/sty940).

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
     * gpz==1.0 - [GPz](https://github.com/OxfordML/GPz/tree/695a83aa3959d1c849046dd2bad25044603f2c78)


## Usage

After setting the relevant parameters in the `params.py` file (or similar), the code is run following:

```
python eazy-pype.py -p params.py
```

See the [User Guide](https://github.com/dunkenj/eazy-pype/docs/UserGuide.md) for a more detailed description of how the code is run, the stages involved and the typical procedure used when applying the dataset to a large dataset.
