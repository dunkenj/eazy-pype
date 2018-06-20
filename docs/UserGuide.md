# eazy-pype - User Guide (In Progress)

Eazy-pype requires a working installation of the [EAzY](http://github.com/gbrammer/eazy-photoz/) template fitting photometric redshift code in addition to the Python package requirements outlined in [requirements.txt](). The code is based around a main routine, ```eazy-pype.py```, which then calls functions from the other python modules or directly calls them.
After setting the relevant parameters in the `params.py` file (or similar), the code is run following:

```
python eazy-pype.py -p params.py
```

where ```params.py```(or similar name) is the specific parameter file for a given field (see below). In principle, depending on the options selected the above command can consecutively run through all stages of the procedure. However, in practice this is typically run in stages with checks at critical stages. In the sections below we outline the typical procedure taken.

## The Parameter file


```
eazypath = '... path .../eazy-photoz/src/eazy'
```

```
working_folder = 'HELP/EGS'
photometry_catalog = 'master_catalogue_egs_20180501_processed.fits.mod'
photometry_format = 'fits'

filter_file = 'egs_filters.res'
translate_file = 'egs.translate'
```

```
zspec_col = 'z_spec'
mag_col = '_mag'
magerr_col = '_magerr'
flux_col = 'flux'
fluxerr_col = 'fluxerr'
```

```
do_zp = False
do_zp_tests = False

do_subcats = False
do_full = False
do_stellar = False
do_hb = True
do_merge = True
```

- ```do_zp```:
- ```do_zp_tests```:
- ```do_subcats```:
- ```do_full```:
- ```do_stellar```:
- ```do_hb```:
- ```do_merge```:  


### Training parameters
```
Ncrossval = 1
test_fraction = 0.2

process_outliers = True
correct_extinction = True
```

### Fitting Parameters
```
# Templates: Any combination of 'eazy', 'cosmos', 'atlas'
templates = ['eazy', 'atlas', 'cosmos']
fitting_mode = ['a', '1', '1']
defaults = ['defaults/zphot.eazy',
            'defaults/zphot.atlas',
            'defaults/zphot.cosmos']

stellar_params = 'defaults/zphot.pickles'

additional_errors = [0.0, 0.0, 0.0]
template_error_norm = [1., 1., 1.]
template_error_file = ''
lambda_fit_max = [5., 30., 30.]
```

### Combination Parameters
```

include_prior = True
include_prior_gal = True
include_prior_agn = True
fbad_prior = 'mag' # One of: 'flat', 'vol' or 'mag'
prior_parameter_path = 'bootes_I_prior_coeff.npz'
prior_fname = 'r_any_mag'
prior_colname = 'r_any_mag'
alpha_colname = 'r_any_mag'

gpz = True

ir_gpz_path = None
xray_gpz_path = None
opt_gpz_path = None
gal_gpz_paths = ['/data2/ken/HELP/EGS/gpz/megacam_ugrizweight_save.pkl',
                 '/data2/ken/HELP/EGS/gpz/suprime_grizy_weight_save.pkl']

```


### System Parameters

```
block_size = 1e4
ncpus = 4
```



## Catalog Pre-processing

If either ```process_outliers = True``` or ```correct_extinction = True```, during the initial loading of the input catalog,

## Zeropoint Offset Calculation

```do_zp = True```

## Running Template Fits for the Full Field


## Tuning Bayesian Combination Hyper-parameters

## Constructing consensus estimates

## Final Catalog Merging
