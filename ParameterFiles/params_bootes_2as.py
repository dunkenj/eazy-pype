"""
Main inputs:
(Change for all fields)

"""
eazypath = '/data2/ken/photoz/eazy-photoz/src/eazy'
working_folder = '/data2/ken/photoz/bootes_2as'
photometry_catalog = 'Bootes_merged_Icorr_2014a_all_ap2_mags.zs.fits'
photometry_format = 'fits'

filter_file = 'filter.bootes_mbrown_2014a.res'
translate_file = 'brown.zphot.2014.translate'

zspec_col = 'z_spec'

flux_col = 'flux'
fluxerr_col ='fluxerr'

do_zp = True
do_zp_tests = False
do_subcats = False

do_full = False
do_stellar = False
do_hb = False
do_merge = False

"""
Training parameters

"""
Ncrossval = 1
test_fraction = 0.2

process_outliers = True
correct_extinction = True

"""
Fitting Parameters
(Change only when needed)

"""

# Templates: Any combination of 'eazy', 'swire', 'atlas'
templates = ['eazy', 'atlas', 'cosmos']#, 'swire']#, 'cosmos', 'atlas'] #,'cosmos', 'atlas']  
fitting_mode = ['a', '1', '1']

defaults = ['defaults/zphot.eazy',
            'defaults/zphot.atlas',
            'defaults/zphot.cosmos']
            #'defaults/zphot.eazy',
            #'defaults/zphot.atlas',
            #'defaults/zphot.swire']

stellar_params = 'defaults/zphot.pickles'

additional_errors = [0.0, 0.0, 0.0]
template_error_norm = [1., 1., 1.]
template_error_file = ''
lambda_fit_max = [5., 30., 30.]



"""
Combination Parameters

"""

fbad_prior = 'mag' # 'flat', 'vol' or 'mag'
prior_parameter_path = 'bootes_I_prior_coeff.npz'
prior_fname = 'I'
prior_colname = 'I_mag'
alpha_colname = 'I_mag'


"""
System Parameters
(Specific system only - fixed after installation)

"""

block_size = 1e4
ncpus = 10


