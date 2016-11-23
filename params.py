"""
Main inputs:
(Change for all fields)

"""
working_folder = '/data2/ken/photoz/ezpipe-test'
photometry_catalog = 'Bootes_merged_Icorr_2014a_all_ap4_mags.testsample.cat'
photometry_format = 'ascii.commented_header'

filter_file = 'filter.bootes_mbrown_2014a.res'
translate_file = 'brown.zphot.2014.translate'

zspec_col = 'z_spec'


do_zp = False
do_full = False
do_stellar = True

"""
Training parameters

"""
Ncrossval = 1
test_fraction = 0.2

process_outliers = True


"""
Fitting Parameters
(Change only when needed)

"""

# Templates: Any combination of 'eazy', 'swire', 'atlas'
templates = ['eazy', 'cosmos', 'atlas']#, 'cosmos', 'atlas'] #,'cosmos', 'atlas'] 
fitting_mode = ['a', '1', '1']
defaults = ['defaults/zphot.eazy',
            'defaults/zphot.cosmos',
            'defaults/zphot.atlas']

stellar_params = 'defaults/zphot.pickles'

additional_errors = [0.05, 0.05, 0.05]
template_error_norm = [0.5, 0.5, 0.5]
template_error_file = ''
lambda_fit_max = [5., 10., 30.]



"""
Combination Parameters

"""

fbad_prior = 'mag' # 'flat', 'vol' or 'mag'
prior_parameter_path = 'bootes_I_prior_coeff.npz'
prior_colname = 'I_mag'


"""
System Parameters
(Specific system only - fixed after installation)

"""

block_size = 1e4
ncpus = 4


