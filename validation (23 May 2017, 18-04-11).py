import numpy as np
import array
import os, sys
import re

import smpy.smpy as S
from scipy.interpolate import CubicSpline

from astropy.table import Table
from astropy import units as u

def f99_extinction(wave):
    """
    Return Fitzpatrick 99 galactic extinction curve as a function of wavelength
    """
    anchors_x = [0., 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846]
    anchors_y = [0., 0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591]
    
    f99 = CubicSpline(anchors_x, anchors_y)
    
    output_x = (1 / wave.to(u.micron)).value    

    return f99(output_x)


def process(catalog_in, translate_file, filter_file,
            cat_format = 'ascii.commented_header',
            id_col = 'id', flux_col = 'flux',
            fluxerr_col = 'fluxerr', exclude_columns=None,
            correct_extinction=True,
            overwrite=True, verbose=False):
    
    input_data = Table.read(catalog_in, format=cat_format)
    if exclude_columns != None:
        input_data.remove_columns(exclude_columns)
    column_names = input_data.columns.keys()

    ID = input_data[id_col]
    
    flux_col_end = flux_col
    fluxerr_col_end = fluxerr_col
    
    try:
        translate_init = Table.read(translate_file, format='ascii.no_header')
        fnames = translate_init['col1']
        fcodes = translate_init['col2']
        flux_cols = np.array([a.startswith('F') for a in fcodes])
        fluxerr_cols = np.array([a.startswith('E') for a in fcodes])

    except:
        raise
    
        # Parse filter file with smpy - get central wavelengths back
    filt_obj = S.LoadEAZYFilters(filter_file)
        
    lambda_cs = np.zeros(flux_cols.sum())
    f99_means = np.zeros(flux_cols.sum())
    
    for il, line in enumerate(translate_init[flux_cols]):
        filtnum = int(line['col2'][1:])-1
        lambda_cs[il] = filt_obj.filters[filtnum].lambda_c.value

        wave = filt_obj.filters[filtnum].wave
        resp = filt_obj.filters[filtnum].response

        f99_ext = f99_extinction(wave)
        f99_means[il] = np.trapz(resp*f99_ext, wave.value) / np.trapz(resp, wave.value)


    wl_order = np.argsort(lambda_cs)

    flux_colnames_ordered = fnames[flux_cols][wl_order]
    fluxerr_colnames_ordered = fnames[fluxerr_cols][wl_order]
    

    f99_means_ordered = f99_means[wl_order]
    
    fluxes = np.zeros((len(input_data), len(flux_colnames_ordered)))    
    fluxerrs = np.zeros((len(input_data), len(flux_colnames_ordered)))
    
    for i in range(len(flux_colnames_ordered)):        
        if correct_extinction:
            try:
                extinctions = f99_means_ordered[i]*input_data['EBV']
                flux_correction = 10**(extinctions/2.5)
                
                isgood = (input_data[fluxerr_colnames_ordered[i]] > 0.)

                input_data[flux_colnames_ordered[i]][isgood] *= flux_correction[isgood]
                input_data[fluxerr_colnames_ordered[i]][isgood] *= flux_correction[isgood]
                input_data[flux_colnames_ordered[i][:-4]+'mag'][isgood] -= extinctions[isgood]
                
                
            except:
                raise
                
        fluxes[:,i] = input_data[flux_colnames_ordered[i]]
        fluxerrs[:,i] = input_data[fluxerr_colnames_ordered[i]]
    
    
    fluxes = fluxes[:,1:-1]
    fluxerrs = fluxerrs[:,1:-1]
    fnames_full = flux_colnames_ordered[1:-1]
    efnames_full = fluxerr_colnames_ordered[1:-1]
    fnames = [filt.split('_')[0] for filt in fnames_full]

    color_a = np.zeros(fluxes.shape)
    color_b = np.zeros(fluxes.shape)

    upper_lims = np.where(np.logical_and(fluxes/fluxerrs < 2., fluxerrs > 0.))
    fluxes[upper_lims] = -99.
    #np.maximum(fluxes[upper_lims], 0) + 2.*fluxerrs[upper_lims]

    color_a[:,1:] = -2.5*np.log10(fluxes[:,:-1]/fluxes[:,1:])
    color_b[:,:-1] = -2.5*np.log10(fluxes[:,:-1]/fluxes[:,1:])


    # Identify anomalous datapoints by extreme red/blue colours
    # Or strong jumps in colour

    drop = np.logical_and(color_a < -1., color_b > 1.)
    spike = np.logical_and(color_a > 2.5, color_b < -2.5)

    criteria1 = np.logical_or(drop, spike)  
    criteria2 = np.logical_or(color_a < -5, color_b > 5)

    anomalous = np.logical_or(criteria1,criteria2) #* (fluxes / fluxerrs > 2.)
    #anomalous_bands = I[anomalous]

    good = (fluxerrs > 0)
    fraction_bad = (good*anomalous).sum(0).astype('float') / good.sum(0)
    gg = zip(fnames, fraction_bad*100)


    for i, name in enumerate(fnames_full):
        if verbose:
            print(name)
        input_data[name][anomalous[:,i]] = -90.
        input_data[efnames_full[i]][anomalous[:,i]] = -90.
    
    outname = catalog_in+'.mod'
    
    if (overwrite == True) and os.path.isfile(outname):
        os.remove(outname)

    input_data.write(outname, format=cat_format)
    
    return outname, gg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--params", type=str,
                        help = "Parameter file")
    parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                        action = "store_true")
    args = parser.parse_args()
    quiet = args.quiet

    params_root = re.split(".py", args.params)[0]
    if os.path.isfile(params_root+".pyc"):
        os.remove(params_root+".pyc")

    import importlib
    try:
        pipe_params = importlib.import_module(params_root)
        print('Successfully loaded "{0}" as params'.format(args.params))
        reload(pipe_params)
    except:
        print('Failed to load "{0}" as params'.format(args.params))
        raise


    catalog_in = '/data2/ken/photoz/ezpipe-bootes/Bootes_merged_Icorr_2014a_all_ap4_mags.zs.fits'
    
    outname, bf, f, fe = process(catalog_in, '{0}/{1}'.format(pipe_params.working_folder, pipe_params.translate_file),
                                '{0}/{1}'.format(pipe_params.working_folder, pipe_params.filter_file),
                                cat_format='fits')


