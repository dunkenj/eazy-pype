import numpy as np
import array
import os, sys
import re


from astropy.table import Table
from astropy import units as u


def process(catalog_in, cat_format = 'ascii.commented_header',
            id_col = 'id', flux_col = 'flux',
            fluxerr_col = 'fluxerr', exclude_columns=None, 
            overwrite=True, verbose=False):
    
    input_data = Table.read(catalog_in, format=cat_format)
    if exclude_columns != None:
        input_data.remove_columns(exclude_columns)
    column_names = input_data.columns.keys()

    ID = input_data[id_col]

    filter_names = []

    flux_col_end = flux_col
    fluxerr_col_end = fluxerr_col

    k,l = 0,0
    for ii in range(len(column_names)):
        if column_names[ii].lower().endswith(flux_col_end.lower()):
            if k == 0:
                fluxes = input_data[column_names[ii]].data
            else:
                fluxes = np.column_stack((fluxes,input_data[column_names[ii]].data))
            k+=1
            filter_names.append(column_names[ii])

        if column_names[ii].lower().endswith(fluxerr_col_end.lower()):
            if l == 0:
                fluxerrs = input_data[column_names[ii]].data
            else:
                fluxerrs = np.column_stack((fluxerrs,input_data[column_names[ii]].data))
            l+=1
            

    fluxes = fluxes[:,1:-1]
    fluxerrs = fluxerrs[:,1:-1]
    fnames_full = filter_names[1:-1]
    fnames = [filt.split('_')[0] for filt in fnames_full]

    color_a = np.zeros(fluxes.shape)
    color_b = np.zeros(fluxes.shape)

    upper_lims = np.where(np.logical_and(fluxes/fluxerrs < 2., fluxerrs > 0.))
    fluxes[upper_lims] = fluxes[upper_lims] + 2.*fluxerrs[upper_lims]

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

    """
    mask = anomalous[:,4] * input_data['FLAG_DEEP'].astype('bool')
    plt.plot(input_data['ALPHA_J2000'][mask], input_data['DELTA_J2000'][mask], 'o', alpha=0.1, mew=0)

    mask = anomalous[:,9] * input_data['FLAG_DEEP'].astype('bool')
    plt.plot(input_data['ALPHA_J2000'][mask], input_data['DELTA_J2000'][mask], 'o', alpha=0.1, mew=0)

    """

    for i, name in enumerate(fnames_full):
        if verbose:
            print(name)
        input_data[name][anomalous[:,i]] = -90.
        input_data[name+'err'][anomalous[:,i]] = -90.
    
    outname = catalog_in+'.mod'
    
    if (overwrite == True) and os.path.isfile(outname):
        os.remove(outname)

    input_data.write(outname, format=cat_format)
    return outname, gg

if __name__ == '__main__':

    catalog_in = '/data2/ken/photoz/ezpipe-test/Bootes_merged_Icorr_2014a_all_ap4_mags.testsample.cat'
    exclude_columns = ['Total_flux', 'E_Total_flux']
    
    frac_removed = process(catalog_in, cat_format='ascii.commented_header', exclude_columns=exclude_columns)


