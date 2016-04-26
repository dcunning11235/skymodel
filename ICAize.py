import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn import preprocessing as skpp

import fnmatch
import os
import os.path
import sys
import random
import pickle

from astropy.utils.compat import argparse

random_state=1234975

data_file = "{}_{}_sources_and_mixing.npz"
pickle_file = "{}_{}_pickle.pkl"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute PCA/ICA/NMF/etc. components over set of stacked spectra, save those out, and pickle model')
    parser.add_argument(
        '--pattern', type=str, default='stacked*exp??????.*', metavar='PATTERN',
        help='File pattern for stacked sky fibers.'
    )
    parser.add_argument(
        '--path', type=str, default='.', metavar='PATH',
        help='Path to work from, if not ''.'''
    )
    parser.add_argument(
        '--n_components', type=int, default=40, metavar='N_COMPONENTS',
        help='Number of ICA/PCA/etc. components'
    )
    parser.add_argument(
        '--method', type=str, default='ICA', metavar='METHOD', choices=['ICA', 'PCA', 'SPCA', 'NMF']
        help='Which dim. reduction method to use'
    )
    parser.add_argument(
        '--ivar_cutoff', type=float, default=0.001, metavar='IVAR_CUTOFF',
        help='data with inverse variace below cutoff is masked as if ivar==0'
    )
    parser.add_argument(
        '--max_iter', type=int, default=1200, metavar='MAX_ITER',
        help='Maximum number of iterations to allow for convergence.  For SDSS data 1000 is a safe number of ICA, while SPCA requires larger values e.g. ~2000 to ~2500'
    )
    parser.add_argument(
        '--filter_split_path', type=str, default=None, metavar='FILTER_SPLIT_PATH',
        help='Path on which to find filter_split file'
    )
    parser.add_argument(
        '--filter_cutpoint', type=str, default=None, metavar='FILTER_CUTPOINT',
        help='Point at which to divide between ''normal'' flux and emission flux'
    )
    parser.add_argument(
        '--which_filter', type=str, default='both', metavar='WHICH_FILTER',
        help='Whether to use ''em''isson, ''nonem''isson, or ''both'''
    )
    args = parser.parse_args()

    comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = \
                load_all_in_dir(args.path, use_con_flux=False, recombine_flux=False,
                                pattern=args.pattern, ivar_cutoff=args.ivar_cutoff)

    filter_split_arr = None
    if args.filter_split_path is not None:
        fstable = Table.read(args.filter_split_path, format="ascii.csv")
	filter_split_arr = fstable["flux_kurtosis_per_wl"] < args.filter_cutpoint

    mask_summed = np.sum(comb_masks, axis=0)
    min_val_ind = np.min(np.where(mask_summed == 0))
    max_val_ind = np.max(np.where(mask_summed == 0))
    print "For data set, minimum and maximum valid indecies are:", (min_val_ind, max_val_ind)

    flux_arr = comb_flux_arr
    if filter_split_arr is not None and args.which_filter != "both":
        flux_arr = np.array(comb_flux_arr, copy=True)

        if args.which_filter == "nonem":
            new_flux_arr[:,filter_split_arr] = 0
        elif args.which_filter == "em":
            new_flux_arr[:,~filter_split_arr] = 0

    sources, components, model = dim_reduce(flux_arr, args.n_components, args.method, args.max_iter, random_state)
    np.savez(data_file.format(args.method, args.which_filter), sources=sources, components=components,
                exposures=comb_exposure_arr, wavelengths=comb_wavelengths)
    pickle(model, args.path, args.method, args.which_filter)


def pickle(model, path='.', method='ICA', filter_str='both', filename=None):
    if filename is None:
        filename = pickle_file.format(method, filter_str)
    output = open(os.path.join(path, filename), 'wb')
    pickle.dump(model, output)
    output.close()

def unpickle(path='.', method='ICA', filter_str='both', filename=None):
    if filename is None:
        filename = pickle_file.format(method, filter_str)
    output = open(os.path.join(path, filename), 'rb')
    model = pickle.load(output)
    output.close()

    return model

def dim_reduce(flux_arr, n, method, max_iter, random_state):
    model = None

    if method == 'ICA':
        model = FastICA(n_components = n, whiten=True, max_iter=max_iter,
                        random_state=random_state, w_init=mixing)
    elif method == 'PCA':
        model = PCA(n_components = n)
    elif method == 'SPCA':
        model = SparsePCA(n_components = n, max_iter=max_iter,
                        random_state=random_state, n_jobs=-1)
    elif method == 'NMF':
        model = NMF(n_components = n, solver='cd', max_iter=max_iter,
                    random_state=random_state)

    sources = model.fit_transform(flux_arr)
    if method == 'ICA':
        components_ = model.mixing_
    else:
        components_ = model.components_

    return source, components_, model

def load_all_in_dir(path, pattern, ivar_cutoff=0):
    flux_list = []
    exp_list = []
    mask_list = []
    ivar_list = []
    wavelengths = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            if file.endswith(".csv"):
                data = Table(Table.read(os.path.join(path, file), format="ascii.csv"), masked=True)
            elif file.endswith(".fits"):
                data = Table(Table.read(os.path.join(path, file), format="fits"), masked=True)

            mask = data['ivar'] <= ivar_cutoff
            ivar_list.append(np.array(data['ivar'], copy=False))

            exp = file.split("-")[2][3:]
            if exp.endswith("csv"):
                exp = int(exp[:-4])
            elif exp.endswith("fits"):
                exp = int(exp[:-5])
            else:
                exp = int(exp)

            if wavelengths is None:
                wavelengths = np.array(data['wavelength'], copy=False)

            flux_list.append(np.array(data['flux'], copy=False))
            mask_list.append(mask)
            exp_list.append(exp)

    flux_arr = np.array(flux_list)
    exp_arr = np.array(exp_list)
    mask_arr = np.array(mask_list)
    ivar_arr = np.array(ivar_list)

    return flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths

if __name__ == '__main__':
    main()
