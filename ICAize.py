import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap

from sklearn import preprocessing as skpp
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.utils.extmath import fast_logdet
from scipy import linalg

import fnmatch
import os
import os.path
import sys
import random
import pickle as pk

from astropy.utils.compat import argparse

random_state=1234975
data_file = "{}_{}_sources_and_mixing.npz"
pickle_file = "{}_{}_pickle.pkl"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute PCA/ICA/NMF/etc. components over set of stacked spectra, save those out, and pickle model'
    )

    parser.add_argument(
        '--pattern', type=str, default='stacked*exp??????.*', metavar='PATTERN',
        help='File pattern for stacked sky fibers.'
    )
    parser.add_argument(
        '--path', type=str, default='.', metavar='PATH',
        help='Path to work from, if not ''.'''
    )

    parser.add_argument(
        '--comp_cv', action='store_true',
        help='If set, comp_min, comp_max, com_step expected; will attempt to find the best e.g. number of components via CV'
    )
    parser.add_argument(
        '--comp_max', type=int, default=50, metavar='COMP_MAX',
        help='Max number of components to use/test'
    )
    parser.add_argument(
        '--comp_min', type=int, default=0, metavar='COMP_MIN',
        help='Min number of compoenents to use/test'
    )
    parser.add_argument(
        '--comp_step', type=int, default=5, metavar='COMP_STEP',
        help='Step size from comp_min to comp_max'
    )

    parser.add_argument(
        '--n_components', type=int, default=40, metavar='N_COMPONENTS',
        help='Number of ICA/PCA/etc. components'
    )
    parser.add_argument(
        '--n_neighbors', type=int, default=10, metavar='N_NEIGHBORS',
        help='Number of neighbots for e.g. IsoMap'
    )
    parser.add_argument(
        '--scale', type=bool, default=False, metavar='SCALE',
        help='Should inputs be scaled?  Will mean subtract and value scale, but does not scale variace.'
    )
    parser.add_argument(
        '--method', type=str, default='ICA', metavar='METHOD', choices=['ICA', 'PCA', 'SPCA', 'NMF', 'ISO', 'KPCA', 'FA'],
        help='Which dim. reduction method to use'
    )
    parser.add_argument(
        '--max_iter', type=int, default=1200, metavar='MAX_ITER',
        help='Maximum number of iterations to allow for convergence.  For SDSS data 1000 is a safe number of ICA, while SPCA requires larger values e.g. ~2000 to ~2500'
    )

    parser.add_argument(
        '--ivar_cutoff', type=float, default=0.001, metavar='IVAR_CUTOFF',
        help='data with inverse variace below cutoff is masked as if ivar==0'
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
                load_all_in_dir(args.path, pattern=args.pattern, ivar_cutoff=args.ivar_cutoff)

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

    scaled_flux_arr = None
    ss = None
    if args.scale:
        ss = skpp.StandardScaler(with_std=False)
        scaled_flux_arr = ss.fit_transform(flux_arr)
    else:
        scaled_flux_arr = flux_arr

    if args.comp_cv:
        model = get_model(args.method, max_iter=args.max_iter, random_state=random_state)
        scores = []

        n_components = np.arange(args.comp_min, args.comp_max, args.comp_step)
        for n in n_components:
            model.n_components = n
            #scores.append(np.mean(cross_val_score(model, scaled_flux_arr, n_jobs=-1)))
            scores.append(ev_score_via_CV(scaled_flux_arr, model, args.method))

        print(n_components)
        print(scores)
        plt.figure()
        plt.plot(n_components, scores, 'b', label='scores')
        plt.xlabel('nb of components')
        plt.ylabel('CV scores')
        plt.show()
    else:
        sources, components, model = dim_reduce(args.method, scaled_flux_arr,
                                        args.n_components, args.n_neighbors,
                                        args.max_iter, random_state)
        np.savez(data_file.format(args.method, args.which_filter), sources=sources,
                    components=components, exposures=comb_exposure_arr,
                    wavelengths=comb_wavelengths)
        pickle((model, ss), args.path, args.method, args.which_filter)

def pickle(model, path='.', method='ICA', filter_str='both', filename=None):
    if filename is None:
        filename = pickle_file.format(method, filter_str)
    output = open(os.path.join(path, filename), 'wb')
    pk.dump(model, output)
    output.close()

def unpickle(path='.', method='ICA', filter_str='both', filename=None):
    if filename is None:
        filename = pickle_file.format(method, filter_str)
    output = open(os.path.join(path, filename), 'rb')
    model, ss = pk.load(output)
    output.close()

    return model, ss

def get_model(method, n=None, n_neighbors=None, max_iter=None, random_state=None):
    model = None

    if method == 'ICA':
        model = FastICA(whiten=True)
    elif method == 'PCA':
        model = PCA()
    elif method == 'SPCA':
        model = SparsePCA(n_jobs=-1)
    elif method == 'NMF':
        model = NMF(solver='cd')
    elif method == 'ISO':
        model = Isomap()
    elif method == 'KPCA':
        model = KernelPCA(kernel='rbf', fit_inverse_transform=True)
    elif method == 'FA':
        model = FactorAnalysis()

    if n is not None:
        model.n_components = n
    if max_iter is not None:
        model.max_iter = max_iter
    if random_state is not None:
        model.random_state = random_state
    if n_neighbors is not None:
        model.n_neighbors = n_neighbors

    return model

def get_components(method, model):
    if method == 'ICA':
        components = model.mixing_
    elif method == 'ISO':
        components = model.embedding_
    elif method == 'KPCA':
        components = model.alphas_
    else:
        components = model.components_

    return components

def dim_reduce(method, flux_arr, n=None, n_neighbors=None, max_iter=None, random_state=None):
    model = get_model(method, n, n_neighbors, max_iter, random_state)
    sources = model.fit_transform(flux_arr)
    components = get_components(method, model)

    return sources, components, model

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

def lll_score_via_CV(flux_arr, model, method, folds=3):
    print(flux_arr.shape, len(flux_arr))
    kf = KFold(len(flux_arr), n_folds=folds)
    scores = []
    for train_index, test_index in kf:
        flux_train, flux_test = flux_arr[train_index], flux_arr[test_index]
        model.fit(flux_train)

        components = get_components(method, model)
        precision = linalg.inv(np.dot(components, components.T))

        flux_test_sub = flux_test - np.mean(flux_train, axis=0)
        print(flux_test_sub.shape, len(flux_test_sub))
        log_like = np.zeros(flux_test.shape[0])
        print(log_like.shape, len(log_like))
        log_like = -0.5 * (flux_test_sub * (np.dot(flux_test_sub, precision))).sum(axis=1)
        log_like -= 0.5 * (flux_test.shape[1] * log(2 * np.pi) - fast_logdet(precision))

        scores.append(np.mean(log_like))

    return np.mean(scores)

def ev_score_via_CV(flux_arr, model, method, folds=3):
    kf = KFold(len(flux_arr), n_folds=folds)
    scores = []
    for train_index, test_index in kf:
        flux_train, flux_test = flux_arr[train_index], flux_arr[test_index]
        model.fit(flux_train)
        flux_conv_test = transform_inverse_transform(flux_test, model, method)
        #model.inverse_transform(mode.transform(flux_test))

        #scores.append(explained_variance_score(flux_test, flux_conv_test)) #, multioutput='uniform_average'))
        scores.append(r2_score(flux_test, flux_conv_test))

    return np.mean(scores)

def transform_inverse_transform(flux_arr, model, method):
    true_invtrans = None
    if method in ['PCA', 'KPCA']:
        true_invtrans = model.inverse_transfrom(mode.transform(flux_arr))

    components = get_components(method, model)
    trans_arr = model.transform(flux_arr)
    att_invtrans = components*trans_arr[:,:]

    if true_invtrans is not None:
        print(sum(true_invtrans - att_invtrans))

    return att_invtrans

if __name__ == '__main__':
    main()
