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
from sklearn.decomposition import NMF
from sklearn.decomposition import DictionaryLearning
from sklearn.manifold import Isomap

from sklearn import preprocessing as skpp
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from sklearn.covariance import MinCovDet
from sklearn.covariance import LedoitWolf

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

import arrayize

random_state=1234975
data_file = "{}_sources_and_mixing.npz"
pickle_file = "{}_pickle.pkl"

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
        '--compacted_path', type=str, default=None, metavar='COMPATED_PATH',
        help='Path to find compacted/arrayized data; setting this will cause --path, --pattern to be ignored'
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
        '--comp_plot', action='store_true',
        help='If set, does a 2-component plot (the first two) for the specified method'
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
        '--scale', action='store_true',
        help='Should inputs be scaled?  Will mean subtract and value scale, but does not scale variace.'
    )
    parser.add_argument(
        '--method', type=str, default='ICA', metavar='METHOD', #choices=['ICA', 'PCA', 'SPCA', 'NMF', 'ISO', 'KPCA', 'FA'],
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
    args = parser.parse_args()

    if args.compacted_path is not None:
        comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_wavelengths = arrayize.load_compacted_data(args.compacted_path)
        comb_masks = comb_ivar_arr <= args.ivar_cutoff
    else:
        comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = \
                load_all_in_dir(args.path, pattern=args.pattern, ivar_cutoff=args.ivar_cutoff)

    mask_summed = np.sum(comb_masks, axis=0)
    min_val_ind = np.min(np.where(mask_summed == 0))
    max_val_ind = np.max(np.where(mask_summed == 0))
    print "For data set, minimum and maximum valid indecies are:", (min_val_ind, max_val_ind)
    for i in range(comb_flux_arr.shape[0]):
        comb_flux_arr[i,:min_val_ind] = 0
        comb_flux_arr[i,max_val_ind+1:] = 0

    flux_arr = comb_flux_arr.astype(dtype=np.float64)
    scaled_flux_arr = None
    ss = None
    if args.scale:
        ss = skpp.StandardScaler(with_std=False)
        scaled_flux_arr = ss.fit_transform(flux_arr)
    else:
        scaled_flux_arr = flux_arr

    if args.comp_plot:
        sources, components, model = dim_reduce(args.method, flux_arr if args.method == 'NMF' else scaled_flux_arr,
                                        args.n_components, args.n_neighbors,
                                        args.max_iter, random_state)
        trans_flux_arr = model.transform(flux_arr if args.method == 'NMF' else scaled_flux_arr)

        if args.n_components == 2:
            plt.scatter(trans_flux_arr[:,0], trans_flux_arr[:,1])
        elif args.n_components == 3:
            f, (ax1, ax2) = plt.subplots(2)
            ax1.scatter(trans_flux_arr[:,0], trans_flux_arr[:,1])
            ax2.scatter(trans_flux_arr[:,1], trans_flux_arr[:,2])
        elif args.n_components == 4:
            f, (ax1, ax2, ax3) = plt.subplots(3)
            ax1.scatter(trans_flux_arr[:,0], trans_flux_arr[:,1])
            ax2.scatter(trans_flux_arr[:,1], trans_flux_arr[:,2])
            ax3.scatter(trans_flux_arr[:,2], trans_flux_arr[:,3])
        plt.show()
    elif args.comp_cv:
        methods = args.method.split(',')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for method in methods:
            model = get_model(method, max_iter=args.max_iter, random_state=random_state)
            scores = []
            ll_scores = []

            n_components = np.arange(args.comp_min, args.comp_max, args.comp_step)
            for n in n_components:
                print("Cross validating for n=", n, "on method", method)

                model.n_components = n
                scores.append(ev_score_via_CV(flux_arr if method == 'NMF' else scaled_flux_arr, model, method))

                if method == 'FA' or method == 'PCA':
                    ll_scores.append(ll_score_via_CV(scaled_flux_arr, model, method))

            if method == 'FA' or method == 'PCA':
                ax2.axhline(cov_mcd_score(scaled_flux_arr, args.scale), color='violet', label='MCD Cov', linestyle='--')
                ax2.axhline(cov_lw_scoare(scaled_flux_arr, args.scale), color='orange', label='LW Cov', linestyle='--')

            print(n_components)
            print(scores)
            ax1.plot(n_components, scores, label=(method + ' scores'))

            if len(ll_scores) > 0:
                print(ll_scores)
                ax2.plot(n_components, ll_scores, '-.', label=(method + ' ll scores'))

        ax1.set_xlabel('nb of components')
        ax1.set_ylabel('CV scores', figure=fig)

        ax1.legend(loc='lower left')
        ax2.legend(loc='lower right')

        plt.show()
    else:
        sources, components, model = dim_reduce(args.method, flux_arr if args.method == 'NMF' else scaled_flux_arr,
                                        args.n_components, args.n_neighbors,
                                        args.max_iter, random_state)
        serialize_data(sources, components, comb_exposure_arr, comb_wavelengths,
                        args.path, args.method)
        pickle_model((model, ss), args.path, args.method)

def serialize_data(sources, components, exposures, wavelengths, path='.', method='ICA',
                    filename=None):
    if filename is None:
        filename = data_file.format(method)
    np.savez(os.path.join(path, filename), sources=sources, components=components,
            exposures=exposures, wavelengths=wavelengths)

def deserialize_data(path='.', method='ICA', filename=None):
    if filename is None:
        filename = data_file.format(method)
    npz = np.load(os.path.join(path, filename))

    sources = npz['sources']
    exposures = npz['exposures']
    wavelengths = npz['wavelengths']
    components = npz['components']

    npz.close()
    return sources, components, exposures, wavelengths

def pickle_model(model, path='.', method='ICA', filename=None):
    if filename is None:
        filename = pickle_file.format(method)
    file = open(os.path.join(path, filename), 'wb')
    pk.dump(model, file)
    file.close()

def unpickle_model(path='.', method='ICA', filename=None):
    if filename is None:
        filename = pickle_file.format(method)
    file = open(os.path.join(path, filename), 'rb')
    model, ss = pk.load(file)
    file.close()

    return model, ss

def get_model(method, n=None, n_neighbors=None, max_iter=None, random_state=None):
    model = None

    if method == 'ICA':
        model = FastICA(whiten=True)
    elif method == 'PCA':
        model = PCA()
    elif method == 'SPCA':
        model = SparsePCA()
    elif method == 'NMF':
        model = NMF(solver='cd')
    elif method == 'ISO':
        model = Isomap()
    elif method == 'KPCA':
        model = KernelPCA(kernel='rbf', fit_inverse_transform=False, gamma=1, alpha=0.0001)
    elif method == 'FA':
        model = FactorAnalysis()
    elif method == 'DL':
        model = DictionaryLearning(split_sign=True, fit_algorithm='cd', alpha=1)

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
    if method == 'ISO':
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

def ll_score_via_CV(flux_arr, model, method, folds=3):
    kf = KFold(len(flux_arr), n_folds=folds)
    scores = []
    for train_index, test_index in kf:
        flux_train, flux_test = flux_arr[train_index], flux_arr[test_index]
        model.fit(flux_train)
        scores.append(model.score(flux_test))

    return np.mean(scores)

def ev_score_via_CV(flux_arr, model, method, folds=3):
    kf = KFold(len(flux_arr), n_folds=folds)
    scores = []

    for train_index, test_index in kf:
        flux_train, flux_test = flux_arr[train_index], flux_arr[test_index]

        if method == 'KPCA':
            model.fit_inverse_transform = False
            model.fit(flux_train)

            sqrt_lambdas = np.diag(np.sqrt(model.lambdas_))
            X_transformed = np.dot(model.alphas_, sqrt_lambdas)
            n_samples = X_transformed.shape[0]
            K = model._get_kernel(X_transformed)
            K.flat[::n_samples + 1] += model.alpha

            model.dual_coef_ = linalg.solve(K, flux_train)
            model.X_transformed_fit_ = X_transformed
            model.fit_inverse_transform = True

            flux_conv_test = transform_inverse_transform(flux_test, model, method)
        else:
            model.fit(flux_train)
            flux_conv_test = transform_inverse_transform(flux_test, model, method)

        scores.append(explained_variance_score(flux_test, flux_conv_test, multioutput='uniform_average'))

    return np.mean(scores)

def transform_inverse_transform(flux_arr, model, method):
    if method in ['PCA', 'KPCA', 'ICA']:
        att_invtrans = model.inverse_transform(model.transform(flux_arr))
    else:
        components = get_components(method, model)
        trans_arr = model.transform(flux_arr)

        att_invtrans = np.zeros(shape=(trans_arr.shape[0], components.shape[1]), dtype=float)
        for flux_n in xrange(trans_arr.shape[0]):
            for comp_n in xrange(components.shape[0]):
                rec_comp = trans_arr[flux_n, comp_n] * components[comp_n]
                att_invtrans[flux_n] += rec_comp

    return att_invtrans

def cov_mcd_score(flux_arr, assume_centered):
    mcd = MinCovDet(assume_centered=assume_centered)
    return np.mean(cross_val_score(mcd, flux_arr))

def cov_lw_score(flux_arr, assume_centered):
    lw = LedoitWolf(assume_centered=assume_centered)
    return np.mean(cross_val_score(lw, flux_arr))


if __name__ == '__main__':
    main()
