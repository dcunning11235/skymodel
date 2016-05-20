import numpy as np
from sklearn import preprocessing as skpp
from astropy.utils.compat import argparse
from random import randint as randint
import ICAize as iz

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
        '--n_components', type=int, default=40, metavar='N_COMPONENTS',
        help='Number of ICA/PCA/etc. components'
    )
    parser.add_argument(
        '--method', type=str, default='ICA', metavar='METHOD',
        choices=['ICA', 'PCA', 'SPCA', 'NMF', 'ISO', 'KPCA', 'FA', 'DL'],
        help='Which dim. reduction method to use'
    )
    parser.add_argument(
        '--scale', action='store_true',
        help='Should inputs variance be scaled?  Defaults to mean subtract and value scale, but w/out this does not scale variance.'
    )
    parser.add_argument(
        '--ivar_cutoff', type=float, default=0.001, metavar='IVAR_CUTOFF',
        help='data with inverse variace below cutoff is masked as if ivar==0'
    )
    parser.add_argument(
        '--n_iter', type=int, default=1200, metavar='MAX_ITER',
        help='Maximum number of iterations to allow for convergence.  For SDSS data 1000 is a safe number of ICA, while SPCA requires larger values e.g. ~2000 to ~2500'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=None, metavar='N_JOBS',
        help='N_JOBS'
    )
    args = parser.parse_args()


    comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = iz.load_data(args)
    model = iz.get_model(args.method, n=args.n_components, n_neighbors=None, max_iter=args.n_iter, random_state=iz.random_state, n_jobs=args.n_jobs)

    ss = skpp.StandardScaler(with_std=False)
    if args.scale:
        ss = skpp.StandardScaler(with_std=True)
    scaled_flux_arr = ss.fit_transform(comb_flux_arr)

    #Heavily copied from J. Vanderplas/astroML bayesian_blocks.py
    N = comb_wavelengths.size
    step = args.n_components * 7

    edges = np.concatenate([comb_wavelengths[:1:step],
                            0.5 * (comb_wavelengths[1::step] + comb_wavelengths[:-1:step]),
                            comb_wavelengths[-1::step]])
    block_length = comb_wavelengths[-1::step] - edges

    # arrays to store the best configuration
    nn_vec = np.ones(N/step) * step
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    for R in range(N/step):
        print("R: " + str(R))

        width = block_length[:R + 1] - block_length[R + 1]
        count_vec = np.cumsum(nn_vec[:R + 1][::-1])[::-1]

        #width = nn_vec[:R + 1] - nn_vec[R + 1]
        #count_vec = np.cumsum(nn_vec[:R + 1][::-1])[::-1]

        #print(width)
        #print(count_vec)
        #raw_input("Pausing... ")

        fit_vec = map(lambda n: iz.score_via_CV(['LL'], scaled_flux_arr[:, :n], model, ss, args.method, folds=3, n_jobs=args.n_jobs), count_vec)
        fit_vec = [d["mle"] for d in fit_vec]

        #print(fit_vec)
        fit_vec[1:] += best[:R]
        #print(fit_vec)

        i_max = np.argmax(fit_vec)
        last[R] = i_max
        best[R] = fit_vec[i_max]

        #print(best)

    change_points =  np.zeros(N/step, dtype=int)
    i_cp = N/step
    ind = N/step
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    print(edges[change_points])


    '''
    t = []
    for i in range(100):
        t.append(randint(1, 30))
    bayesian_blocks(t)
    '''

def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges
    print("Edges:", edges)
    print("Block_length:", block_length)

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]
        print("fit_vec:", fit_vec)
        raw_input("Pausing...")

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

        print("Best:", best)
    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]

if __name__ == '__main__':
    main()
