import numpy as np

from astropy.table import Table
import fnmatch
import sys
import datetime as dt
import posixpath
import os

from astropy.utils.compat import argparse

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Scan stacked sky fibers for negative fluxes... which should not happen, but in fact does.')
    parser.add_argument(
        '--pattern', type=str, default='stacked*exp??????.*', metavar='PATTERN',
        help='File pattern for stacked sky fibers.'
    )
    parser.add_argument(
        '--path', type=str, default='.', metavar='PATH',
        help='Path to work from, if not ''.'''
    )
    parser.add_argument(
        '--rewrite', action='store_true',
        help='Flag to control whether or not negative values are replaced with 0'
    )
    parser.add_argument(
        '--ltzero', type=float, default=0.5, metavar='LTZERO',
        help='Value below zero to consider negative'
    )
    args = parser.parse_args()

    flux_list = []
    exp_list = []
    mask_list = []
    wavelengths = None

    for file in os.listdir(args.path):
        if fnmatch.fnmatch(file, args.pattern):
            if file.endswith('.fits'):
                data = Table.read(os.path.join(args.path, file), format="fits")
            elif file.endswith('.csv'):
                data = Table.read(os.path.join(args.path, file), format="ascii.csv")

            #mask = data['ivar'] == 0

            #Get rid of shit like this, if going to not just be my hacky util
            exp = int(file.split("-")[2][3:9])

            neg_mask = data['flux'] < -(args.ltzero)
            set_neg_mask = neg_mask #& ~mask
            if np.any(set_neg_mask):
                print file, exp, data['wavelength'][set_neg_mask][0], data['flux'][set_neg_mask][0],
                if not args.rewrite:
                    print "FOUND"
                else:
                    print "REPAIRING..."
                    data['ivar'][set_neg_mask] = 0
                    data['flux'][set_neg_mask] = 0

                    if file.endswith('.fits'):
                        data.write(os.path.join(args.path, file), format="fits", overwrite=True)
                    elif file.endswith('.csv'):
                        data.write(os.path.join(args.path, file), format="ascii.csv", overwrite=True)

if __name__ == '__main__':
    main()
