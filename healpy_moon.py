import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
import numpy.random as rand
import astropy.coordinates as coord
import astropy.units as u
import datetime as dt

from astropy.utils.compat import argparse

import random_forest_spectra as rfs
import ICAize as iz
import animate as ani

NSIDE = 64

def valid_date(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Build and test models based on dim reductions and provided spectra'
    )
    parser.add_argument(
        '--metadata_path', type=str, default='.', metavar='PATH',
        help='Metadata path to work from, if not ''.'''
    )
    parser.add_argument(
        '--lunar_metadata', type=str, default=None, metavar='LUNAR_METADATA',
        required=True, help='File containing lunar ephemeris metadata.'
    )
    parser.add_argument(
        '--start_dt', type=ani.valid_date, help='DateTime to plot sky for'
    )
    parser.add_argument(
        '--end_dt', type=ani.valid_date, help='DateTime to plot sky for'
    )
    parser.add_argument(
        '--ra', type=str
    )
    parser.add_argument(
        '--dec', type=str
    )
    args = parser.parse_args()

    m = np.zeros(hp.nside2npix(NSIDE))
    lunar_row, solar_row, ss_count, ss_area = ani.get_metadata_for_dt(args.datetime, args.lunar_metadata, args.solar_metadata, args.sunspot_metadata)

    


    fig = plt.figure(1, figsize=(10, 7.5))
    hp.mollview(m, coord=['C'], title="Mollview image RING", fig=1)
    plt.show()

if __name__ == '__main__':
    main()
