import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
import numpy.random as rand
import astropy.coordinates as coord
import astropy.units as u
import datetime as dt
from functools import partial
from astropy.utils.compat import argparse

import random_forest_spectra as rfs
import ICAize as iz
import animate as ani

from matplotlib import animation

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
        '--model_path', type=str, default='model.pkl', metavar='MODEL_PATH',
        help='COMPLETE path from which to load a model'
    )
    parser.add_argument(
        '--start_dt', type=ani.valid_date, help='DateTime to plot sky for'
    )
    parser.add_argument(
        '--end_dt', type=ani.valid_date, help='DateTime to plot sky for'
    )
    parser.add_argument(
        '--lunar_metadata', type=str, default=None, metavar='LUNAR_METADATA',
        required=True, help='File containing lunar ephemeris metadata.'
    )
    parser.add_argument(
        '--solar_metadata', type=str, default=None, metavar='SOLAR_METADATA',
        required=True, help='File containing solar ephemeris metadata.'
    )
    parser.add_argument(
        '--sunspot_metadata', type=str, default=None, metavar='SUNSPOT_METADATA',
        required=True, help='File containing sunspot metadata.'
    )
    parser.add_argument(
        '--method', type=str
    )
    parser.add_argument(
        '--ra', type=str
    )
    parser.add_argument(
        '--dec', type=str
    )
    parser.add_argument(
        '--dm_path', type=str
    )
    args = parser.parse_args()

    obs_coord = coord.SkyCoord(args.ra, args.dec, frame='icrs')
    metadata_tups = ani.get_sky_for_coord(args.start_dt, args.end_dt, obs_coord, args.lunar_metadata,
                        args.solar_metadata, args.sunspot_metadata)
    spectra, labels = ani.animate_sky_spectra_for_coord(args.start_dt, args.end_dt, obs_coord, args.lunar_metadata,
                        args.solar_metadata, args.sunspot_metadata, args.model_path, args.dm_path, args.method)

    #print(spectra)
    xscale = np.arange(len(spectra[0]))
    #print(xscale)

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 7200), ylim=(0, 500))
    line, = ax.plot([], [])

    def init_func():
        line.set_data([], [])
        return line,

    def animate_func(i, data, xscale):
        line.set_data(xscale, data[i])
        plt.title(labels[i])
        return line,

    p_animate_func = partial(animate_func, data=spectra, xscale=xscale)

    anim = animation.FuncAnimation(fig, p_animate_func, init_func=init_func, frames=len(spectra), interval=1000) #, blit=True)
    anim.save('polaris_sky_animation.mp4', fps=1, extra_args=['-vcodec', 'libx264'])
    plt.show()

if __name__ == '__main__':
    main()
