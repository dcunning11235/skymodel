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

NSIDE = 16

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
        '--model_path', type=str, default='model.pkl', metavar='MODEL_PATH',
        help='COMPLETE path from which to load a model'
    )
    parser.add_argument(
        '--datetime', type=valid_date, help='DateTime to plot sky for'
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
    args = parser.parse_args()

    m = np.zeros(hp.nside2npix(NSIDE))

    model = rfs.load_model(args.model_path)
    lunar_row, solar_row, ss_count, ss_area = rfs.load_metadata_for_dt_altaz(args.datetime, args.lunar_metadata, args.solar_metadata, args.sunspot_metadata)
    dm, ss = iz.unpickle_model(path='.', method=args.method)
    print(ss, ss.mean_)
    print(dm)

    #base_coord = coord.SkyCoord(frame='altaz', obstime=args.datetime, location=coord.EarthLocation.from_geodetic(lat=32.780278, lon=-105.820278, height=2788))
    for i in range(hp.nside2npix(NSIDE)):
        theta, phi = hp.pixelfunc.pix2ang(NSIDE, i)
        if theta < np.pi / 3:
            theta = 90 - theta * 180.0 / np.pi
            phi = phi * 180.0 / np.pi
            base_coord = coord.SkyCoord(alt=theta * u.deg, az=phi * u.deg, frame='altaz',
                        obstime=args.datetime,
                        location=coord.EarthLocation.from_geodetic(lat=32.780278, lon=-105.820278, height=2788))
            #base_coord.alt=theta * u.deg
            #base_coord.az=phi * u.deg
            metadata = rfs.metadata_for_alt_az(base_coord, lunar_row, solar_row, ss_count, ss_area)

            np_metadata = np.array(metadata)
            pred = model.predict(np_metadata.reshape(1, -1))
            inv_spec = iz.inverse_transform(pred, dm, ss, args.method)

            m[i] = np.sum(inv_spec)/np.prod(inv_spec.shape)

    fig = plt.figure(1, figsize=(10, 7.5))
    hp.mollview(m, coord=['C'], title="Mollview image RING", fig=1)
    #hp.orthview(m, title="Mollview image RING", fig=1,min=0, max=1)
    plt.show()


if __name__ == '__main__':
    main()



#rands = rand.randint(1, 1000, len(m))
#m = m*rands
'''
m[hp.pixelfunc.ang2pix(NSIDE, 0, 0)] = 5
print(hp.pixelfunc.ang2pix(NSIDE, 0, 0))

m[hp.pixelfunc.ang2pix(NSIDE, np.pi/2, 0)] = 10
print(hp.pixelfunc.ang2pix(NSIDE, np.pi/2, 0))

m[hp.pixelfunc.ang2pix(NSIDE, np.pi, 0)] = 15
print(hp.pixelfunc.ang2pix(NSIDE, np.pi, 0))

m[hp.pixelfunc.ang2pix(NSIDE, 0, np.pi/2)] = 20
print(hp.pixelfunc.ang2pix(NSIDE, 0, np.pi/2))

m[hp.pixelfunc.ang2pix(NSIDE, 0, np.pi)] = 25
print(hp.pixelfunc.ang2pix(NSIDE, 0, np.pi))

m[hp.pixelfunc.ang2pix(NSIDE, np.pi/2, np.pi/2)] = 30
print(hp.pixelfunc.ang2pix(NSIDE, np.pi/2, np.pi/2))

m[hp.pixelfunc.ang2pix(NSIDE, np.pi, np.pi)] = 35
print(hp.pixelfunc.ang2pix(NSIDE, np.pi, np.pi))
'''
