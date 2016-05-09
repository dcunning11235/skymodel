import os.path

import numpy as np

import astropy.table as astab
import astropy.coordinates as ascoord

from astropy.utils.compat import argparse

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Add in solar and ecliptic data to observational metadata'
    )

    parser.add_argument(
        '--metadata_file_path', type=str, default=None, metavar='PATH',
        help='Metadata file path to work from, if not ''.'''
    )
    parser.add_argument(
        '--metadata_path', type=str, default='.', metavar='PATH',
        help='Metadata path to work from, if not ''.'''
    )
    parser.add_argument(
        '--input', type=str, default='FITS', metavar='OUTPUT',
        help='Output format, either of FITS or CSV, defaults to FITS.'
    )
    parser.add_argument(
        '--solar_metadata', type=str, default='solar_ephemeris.csv',
        help='Solar metadata file (from parse_ephemeris.py)'
    )
    args = parser.parse_args()

    if args.metadata_file_path is None:
        if args.input == 'CSV':
            obs_md_table = astab.Table.read(os.path.join(args.metadata_path, "annotated_metadata.csv"), format="ascii.csv")
        elif args.input == 'FITS':
            obs_md_table = astab.Table.read(os.path.join(args.metadata_path, "annotated_metadata.fits"), format="fits")
    else:
        obs_md_table = Table.read(args.metadata_file_path)
    solar_md_table = astab.Table.read(args.solar_metadata, format="ascii.csv")
    solar_md_table.rename_column('UTC', 'EPHEM_DATE')

    obs_md_table = astab.join(obs_md_table, solar_md_table['EPHEM_DATE', 'RA_ABS', 'DEC_ABS'])

    boresight_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA'], dec=obs_md_table['DEC'], distance=1.0, unit=('deg', 'deg', 'AU'), frame='fk5')
    boresight_ecliptic = boresight_ra_dec.transform_to('heliocentrictrueecliptic')

    solar_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA_ABS'], dec=obs_md_table['DEC_ABS'], distance=1.0, unit=('deg', 'deg', 'AU'), frame='icrs')
    solar_ecliptic = solar_ra_dec.transform_to('heliocentrictrueecliptic')

    obs_md_table.add_column(astab.Column(boresight_ecliptic.lat, dtype=float, name="ECLIPTIC_PLANE_SEP"))
    belp = np.mod(boresight_ecliptic.lon.value + 360.0, 360.0)
    selp = np.mod(solar_ecliptic.lon.value + 360.0, 360.0)
    lon_diff = np.abs(belp - selp)
    lon_diff[lon_diff > 180] -= 360
    lon_diff = np.abs(lon_diff)
    obs_md_table.add_column(astab.Column(lon_diff, dtype=float, name="ECLIPTIC_PLANE_SOLAR_SEP"))

    obs_md_table.remove_columns(['RA_ABS', 'DEC_ABS'])

    if args.metadata_file_path is None:
        if args.input == 'CSV':
            obs_md_table.write(os.path.join(args.metadata_path, "annotated_metadata.csv"), format="ascii.csv")
        elif args.input == 'FITS':
            obs_md_table.write(os.path.join(args.metadata_path, "annotated_metadata.fits"), format="fits", overwrite=True)
    else:
        obs_md_table.write(args.metadata_file_path, overwrite=True)

if __name__ == '__main__':
    main()
