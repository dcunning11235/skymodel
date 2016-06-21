import numpy as np
import astropy.coordinates as ascoord
import astropy.units as u
import datetime as dt
import math
import pytz
import annotate_obs_metadata as aom
import sys
from astropy.table import Table
import random_forest_spectra as rfs
import ICAize as iz

from astropy.utils.compat import argparse

APACHE_POINT = ascoord.EarthLocation.from_geodetic(lat=32.780278, lon=-105.820278, height=2788)

def valid_date(s):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            return dt.datetime.strptime(s, "%Y-%b-%d %H:%M")
        except:
            msg = "Not a valid date: '{0}'.".format(s)
            raise argparse.ArgumentTypeError(msg)

def get_metadata_for_dt(dt, lunar_metadata, solar_metadata, sunspot_metadata):
    apo_tz = pytz.timezone('America/Denver')
    block_dt_str = None
    try:
        block_dt_str = [None]*len(dt)
        for i, d in enumerate(dt):
            block_dt = d + aom.get_block_delta(d)
            block_dt = apo_tz.localize(block_dt).astimezone(pytz.utc)
            block_dt_str[i] = block_dt.strftime("%Y-%b-%d %H:%M")
    except:
        print("Unexpected error:", sys.exc_info())

        block_dt = dt + aom.get_block_delta(dt)
        block_dt = apo_tz.localize(block_dt).astimezone(pytz.utc)
        block_dt_str = [ block_dt.strftime("%Y-%b-%d %H:%M") ]

    if lunar_metadata is not None and isinstance(lunar_metadata, basestring):
        lunar_data = Table.read(lunar_metadata, format="ascii.csv")
    else:
        lunar_data = lunar_metadata

    if solar_metadata is not None and isinstance(solar_metadata, basestring):
        solar_data = Table.read(solar_metadata, format="ascii.csv")
    else:
        solar_data = solar_metadata

    if if sunspot_metadata is not None and isinstance(sunspot_metadata, basestring):
        sunspot_md_table = Table.read(sunspot_metadata, format="ascii.csv")
    else:
        sunspot_md_table = sunspot_metadata

    metadata = [None]*len(block_dt_str)
    for i, dt_str in enumerate(block_dt_str):
        lunar_row = None, solar_row = None, ss_count = None, ss_area = None

        if lunar_data is not None:
            lunar_row = lunar_data[lunar_data['UTC'] == dt_str]
        if solar_data is not None:
            solar_row = solar_data[solar_data['UTC'] == dt_str]
        if sunspot_md_table is not None:
            ss_count, ss_area = aom.find_sunspot_data(dt_str, sunspot_md_table)

        metadata[i] = (dt_str, lunar_row, solar_row, ss_count, ss_area)

    return metadata, lunar_data, solar_data, sunspot_md_table

def metadata_for_alt_az(base_altaz_coord, lunar_row, solar_row, ss_count, ss_area):
    base_radec_coord = base_altaz_coord.transform_to('icrs')
    base_radec_coord = ascoord.SkyCoord(base_radec_coord, distance = 1000000.0 * u.kpc, frame='icrs')
    airmass = 1.0/np.cos(base_altaz_coord.alt.value / 180.0 * np.pi)

    lunar_radec_coord = ascoord.SkyCoord(ra=lunar_row['RA_ABS'], dec=lunar_row['DEC_ABS'], unit='deg', frame='icrs')
    lunar_sep = base_radec_coord.separation(lunar_radec_coord)

    solar_radec_coord = ascoord.SkyCoord(ra=solar_row['RA_ABS'], dec=solar_row['DEC_ABS'], unit='deg', frame='icrs')
    solar_sep = base_radec_coord.separation(solar_radec_coord)

    galactic_core_coord = ascoord.SkyCoord(l=0, b=0, unit='deg', frame='galactic')
    galactic_sep = base_radec_coord.separation(galactic_core_coord)
    base_galactic_coord = base_altaz_coord.transform_to('galactic')

    base_ecliptic = base_radec_coord.transform_to('heliocentrictrueecliptic')

    solar_ra_dec = ascoord.SkyCoord(ra=solar_row['RA_ABS'], dec=solar_row['DEC_ABS'], distance=1.0, unit=('deg', 'deg', 'AU'), frame='icrs')
    solar_ecliptic = solar_ra_dec.transform_to('heliocentrictrueecliptic')

    belp = np.mod(base_ecliptic.lon.value + 360.0, 360.0)
    selp = np.mod(solar_ecliptic.lon.value + 360.0, 360.0)
    lon_diff = np.abs(belp - selp)
    lon_diff[lon_diff > 180] -= 360
    lon_diff = np.abs(lon_diff)

    return base_radec_coord.ra.value, base_radec_coord.dec.value, base_altaz_coord.az.value, \
            base_altaz_coord.alt.value, airmass, ss_count[0], ss_area[0], lunar_row['MG_APP'][0], \
            lunar_row['ELV_APP'][0], lunar_sep.value[0], solar_row['ELV_APP'][0], \
            solar_sep.value[0], galactic_sep.value, base_galactic_coord.l.value, \
            base_ecliptic.lat.value, lon_diff[0]

# in LOCAL time
def get_whole_sky_for_datetime(obs_time):
    pass

def get_sky_for_coord(obs_time_start, obs_time_end, point_coord, lunar_metadata_file,
                    solar_metadata_file, sunspot_metadata_file):
    d_time = dt.timedelta(minutes=15)
    diff_times = obs_time_end - obs_time_start
    chunks = int(math.ceil(diff_times.total_seconds() / 900))
    times = [obs_time_start] * chunks
    for i in range(1, chunks):
        times[i] = times[i-1] + d_time

    #datetime_str, lunar_row, solar_row, ss_count, ss_area
    metadata_tups, lunar_data, solar_data, sunspot_data = get_metadata_for_dt(times,
                        lunar_metadata_file, solar_metadata_file, sunspot_metadata_file)

    model_metadata_tups = [None]*len(metadata_tups)
    dates = [None]*len(metadata_tups)
    for i, raw_metadata in enumerate(metadata_tups):
        APACHE_POINT_FRAME = ascoord.AltAz(location=APACHE_POINT, obstime=valid_date(raw_metadata[0]))
        base_coord = point_coord.transform_to(APACHE_POINT_FRAME)

        model_metadata_tups[i] = metadata_for_alt_az(base_coord, raw_metadata[1],
                            raw_metadata[2], raw_metadata[3], raw_metadata[4])
        dates[i] = raw_metadata[0]

    return model_metadata_tups, dates, lunar_data, solar_data, sunspot_data

def animate_sky_spectra_for_coord(obs_time_start, obs_time_end, point_coord, lunar_metadata_file,
                    solar_metadata_file, sunspot_metadata_file, model_path, dm_path, dm_method):
    metadata_tups, dates, lunar_data, solar_data, sunspot_data = get_sky_for_coord(obs_time_start,
                    obs_time_end, point_coord, lunar_metadata_file, solar_metadata_file, sunspot_metadata_file)

    model = rfs.load_model(model_path)
    dm, ss, model_args = iz.unpickle_model(path=dm_path, method=dm_method)

    inv_spec = []
    labels = []
    for i, metadata in enumerate(metadata_tups):
        #print(metadata)
        np_metadata = np.array(metadata)
        pred = model.predict(np_metadata.reshape(1, -1))
        inv_spec.append(iz.inverse_transform(pred, dm, ss, dm_method, model_args)[0, :])
        labels.append(dates[i] + "(ALT,AZ): (" + str(metadata[3]) + ", " + str(metadata[2]) + ")")

    return inv_spec, labels
