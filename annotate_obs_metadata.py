import numpy as np
from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join
import astropy.coordinates as ascoord
import sys
import datetime as dt
import bossdata.path as bdpath
import bossdata.remote as bdremote
import bossdata.spec as bdspec

from progressbar import ProgressBar, Percentage, Bar

from astropy.utils.compat import argparse

ephemeris_block_size_minutes = 15
ephemeris_block_size = dt.timedelta(minutes=ephemeris_block_size_minutes)
ephemeris_max_block_size = dt.timedelta(minutes=1.5*ephemeris_block_size_minutes)

def find_ephemeris_lookup_date(tai_beg, tai_end, obs_md_table):
    '''
    Want to find the 15-minute increment (0, 15, 30, 45) that is sandwiched between the two
    passed datetimes.  However, spans can be less than 15 minutes, so also need handle this
    case; here, will round to whichever increment has the smallest delta between tai_beg
    and tai_end.  I've not seen it, but I have to assume that spans can also be longer than
    15 minutes.
    '''
    vectfunc = np.vectorize(tai_str_to_datetime)
    tai_end_dt = vectfunc(tai_end)
    tai_beg_dt = vectfunc(tai_beg)

    vectfunc = np.vectorize(get_ephemeris_block_in_interval)

    mask = (tai_end_dt - tai_beg_dt) <= ephemeris_max_block_size
    ret = np.zeros((len(tai_end_dt),), dtype=dt.datetime)
    ret[mask] = vectfunc(tai_beg_dt[mask], tai_end_dt[mask])

    def _lookup_str_format(dtval):
        if isinstance(dtval, dt.datetime):
            return dtval.strftime("%Y-%b-%d %H:%M")
        return ""

    vectfunc = np.vectorize(_lookup_str_format)
    ret = vectfunc(ret)

    return ret[mask], obs_md_table[mask]

def find_sunspot_data(ephemeris_block, sunspot_md_table):
    count_ret = np.zeros((len(ephemeris_block,), ), dtype=float)
    area_ret = np.zeros((len(ephemeris_block,), ), dtype=float)

    #terrible for performance, but one-time only
    for i, somedate in enumerate(ephemeris_block):
        lookup_str = ephemeris_block[i].split()[0]
        result = sunspot_md_table[sunspot_md_table["DATE"]==lookup_str]["SESC","SSAREA"]
        count_ret[i], area_ret[i] = result["SESC"][0], result["SSAREA"][0]

    return count_ret, area_ret

def get_ephemeris_block_in_interval(tai_beg, tai_end):
    tai_end_dt = tai_str_to_datetime(tai_end)
    tai_beg_dt = tai_str_to_datetime(tai_beg)
    tai_beg_block = round_tai(tai_beg_dt)
    tai_end_block = round_tai(tai_end_dt)

    if tai_beg_block < tai_end_dt  and  tai_beg_block >= tai_beg_dt:
        return tai_beg_block
    elif tai_end_block < tai_end_dt  and  tai_end_block >= tai_beg_dt:
        return tai_end_block
    else:
        end_delta = get_tai_block_delta(tai_end_dt, "down")
        beg_delta = get_tai_block_delta(tai_beg_dt, "up")
        if abs(beg_delta) < abs(end_delta):
            return tai_beg_dt + beg_delta
        else:
            return tai_end_dt + end_delta

def tai_str_to_datetime(tai):
    if isinstance(tai, basestring):
        if tai.count(':') == 2:
            if tai.count('.') == 1:
                tai = dt.datetime.strptime(tai, "%Y-%m-%dT%H:%M:%S.%f")
                tai = tai.replace(microsecond=0)
            else:
                tai = dt.datetime.strptime(tai, "%Y-%m-%dT%H:%M:%S")
        elif tai.count(':') == 1:
            tai = dt.datetime.strptime(tai, "%Y-%m-%dT%H:%M")

    return tai

def get_tai_block_delta(tai, direction="closest"):
    tai = tai_str_to_datetime(tai)

    return get_block_delta(tai, direction)

def get_block_delta(dt, direction="closest"):
    delta_mins = - ((tai.minute % ephemeris_block_size_minutes) + (tai.second / 60.0))
    if direction == 'closest':
        if delta_mins <= -ephemeris_block_size_minutes/2:
            delta_mins = ephemeris_block_size_minutes + delta_mins
    elif direction == 'down':
        #nada
        True
    elif direction == 'up':
        delta_mins = ephemeris_block_size_minutes + delta_mins

    return dt.timedelta(minutes=delta_mins)

def round_tai(tai):
    tai = tai_str_to_datetime(tai)
    tdelta = get_tai_block_delta(tai)

    return tai + tdelta

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Pull metadata from FITS for PLATE/MJD combos, output.')

    parser.add_argument(
        '--obs_metadata', type=str, default=None, metavar='OBS_METADATA',
        required=True, help='File containing observation metadata.'
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
        '--output', type=str, default='FITS', metavar='OUTPUT',
        help='Output format, either of FITS or CSV, defaults to FITS.'
    )
    args = parser.parse_args()

    if args.output.upper() == 'CSV':
        obs_md_table = Table.read(args.obs_metadata, format="ascii.csv")
    elif args.output.upper == 'FITS':
        obs_md_table = Table.read(args.obs_metadata, format="fits")
    else:
        obs_md_table = Table.read(args.obs_metadata)

    lunar_md_table = Table.read(args.lunar_metadata, format="ascii.csv")
    lunar_md_table.rename_column('UTC', 'EPHEM_DATE')
    solar_md_table = Table.read(args.solar_metadata, format="ascii.csv")
    solar_md_table.rename_column('UTC', 'EPHEM_DATE')
    sunspot_md_table = Table.read(args.sunspot_metadata, format="ascii.csv")

    print "Table has {} entries".format(len(obs_md_table))
    lookup_date, obs_md_table = find_ephemeris_lookup_date(obs_md_table['TAI-BEG'], obs_md_table['TAI-END'], obs_md_table)
    print "Successfully got {} ephemeris date entries".format(len(lookup_date))
    ephem_date_col = Column(lookup_date, name="EPHEM_DATE")
    obs_md_table.add_column(ephem_date_col)

    sunspot_count, sunspot_area = find_sunspot_data(ephem_date_col, sunspot_md_table)
    sunspot_count_col = Column(sunspot_count, name="SS_COUNT")
    sunspot_area_col = Column(sunspot_area, name="SS_AREA")
    obs_md_table.add_column(sunspot_count_col)
    obs_md_table.add_column(sunspot_area_col)

    galactic_core = ascoord.SkyCoord(l=0.0, b=0.0, unit='deg', frame='galactic')

    #Join lunar data to the table
    obs_md_table = join(obs_md_table, lunar_md_table['EPHEM_DATE', 'RA_APP', 'DEC_APP', 'MG_APP', 'ELV_APP'])

    lunar_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA_APP'], dec=obs_md_table['DEC_APP'], unit='deg', frame='icrs')
    boresight_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA'], dec=obs_md_table['DEC'], unit='deg', frame='fk5')
    lunar_seps = boresight_ra_dec.separation(lunar_ra_dec).degree
    obs_md_table.add_column(Column(lunar_seps, dtype=float, name="LUNAR_SEP"))

    obs_md_table.rename_column("MG_APP", "LUNAR_MAGNITUDE")
    obs_md_table.rename_column("ELV_APP", "LUNAR_ELV")
    obs_md_table.remove_columns(['RA_APP', 'DEC_APP'])

    #Join solar data to the table
    obs_md_table = join(obs_md_table, solar_md_table['EPHEM_DATE', 'RA_APP', 'DEC_APP', 'ELV_APP'])
    solar_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA_APP'], dec=obs_md_table['DEC_APP'], unit='deg', frame='icrs')
    boresight_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA'], dec=obs_md_table['DEC'], unit='deg', frame='fk5')
    solar_seps = boresight_ra_dec.separation(solar_ra_dec).degree
    obs_md_table.add_column(Column(solar_seps, dtype=float, name="SOLAR_SEP"))

    obs_md_table.rename_column("ELV_APP", "SOLAR_ELV")
    obs_md_table.remove_columns(['RA_APP', 'DEC_APP'])

    #Add in galactic data
    boresight_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA'], dec=obs_md_table['DEC'], unit='deg', frame='fk5')
    obs_md_table.add_column(Column(boresight_ra_dec.separation(galactic_core).degree,
                                dtype=float, name="GALACTIC_CORE_SEP"))
    boresight_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA'], dec=obs_md_table['DEC'], unit='deg', frame='fk5')
    obs_md_table.add_column(Column(boresight_ra_dec.transform_to('galactic').b.degree,
                                dtype=float, name="GALACTIC_PLANE_SEP"))
    #print obs_md_table
    if args.output == 'CSV':
        obs_md_table.write("annotated_metadata.csv", format="ascii.csv")
    elif args.output == 'FITS':
        obs_md_table.write("annotated_metadata.fits", format="fits")

if __name__ == '__main__':
    main()
