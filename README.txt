These utilities are meant to be run in an order (exmaple below), and depend on
manually downloaded external data files (the testing set, which covers the start
of SDSS DR12 to July 1, 2014, is available in ./source_data.)

1.)  observation_metadata.py
    This is the util that outputs metadata for plates-exposures.  It has two
modes, "--no_gather" which simply dumps metadata to stdout 'regular' (unflagged)
which saves data to either a FITS or CSV file named "observation_metadata".  It
obtains the metadata from the FIBER 1 spec file, not from plate/frame/etc.
files.
    ** see "python observation_metadata.py --help" for various badly named
options.
    **  Issues/Notes:  The output filename is hardcoded.  The metadata extracted
is hardcoded.
    **  Input:  A file of format output by bossquery, e.g.:
        bossquery --what "PLATE,MJD" --where "PLATE<=4500 and FIBER=1" --save plate_mjd_to_4500.dat
    **  Output:  A file of either CSV or FITS format with per-exposure metadata.

2.)  annotated_obs_metadata.py
    This takes the file output of observation_metadata.py and annotates it with
solar and lunar ephemeris data, as well as sunspot data.  The solar and lunar
ephemeris data are results of parse_ephemeris.py (see below), and the sunspot
data of parse_solar_activity.py (see below, again.)  Each plate-exposure
line/record is matched to the closest ephemeris time (which are in 15 minute
blocks), and the closest sunspot time (which are in day blocks) and the combined
data is output to a "annotated_metadata" file (either FITS or CSV.)
    ** see "python annotate_obs_metadata.py --help" for various badly named
options.
    **  Issues/Notes:  The output filename is hardcoded.  Assumptions about the
annotation data (e.g. ephemeris block size) are hardcoded.  Input and output
file formats must be the same, for no good reason.
    **  Input:  FITS or CSV output from observation_metadata.py.
    **  Output:  A file of either CSV or FITS format with annotated per-exposure
metadata.

3.)  parse_ephemeris.py
    Takes the one or more files from the Horizons JPL service
(http://ssd.jpl.nasa.gov/horizons.cgi) and converts them into a single CSV
formatted file for input into the annotate_obs_metadata.py util.  Note that this
util is used to process both solar and lunar ephemeris data, but those should be
processed separately.  A long header comment in parse_ephermeris.py gives fuller
details, but the Horizon tool should be used with the following settings:

Common options:
    Ephemeris Type:  Observer
    Observer Location:  Apache Point-Sloan Digital Sky Survey [645] ( 254 10' 45.9'' E, 32 46' 49.8'' N, 2791.2 m )
    Step: 15 m (minutes)

Lunar Table Settings options:
    1-5,8-10,20,23,24,33

Solar Table Settings options:
    1-5,9,20,23,24,33

The server imposes a limit on output size; if you wish to pull additional data
you will need to do so in chunks (9 months or a year is fine.)  This is one
reason the util takes multiple files; it will glue them together; note, however,
NO ATTEMPT is made to filter out duplicates!  Note also that the Horizons output
has a lot of documentation inline with the file:  all the header and footer
text needs to be deleted, including the column labels (which Astropy will not
handle). RECOMMENDATION:  Just cut'n'paste the header from the existing files in
./source_data.

4.)  parse_solar_activity.py
    Util to take yearly, aggregated NOAA daily sunspot data (from
ftp://ftp.swpc.noaa.gov/pub/indicies/old_indices/) and parse into a single CSV
formatted file for input into the annotate_obs_metadata.py util.  (The files of
interest in the directory are YYYY_DSD.txt, one for each year; current year
data, if needed, is YYYYQN_DSD.txt, where 'Q' indicates quarter and N is 1-4.)
Like the Horizons files, these include header comments that need to be chopped
out (by hand); and again, you should just copy the header form the files in
./source_data.

5.)  stack.py
Takes a file containing fibers (PLATE, MJD, FIBER) and stacks fibers for each
plate/mjd, giving a weighted average, re-sampled output.  This is saved as either
FITS or csv format file.  Optionally also saves out a file of 'pins' (csv only);
these are the wavelengths of notable peaks in the spectra.  There are slight
differences in where peaks are positioned, even in calibrated data (<~ 1 A).
It would be useful to come up with a way to re-align spectra using e.g. atmospheric
emissions...

6.)  arrayize.py
This is an optional utility that takes all the stacked files (FITS or CSV) and
saves them in .pkl file.
