"""
This script will apply the QuasarNET conditions for reclassifying objects
flagged for visual inspection back to quasars (where we can trust the 
QuasarNET classification). Short version: for records flagged as 'VI',
if QuasarNET has a classification of IS_QSO==1 AND Z_QN<2.0 it's a quasar.
Otherwise, leave the flag for VI. Once reclassified, this will output
the next step in the proto-superset catalog AND a catalog of VI-flagged
records.


Dependencies
----------
QuasarNET catalog file in the data folder
git repository : https://github.com/bradlyke/utilities


Input file requirements
----------
Input file must be a FITS file in the form output by spAll_class_full.py


Parameters
----------
input_file_name : :class:'str'
                  The name of the Superset file
qnet_file_name : :class:'str'
                  The name of the QuasarNET catalog file


Output
----------
Two catalog files
1) Reclassified superset file (AUTOCLASS_PQN now populated)
2) Subcatalog of objects flagged for visual inspection

"""

from astropy.io import fits
import numpy as np
import cat_tools as ct
import sys
import time

def qnet_burn(infile,qnet_file):
    #First we need to open the files for processing
    #Since we are changing the values in a column of strings in the preliminary
    #superset file, we have to load that file (drfile) as a structured array
    #or the updated values won't be saved on file write.
    qfile = fits.open(qnet_file)[1].data
    drfile = ct.file_load(infile)
    num_dr = len(drfile)
    num_qnet = len(qfile)

    #Match the two catalogs on a Plate-MJD-FiberID hash string
    drhash,qdhash = ct.mk_hash(drfile),ct.mk_hash(qfile)
    drargs,qdargs = ct.rec_match_srt(drhash,qdhash)

    #Copy the AUTOCLASS_DR14Q classifications to AUTOCLASS_PQN
    drfile['AUTOCLASS_PQN'] = drfile['AUTOCLASS_DR14Q']

    #Find the safe VI flagged records that need to be changed back to quasars.
    wqso = np.where((drfile['AUTOCLASS_PQN'][drargs]=='VI')&(qfile['IS_QSO'][qdargs]==1)&(qfile['ZBEST'][qdargs]<2.0))[0]
    drfile['AUTOCLASS_PQN'][drargs[wqso]] = 'QSO'

    #Find the VI flagged records left over that are post-DR15
    wvi = np.where((drfile['AUTOCLASS_PQN']=='VI')&(drfile['MJD']>57905))[0]

    #Create the header HDU for both files from the proto-superset file
    prim_hdr = fits.open(infile)[0].header
    prim_hdu = fits.PrimaryHDU(header=prim_hdr)
    
    #Create the data HDU for the next step in the proto-superset catalog
    super_hdu = fits.BinTableHDU.from_columns(drfile,name='CATALOG')
    super_hdulist = fits.HDUList([prim_hdu,super_hdu])
    
    #Create the VI-flagged catalog
    vi_data = np.array(drfile[wvi])
    vi_hdu = fits.BinTableHDU.from_columns(vi_data,name='CATALOG')
    vi_hdulist = fits.HDUList([prim_hdu,vi_hdu])

    #And write out both catalogs with a date tag appended to the file name.
    ofile_dtag = time.strftime('%Y%m%d')
    super_outname = '../data/DR16Q_vdb_rem_PQN_{}.fits'.format(ofile_dtag)
    vi_outname = '../data/DR16Q_VICAT_{}.fits'.format(ofile_dtag)
    
    super_hdulist.writeto(super_outname)
    vi_hdulist.writeto(vi_outname)

    return super_outname,vi_outname


if __name__=='__main__':
    input_file_name = sys.argv[1] #Should be something like DR16Q_vdb_rem_{DATE}.fits
    qnet_file_name = sys.argv[2]
    input_file = '../data/{}'.format(input_file_name)
    qnet_file = '../data/{}'.format(qnet_file_name)
    full_cat_name, vi_cat_name = qnet_burn(input_file,qnet_file)
