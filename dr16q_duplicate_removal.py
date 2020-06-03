"""
This script will reduce the superset of observations to the set of 
quasars (quasar-only catalog). Due to the varied sources for quasar records
in the superset (DR7Q, DR12Q, spAll), a unique identifier does not exist
across catalog sources for quasars. This script uses coordinate matching
with a radius of 0.5 arcsec, matching the catalog against itself.

The cutdown of the superset happens in two parts: 
  1) The superset of observations is reduced to the set of objects via 
     dupe_remove(). This outputs its own catalog.
  2) qso_cutdown() picks out only the objects that are quasars based on
     the already-populated column IS_QSO_FINAL.

Dependencies
----------
spherematch : this is a utility function in the pydl package.
git repository : https://github.com/bradlyke/utilities


Input file requirements
----------
The input file must be a FITS file using the same
  column names that appear in utilities/empty_cat_maker.supercat_maker()
  It must also have gone through the final quasar classifier first:
  this means that IS_QSO_FINAL must be populated.


Parameters
----------
input_file_name : :class:'str'
                  The file name/path of the FITS superset catalog


Output
----------
DR16Q_DupeRem_SuperCat_{date}.fits : the duplicate-removed catalog
DR16Q_QSOCat_{date}.fits : the quasar-only catalog


"""

from astropy.io import fits
import numpy as np
from pydl.pydlutils.spheregroup import spherematch
import progressBar_fancy as pbf
import sys
import time
import cat_tools as ct
import tmark
import empty_cat_maker as ecm

# This is the first function called. It will reduce the superset catalog from
# the set of all OBSERVATIONS down to the set of OBJECTS.
# Lines with 'tmark.tm' are terminal output text for user feedback on 
# runtimes.
def dupe_remove(infile):
    sd = ct.file_load(infile) #Load the file into a structured array
    tmark.tm('Starting Spherematch') 
    
    #This is the magic line that does the source matching based on coordinates.
    #It will find all matches within 0.5 arcsec (not just the closest). This
    #command will take a while, so you might want to get a cup of coffee.
    mat = spherematch(sd['RA'],sd['DEC'],sd['RA'],sd['DEC'],0.5/3600.0,maxmatch=0)

    tmark.tm('Removing Self matches')
    #Because we are matching the catalog to itself, we need to remove 
    #the self-matches for observations. The matching arrays aren't in a 
    #particular order of distance, but both match arrays are in the same
    #order, so we need to use the array index to find self-matches.
    selfmarr = np.zeros(len(mat[0]),dtype='i2')
    for i in range(len(mat[0])):
        if mat[0][i] == mat[1][i]:
            selfmarr[i] = 1

    #Now grab only the records that were NOT self-matches.
    w0 = np.where(selfmarr==0)[0]
    mat0,mat1 = mat[0][w0],mat[1][w0]

    #Now we remove duplicates. This initializes a mask array to keep
    #everything by default.
    rem_arr = np.zeros(0) #This will be the temporary array of records to remove
    fullmask_arr = np.zeros(len(sd),dtype='i2')
    fullmask_arr[:] = 1

    #This is the primary part of this script. It will flag records that are
    #to be removed by appending them to the rec_arr array intialized above.
    #It will select the 'primary' record based on the following:
    #    1) Pick the record with the most confident visual inspection (VI)
    #    2) If some or all have the same VI confidence (or none) select
    #       the record with the highest SN_MEDIAN_ALL (signal-to-noise)
    tmark.tm('Removing Duplicates')
    for i in range(len(mat0)):
        rectemp = mat0[i]
        #As it iterates through, it will identify records to remove, but
        #may identify the same record more than once. If the record was
        #already flagged for removal, just skip it and go to the next
        #record.
        if rectemp in rem_arr:
            continue

        #Find all instances of the record that this iteration is looking at
        w = np.where(mat0==rectemp)[0]
        rem_arr = np.append(rem_arr,rectemp) #Add the one we're looking at
        rem_arr = np.append(rem_arr,mat1[w]) #Add all of the other matches
        #Now we need to gather up all of the records that match THIS object only
        #We assume, at first, that we will remove ALL instances of this object
        recarr_temp = np.array(rectemp)
        recarr_temp = np.append(recarr_temp,mat1[w])

        #A previous version used a 'voting' metric to determine if an object
        #was a QSO when multiple observations may be classified differently.
        #This was not used in the final version, but was left here because
        #I liked it.
        '''
        ###############################################
        #This is the voting block. This can be removed if voting is bad
        if len(recarr_temp)>=4:
            qso_votes_arr = np.array(sd['IS_QSO_FINAL'][recarr_temp])
            qso_votes_arr = np.where(qso_votes_arr<1,0,1)
            qso_tot = np.sum(qso_votes_arr)
            qso_pct = qso_tot / len(qso_votes_arr)
            if qso_pct < 0.6:
                sd['IS_QSO_FINAL'][recarr_temp] = sd['IS_QSO_FINAL'][recarr_temp] * 10
        ###############################################
        '''

        #For the object we're looking at, find the VI confidence for all
        #observations to test if they are all the same.
        zc0 = np.all(sd['Z_CONF'][recarr_temp]==0)
        zc1 = np.all(sd['Z_CONF'][recarr_temp]==1)
        zc2 = np.all(sd['Z_CONF'][recarr_temp]==2)
        zc3 = np.all(sd['Z_CONF'][recarr_temp]==3)
        if (zc0|zc1|zc2|zc3):
            #If they all have the same VI confidence we need to pick the
            #record with the highest SN_MEDIAN_ALL
            snamax = np.argmax(sd['SN_MEDIAN_ALL'][recarr_temp])
            #Since we assumed we were removing all instances, put back the
            #record we have chosen to keep (by removing it from the removal list).
            duperec = np.delete(recarr_temp,snamax)

            #Since we are 'removing' the duplicate observation records, we want
            #to keep the Plate, MJD, and FiberID for those we removed, and store
            #that information in the primary record's _DUPLICATE fields.
            #This makes sure it doesn't overwrite the first _DUPLICATE array 
            #position, but fills them in order.
            #First, check if the first position has already been filled.
            if sd['PLATE_DUPLICATE'][recarr_temp[snamax],0] != 0:
                first_filled = 1
            else:
                first_filled = 0
            #Fill the other positions in _DUPLICATE depending on whether
            #the first position was already filled or not.
            for j in range(len(duperec)):
                if first_filled == 1:
                    h = j + 1
                else:
                    h = j
                sd['PLATE_DUPLICATE'][recarr_temp[snamax],h] = sd['PLATE'][duperec[j]]
                sd['MJD_DUPLICATE'][recarr_temp[snamax],h] = sd['MJD'][duperec[j]]
                sd['FIBERID_DUPLICATE'][recarr_temp[snamax],h] = sd['FIBERID'][duperec[j]]
                #We also want to track if the duplicate observations were 
                #taken with the SDSS (1) or BOSS (2) spectrographic instrument
                #and we want to track how many of each we have (NSPEC_)
                if sd['PLATE'][duperec[j]] < 3500:
                    sd['NSPEC_SDSS'][recarr_temp[snamax]]+=1
                    sd['SPECTRO_DUPLICATE'][recarr_temp[snamax],h] = 1
                else:
                    sd['NSPEC_BOSS'][recarr_temp[snamax]]+=1
                    sd['SPECTRO_DUPLICATE'][recarr_temp[snamax],h] = 2

                fullmask_arr[duperec[j]] = 0 #Mark the duplicate records for removal
            #We also want the total number of duplicate observations
            sd['NSPEC'][recarr_temp[snamax]] = sd['NSPEC_BOSS'][recarr_temp[snamax]] + sd['NSPEC_SDSS'][recarr_temp[snamax]]
        
        #Here, if the VI confidence values were NOT all the same, pick the record
        #with the highest confidence. The rest, here, is marking the duplicates
        #the same way we did before.
        else:
            zcmax = np.argmax(sd['Z_CONF'][recarr_temp])
            duperec = np.delete(recarr_temp,zcmax)
            if sd['PLATE_DUPLICATE'][recarr_temp[zcmax],0] != 0:
                first_filled = 1
            else:
                first_filled = 0
            for j in range(len(duperec)):
                if first_filled == 1:
                    h = j+1
                else:
                    h = j
                sd['PLATE_DUPLICATE'][recarr_temp[zcmax],h] = sd['PLATE'][duperec[j]]
                sd['MJD_DUPLICATE'][recarr_temp[zcmax],h] = sd['MJD'][duperec[j]]
                sd['FIBERID_DUPLICATE'][recarr_temp[zcmax],h] = sd['FIBERID'][duperec[j]]
                if sd['PLATE'][duperec[j]] < 3500:
                    sd['NSPEC_SDSS'][recarr_temp[zcmax]]+=1
                    sd['SPECTRO_DUPLICATE'][recarr_temp[zcmax],h] = 1
                else:
                    sd['NSPEC_BOSS'][recarr_temp[zcmax]]+=1
                    sd['SPECTRO_DUPLICATE'][recarr_temp[zcmax],h] = 2

                fullmask_arr[duperec[j]] = 0
            sd['NSPEC'][recarr_temp[zcmax]] = sd['NSPEC_BOSS'][recarr_temp[zcmax]] + sd['NSPEC_SDSS'][recarr_temp[zcmax]]
        pbf.pbar(i,len(mat0)) #A progress bar to track where the script is at now.

    #Now that we know which ones are the primary records, we need to clean up
    #the initialized values in NSPEC and _DUPLICATE fields. Doing this for all
    #records took almost the same amount of time as only doing it for records
    #we keep, so it just iterates through all of them as a safety measure.
    tmark.tm('Cleaning up 0s')
    for i in range(len(sd)):
        #The NSPEC field in the superset was initialized with a -1
        #The _DUPLICATE fields were initialized with 0 values (-1 here already
        #meant something else, so initialized with 0).
        if sd['NSPEC'][i] == -1:
            sd['NSPEC'][i] = 0 #Change unique objects to have 0 duplicates
            sd['PLATE_DUPLICATE'][i,:] = -1 #And change all _DUPLICATE values to -1
            sd['MJD_DUPLICATE'][i,:] = -1
            sd['FIBERID_DUPLICATE'][i,:] = -1
            sd['SPECTRO_DUPLICATE'][i,:] = -1
            continue
        #If it is not a unique object, change unfilled 0 values in _DUPLCATE to -1
        num_dupes = sd['NSPEC'][i]
        sd['PLATE_DUPLICATE'][i,num_dupes:] = -1
        sd['MJD_DUPLICATE'][i,num_dupes:] = -1
        sd['FIBERID_DUPLICATE'][i,num_dupes:] = -1
        sd['SPECTRO_DUPLICATE'][i,num_dupes:] = -1

    #And write out the catalog with only the records to keep (fullmask_arr == 1)
    tmark.tm('Writing Catalog')
    wkeep = np.where(fullmask_arr==1)[0]
    prim_hdr = fits.open(infile)[0].header
    prim_hdu = fits.PrimaryHDU(header=prim_hdr)

    data_hdu = fits.BinTableHDU.from_columns(sd[wkeep],name='CATALOG')
    data_out = fits.HDUList([prim_hdu,data_hdu])

    #Since this was script was iterated, make sure the script isn't trying
    #to overwrite an older version of the file by appending the output date.
    odtag = time.strftime('%Y%m%d')
    outfile_name = '../data/DR16Q_DupeRem_SuperCat_{}.fits'.format(odtag)
    data_out.writeto(outfile_name)
    tmark.tm('DupeRem catalog written')
    return outfile_name #Return the output name to feed to the next function.

#This function performs the second part of the overall script - reducing the 
#catalog of OBJECTS down to a catalog of QUASARS.
def qso_cutdown(infile):
    tmark.tm('Starting QSO Cutdown')
    supercat = fits.open(infile)[1].data #Open the object catalog as a FITS record
    wqso = np.where(supercat['IS_QSO_FINAL']>0)[0]
    #The quasar-only catalog will have many more columns of data than the superset
    #so we will create these new columns as empty.
    #The original supercat_maker() initialized the _DUPLICATE fields to have 99
    #elements (it was unknown what the maximum number of duplicates was). Looking
    #At the output from the dupe_remove() function, we found that the maximum
    #number of duplicates was 74. This was hardcoded back into the supercat_maker()
    #and multiwave_maker() functions to reduce the file sizes as much as possible.
    qsocat = ecm.multiwave_maker(len(wqso))
    tmark.tm('Creating QSO-only struct')
    for cname in supercat.columns.names:
        #The supercat file has one column NOT in the qsocat file, PRIM_REC
        #All records in the quasar-only catalog ARE primary records already
        #But users looking at the superset catalog may want to know which
        #records were considered the primary records. Skip that column.
        if cname == 'PRIM_REC':
            continue
        else:
            qsocat[cname] = supercat[cname][wqso]

    #And write out the quasar-only catalog.
    tmark.tm('Writing QSO-only catalog')
    prim_hdr = fits.open(infile)[0].header
    prim_hdu = fits.PrimaryHDU(header=prim_hdr)

    data_hdu = fits.BinTableHDU.from_columns(qsocat,name='CATALOG')
    data_out = fits.HDUList([prim_hdu,data_hdu])
    odtag = time.strftime('%Y%m%d')
    outfile_name = '../data/DR16Q_QSOCat_{}.fits'.format(odtag)
    data_out.writeto(outfile_name)
    tmark.tm('QSO-only catalog Written')
    return outfile_name

#This is for calling the script from the command line.
#  python dr16q_duplicate_removal.py <filename>
#  and wait.
if __name__=='__main__':
    #Give the name of the input file that is in the data folder.
    input_file_name = sys.argv[1]
    input_file = '../data/{}'.format(input_file_name)
    tmark.tm('Starting script')
    #Call duplicate removal, then quasar cutdown.
    duperem_name = dupe_remove(input_file)
    qsoname = qso_cutdown(duperem_name)
