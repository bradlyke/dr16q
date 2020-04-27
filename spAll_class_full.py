"""
This program will generate the DR16Q Superset file and DR16Q quasar-only
file from the spAll-v5_13_0.fits file. This input file is ~16Gb, so be careful.
Because the files in use are large, and some functions can take a long time
each function will output its own completed file, which the next
function will load in turn. Have some HDD space free. A lot.

"""

from astropy.io import fits
import numpy as np
import sys
import os
import time
from pydl.pydlutils.sdss import sdss_flagval as flagval
import sdss_name_v2 as sdn
import cat_tools as ct
import progressBar as pb
import empty_cat_maker as ecm

#This function will select only the object records that were targeted as quasars
#within the spAll file. This will only include, by its nature, BOSS and eBOSS
#observations.
def cat_cut():
    #BOSS_TARGET1 bit flag names
    bt1_flag_names = np.array(['QSO_CORE','QSO_BONUS','QSO_KNOWN_MIDZ','QSO_KNOWN_LOHIZ',
                    'QSO_NN','QSO_UKIDSS','QSO_KDE_COADD','QSO_LIKE','QSO_FIRST_BOSS',
                    'QSO_KDE','QSO_CORE_MAIN','QSO_BONUS_MAIN','QSO_CORE_ED',
                    'QSO_CORE_LIKE','QSO_KNOWN_SUPPZ'])
    #EBOSS_TARGET0 bit flag names
    et0_flag_names = np.array(['QSO_EBOSS_CORE','QSO_PTF','QSO_REOBS','QSO_EBOSS_KDE',
                    'QSO_EBOSS_FIRST','QSO_BAD_BOSS','QSO_BOSS_TARGET','QSO_SDSS_TARGET',
                    'QSO_KNOWN','SPIDERS_RASS_AGN','SPIDERS_ERASS_AGN','TDSS_A',
                    'TDSS_FES_DE','TDSS_FES_NQHISN','TDSS_FES_MGII','TDSS_FES_VARBAL',
                    'SEQUELS_PTF_VARIABLE'])
    #EBOSS_TARGET1 bit flag names
    et1_flag_names = np.array(['QSO1_VAR_S82','QSO1_EBOSS_CORE','QSO1_PTF','QSO1_REOBS',
                    'QSO1_EBOSS_KDE','QSO1_EBOSS_FIRST','QSO1_BAD_BOSS','QSO_BOSS_TARGET',
                    'QSO_SDSS_TARGET','QSO_KNOWN','TDSS_TARGET','SPIDERS_TARGET','S82X_TILE1',
                    'S82X_TILE2','S82X_TILE3'])
    #EBOSS_TARGET2 bit flag names
    et2_flag_names = np.array(['SPIDERS_RASS_AGN','SPIDERS_ERASS_AGN','SPIDERS_XMMSL_AGN',
                    'TDSS_A','TDSS_FES_DE','TDSS_FES_NQHISN','TDSS_FES_MGII','TDSS_FES_VARBAL',
                    'TDSS_B','TDSS_FES_HYPQSO','TDSS_CP','S82X_BRIGHT_TARGET',
                    'S82X_XMM_TARGET','S82X_WISE_TARGET','S82X_SACLAY_VAR_TARGET',
                    'S82X_SACLAY_BDT_TARGET','S82X_SACLAY_HIZ_TARGET',
                    'S82X_RICHARDS15_PHOTOQSO_TARGET','S82X_PETERS15_COLORVAR_TARGET',
                    'S82X_LSSTZ4_TARGET','S82X_UNWISE_TARGET','S82X_GTRADMZ4_TARGET',
                    'S82X_CLAGN1_TARGET','S82X_CLAGN2_TARGET'])
    #ANCILLARY_TARGET1 bit flag names
    at1_flag_names = np.array(['BLAZGVAR','BLAZR','BLAZXR','BLAZXRSAM','BLAZXRVAR',
                    'XMMBRIGHT','XMMGRIZ','XMMHR','XMMRED','FBQSBAL','LBQSBAL','ODDBAL',
                    'OTBAL','PREVBAL','VARBAL','QSO_AAL','QSO_AALS','QSO_IAL','QSO_RADIO',
                    'QSO_RADIO_AAL','QSO_RADIO_IAL','QSO_NOAALS','QSO_GRI','QSO_HIZ',
                    'QSO_RIZ','BLAZGRFLAT','BLAZGRQSO','BLAZGX','BLAZGXQSO','BLAZGXR',
                    'CXOBRIGHT','CXORED'])
    #ANCILLARY_TARGET2 bit flag names
    at2_flag_names = np.array(['HIZQSO82','HIZQSOIR','KQSO_BOSS','QSO_VAR','QSO_VAR_FPG',
                    'RADIO_2LOBE_QSO','QSO_SUPPZ','QSO_VAR_SDSS','QSO_WISE_SUPP',
                    'QSO_WISE_FULL_SKY','DISKEMITTER_REPEAT','WISE_BOSS_QSO',
                    'QSO_XD_KDE_PAIR','TDSS_PILOT','SPIDERS_PILOT','TDSS_SPIDERS_PILOT',
                    'QSO_VAR_LF','QSO_EBOSS_W3_ADM','XMM_PRIME','XMM_SECOND',
                    'SEQUELS_TARGET','RM_TILE1','RM_TILE2','QSO_DEEP'])

    #Turn the bit flag name arrays into a bit mask value
    bt1 = flagval('BOSS_TARGET1',bt1_flag_names)
    et0 = flagval('EBOSS_TARGET0',et0_flag_names)
    et1 = flagval('EBOSS_TARGET1',et1_flag_names)
    et2 = flagval('EBOSS_TARGET2',et2_flag_names)
    at1 = flagval('ANCILLARY_TARGET1',at1_flag_names)
    at2 = flagval('ANCILLARY_TARGET2',at2_flag_names)

    loc = os.environ.get('BOSS_SPECTRO_REDUX') #Get the location for the spAll file.
    vnum = 'v5_13_0'
    full_file = '{0}/{1}/spAll-{2}.fits'.format(loc,vnum,vnum)
    spfull = fits.open(full_file)[1].data #Load the whole 16Gb file.
    
    #Cut down to only the records targeted as quasars using bitmasks.
    wq = np.where((spfull['BOSS_TARGET1']&bt1)|(spfull['EBOSS_TARGET0']&et0)|
            (spfull['EBOSS_TARGET1']&et1)|(spfull['EBOSS_TARGET2']&et2)|
            (spfull['ANCILLARY_TARGET1']&at1)|(spfull['ANCILLARY_TARGET2']&at2))[0]

    #If the user wants to track which version of the spAll file was used, this
    #will record that file version in the Primary (HDU0) header.
    fname = 'spAll-{}'.format(vnum)
    prim_hdr = fits.Header()
    prim_hdr['DATA_REL']=(fname,'spAll file used')
    prim_hdu = fits.PrimaryHDU(header=prim_hdr)

    #Output the cutdown file of only those records targeted as quasars.
    data_out = np.array(spfull[wq])
    data_hdu = fits.BinTableHDU.from_columns(data_out)
    data_of = fits.HDUList([prim_hdu,data_hdu])
    file_dtag = time.strftime('%Y%m%d')
    outfile_name = 'DR16Q_bitcut_{}'.format(file_dtag)
    #FET means FITS (writer) ERROR TRAP. This will force the file to be written
    #just in case a file already exists with that file name.
    superset_name = ct.fet(data_of,outfile_name)

    return superset_name #To load this output in the next function, return output name.

#################################CLASSIFICATION#################################

################################################################################
#                                                                              #
#  SCRIPT: VI Catalogue Flagging Algorithm                                     #
#  AUTHOR: Brad Lyke, with base algorithm design by Isabelle Paris             #
#                                                                              #
#  PURPOSE: This script will analyze all objects from the latest spAll file    #
#           from SDSS and generate a DRQ flag for visual inspection followup.  #
#           The algorithm pseudocode is as follows:                            #
#           1)If the object 'CLASS' is 'STAR', then label as 'STAR'            #
#           2)If the object 'CLASS' is 'GALAXY' and 'Z' < 1, then 'GALAXY'     #
#           3)If the 'CLASS' is 'GALAXY' and 'Z' >= 1, check spZall next top 4 #
#             model fits and decide:                                           #
#             --If 1+ are 'GALAXY' then 'GALAXY'                               #
#           4)If the 'CLASS' is 'QSO', check spZall next top 4 fits and decide:#
#             --If 2+ are 'STAR' then 'STAR',                                  #
#             --Else If < 2 are 'STAR', ZWARNING=0 then 'QSO'                  #
#             --Else then 'QFLAG'                                              #
#           If, after these, the object has no label (or QFLAG which is an     #
#           internal variable for efficiency), then it will need VI followup.  #
#                                                                              #
#  VARIABLES: sp_cat - The name of the current spAll file to use, passed as a  #
#                      system argument.                                        #
#                                                                              #
#  INPUT: This file expects the input catalogue to only be items from the      #
#           spAll file that have OBJTYPE 'QSO'. Do not use full spAll file.    #
#         Runs from terminal. Command as follows                               #
#            python spAll_class_full.py [spAll filename]                       #
#            eg. python spAll_class_full.py spAll-v5_13_0.fits                 #
#  OUTPUT: -A new fits file with object classifications in AUTOCLASS_DR14Q.    #
#          -Prints to the terminal user-friendly feedback about progress on    #
#           number of objects matched. Runtimes not tracked as this will be a  #
#           run-once script for the newest spAll catalogue.                    #
#                                                                              #
# Note: REQUIRES PYTHON 3.5 or later.                                          #
################################################################################

################################################################################
#                                                                              #
#  FUNCTION: GALAXY CONFIRMATION                                               #
#  PURPOSE: This is the function that chooses a flag for a GALAXY that has a   #
#           Z >= 1. This is step 3 in the pseudocode above.                    #
#                                                                              #
#  ACCEPTS: fil  - File name of spZall file to check.                          #
#           p_in - The 'PLATE' value of the object currently being checked.    #
#           m_in - The 'MJD' value of the object currently being checked.      #
#           f_in - The 'FIBERID' value of the object currently being checked.  #
#  RETURNS: out_Gflag - The classification flag for the object, saved to the   #
#             mask_arr that holds all classifications for the file.            #
#  INPUTS:  None                                                               #
#  OUTPUTS: None                                                               #
#  CALLED BY: obj_class() in same file.                                        #
#                                                                              #
################################################################################
def gal3_confirm(fil,p_in,m_in,f_in):
    #OBJ_CLASS() for loop iterates through FIBERIDs for the objects in a file.
    #This function, then, only needs to check the current file for the matching
    #FIBERID. PLATE and MJD are not currently used, but can be in case of a
    #larger or combined spZall file that holds multiple plate-mjd combos.
    wg = np.where(fil['FIBERID'] == f_in)[0]

    #Check the 'CLASS' fields for the next top 4 fits and count them.
    t_gal,t_galC = np.unique(fil['CLASS'][wg[1:5]],return_counts=True)
    tGw = np.where(t_gal == 'GALAXY')[0]

    #The next 2 IF/ELSE statements are error-trapping in case one of the
    #counts in t_galC is 0. If it is, then it doesn't show up in t_gal, so
    #the np.where turns up empty (not 0). These manually force the value to
    #zero in such a case.
    if len(tGw) > 0:
        tG_check = t_galC[tGw][0]
    else:
        tG_check = 0

    #These now assign the classification flag based on the pseudocode in the
    #header above. If the number of galaxies is >= 1, mark it galaxy.
    #Otherwise leve it blank for future visual inspection.
    if tG_check >= 1:
        out_Gflag = 'GALAXY'
    else:
        out_Gflag = ''

    #Return the classification flag to obj_class()
    return out_Gflag

################################################################################
#                                                                              #
#  FUNCTION: QSO CONFIRMATION                                                  #
#  PURPOSE: This is the function that chooses a flag for QSOs from             #
#             step 4 in the pseudocode above.                                  #
#                                                                              #
#  ACCEPTS: fil   - File name of spZall file to check.                         #
#           p_in  - The 'PLATE' value of the object currently being checked.   #
#           m_in  - The 'MJD' value of the object currently being checked.     #
#           f_in  - The 'FIBERID' value of the object currently being checked. #
#           zw_in - The 'ZWARNING' value of the object currently being checked.#
#  RETURNS: out_Qflag - The classification flag for the object, saved to the   #
#             mask_arr that holds all classifications for the catalogue        #
#  INPUTS:  None                                                               #
#  OUTPUTS: None                                                               #
#  CALLED BY: obj_class() in same file.                                        #
#                                                                              #
################################################################################
def qso_confirm(fil,p_in,m_in,f_in,zw_in):
    #OBJ_CLASS() for loop iterates through FIBERIDs for the objects in a file.
    #This function, then, only needs to check the current file for the matching
    #FIBERID. PLATE and MJD are not currently used, but can be in case of a
    #larger or combined spZall file that holds multiple plate-mjd combos.
    wq = np.where(fil['FIBERID'] == f_in)[0]

    #Check the 'CLASS' field of the next top 4 fits. Find 'STAR', 'GALAXY',
    #and 'QSO'
    t_QSO,t_QSOC = np.unique(fil['CLASS'][wq[1:5]],return_counts=True)
    tQSw = np.where(t_QSO == 'STAR')[0]
    tQQw = np.where(t_QSO == 'QSO')[0]

    #The next 2 IF/ELSE statements are error-trapping in case one of the
    #counts in t_QSOC is 0. If it is, then it doesn't show up in the t_QSO array
    #so the np.where turns up empty (not 0). These manually force the value to
    #zero in such a case.
    if len(tQSw) > 0:
        tQS_check = t_QSOC[tQSw][0]
    else:
        tQS_check = 0
    if len(tQQw) > 0:
        tQQ_check = t_QSOC[tQQw][0]
    else:
        tQQ_check = 0

    #These now assign the classification flag based on the pseudocode in the
    #header above. The number of STARS is checked FIRST, then the number of stars 
    #is compared along with ZWARNING and the number of other QSOs. 
    #Failing these checks a QFLAG classification is assigned (meaning VI later).
    if tQS_check >= 2:
        out_Qflag = 'STAR'
    elif ((tQS_check < 2) & (zw_in == 0)):
        out_Qflag = 'QSO'
    else:
        out_Qflag = 'QFLAG'

    #Send the classification flag back to obj_class() loop that called it.
    return out_Qflag

################################################################################
#                                                                              #
#  FUNCTION: OBJECT CLASSIFICATION MAIN                                        #
#  PURPOSE: This is the function that actually iterates through all of the     #
#           objects in the cutdown catalog file input by the user. It will call#
#           gal3_confirm() and qso_confirm() in different for loops.           #
#                                                                              #
#  ACCEPTS: ifile - The name of the input cutdown catalog file, from cat_cut() #
#  RETURNS: None                                                               #
#  INPUTS:  None                                                               #
#  OUTPUTS: --A new FITS catalog. The spAll file is recreated with a new       #
#             column up front called 'AUTOCLASS_DR14Q' which holds the         #
#             classification flags from mask_arr.                              #
#           --User-friendly text output about current progress.                #
#  OUTPUTS TO: FITS file: Folder where the script was called from.             #
#              TEXT : Screen output.                                           #
#  CALLED BY: None                                                             #
#                                                                              #
################################################################################
def obj_class(ifile):
    #Open the cutdown catalog file and find the total number of objects.
    data = fits.open(ifile)[1].data
    dnum = len(data)

    #Counts the number of objects by type at the beginning for feedback and
    #benchmark purposes. Outputs this to the screen.
    nStar = len(np.where(data['CLASS'] == 'STAR')[0])
    nGal = len(np.where(data['CLASS'] == 'GALAXY')[0])
    nQSO = len(np.where(data['CLASS'] == 'QSO')[0])
    print('\n')
    print('----------------------------------------')
    print('Number of Starting Objects  : '+str(dnum))
    print('Number of Stars before      : '+str(nStar))
    print('Number of Galaxies before   : '+str(nGal))
    print('Number of QSOs before       : '+str(nQSO))
    print('----------------------------------------')
    print('\n')

    #This is a placeholder array to hold local flags based on the algorithm. It
    #will become a new column in the output FITS file called 'AUTOCLASS_DR14Q'.
    mask_arr = np.chararray(dnum,itemsize=6,unicode=True)

    #First Classification | STEP 1 in pseudocode.
    wStar = np.where(data['CLASS'] == 'STAR')[0]
    mask_arr[wStar] = 'STAR'

    #Second Classification | STEP 2 in pseudocode.
    wGal = np.where((data['CLASS'] == 'GALAXY')&(data['Z'] < 1))[0]
    mask_arr[wGal] = 'GALAXY'

    #spZall Classifications
    #Classifications 3 and 4 require the spZall files to be checked. To minimize
    #the number of times a file is loaded into memory, a list of necessary files
    #for the GALAXY classification (3) is made and downloaded in turn. While 
    #these files are present, the QSO objects are also checked so that the files 
    #won't need to be loaded again later.
    #GALAXY-CLASS objects
    wG2 = np.where((data['CLASS'] == 'GALAXY')&(data['Z'] >= 1))[0]
    count = len(wG2)

    #QSO-CLASS objects
    wQSO = np.where(data['CLASS'] == 'QSO')[0]
    countQ = len(wQSO)

    #Objects are identified by a unique combination of plate number, mjd, and
    #fiberid. This generates a list of single values for each object, so unique
    #combinations can be found. This allows the use of np.where() calls instead
    #of nested for loops later.
    pm = (data['PLATE']**2) + (data['MJD'] - 1)

    #This finds the unique combinations of plate and mjd which define the spZall
    #files. In each file each fiberid will have a number of pipeline fits.
    #These are the unique plate/mjd combos for GALAXIES.
    uniq,uniqindices = np.unique(pm[wG2], return_index=True)
    ugplates = data['PLATE'][wG2[uniqindices]]
    umjd = data['MJD'][wG2[uniqindices]]

    #This does the same as above, but for QSOs. Here this is used not to find
    #the files for loading, but to identify which objects will appear in the
    #files that are loaded for the GALAXY-3 classification.
    uQ,uQind = np.unique(pm[wQSO],return_index=True)
    uqplates = data['PLATE'][wQSO[uQind]]
    uqmjd = data['MJD'][wQSO[uQind]]

    #Makes a list of file names. This is passed to the for loop later.
    file_list = np.chararray(len(ugplates),itemsize=23,unicode=True)
    for j in range(len(ugplates)):
        file_list[j] = 'spZall-{}-{}.fits'.format(ugplates[j],umjd[j])

    #A counter used by the for loop.
    file_count = len(file_list)

    #Initializing the plate, mjd, fiber, run2d variables.
    plate = 0
    mjd = 0
    fiber = 0
    run2d = 0

    #Where are the spZall files stored
    uloc0 = os.environ.get('BOSS_SPECTRO_REDUX')

    #This loop will go through the files identified in file_list (from GAL-3
    #plate/mjd combos). It will iterate through all of the spZall files. While
    #doing so it outputs a single line about its progress that is overwritten.
    for i in range(file_count):
        #Define the PLATE and MJD we are working with on this iteration.
        plate = ugplates[i]
        mjd = umjd[i]
        run2d = data['RUN2D'][wG2[uniqindices[i]]]

        #Get the file name for the current spZall file.
        z_file = '{}/{}/{}/{}/{}'.format(uloc0,run2d,plate,run2d,file_list[i])

        #dall is the spZall file currently being checked.
        dall = fits.open(z_file)[1].data

        #Each object, identified by plate/mjd/fiber has 134 separate fits in the
        #spZall file. These will find only the fibers appearing in the spZall
        #file we are currently using.
        objs = np.where((data['PLATE'][wG2] == plate)&(data['MJD'][wG2] == mjd))[0]
        fibers,fiberind = np.unique(data['FIBERID'][wG2[objs]],return_index=True)

        #Each spZall file holds 1000 fibers. This iterates through the FIBERIDs
        #identified in the previous step, then passes the information to
        #the gal3_confirm() function above.
        for h in range(len(fibers)):
            fiber_t = fibers[h]
            mask_arr[wG2[objs[fiberind[h]]]] = gal3_confirm(dall,plate,mjd,fiber_t)

        #This does the same as above but for any QSOs that might be in the same
        #files.
        objsQ = np.where((data['PLATE'][wQSO] == plate)&(data['MJD'][wQSO] == mjd))[0]
        fibQ,fibQind = np.unique(data['FIBERID'][wQSO[objsQ]],return_index=True)

        #Again, this iterates through the FIBERIDs. ZWARNING is found first,
        #then everything is passed to the qso_confirm() function above.
        for k in range(len(fibQ)):
            fiber_q = fibQ[k]
            z_warn = data['ZWARNING'][wQSO[objsQ[fibQind[k]]]]
            mask_arr[wQSO[objsQ[fibQind[k]]]] = qso_confirm(dall,plate,mjd,fiber_q,z_warn)

        #This outputs the single line that is overwritten as files are completed
        st = '\rFile: {} | spZall Complete: {}/{}'.format(file_list[i],i+1,file_count)
        sys.stdout.write(st)
        sys.stdout.flush()

    #Leftover QSOs
    #This is for any classification 4 QSOs that weren't covered by the GAL-3
    #list of spZall files. It is the same as above, but only for QSOs.
    wlq = np.where((data['CLASS'] == 'QSO')&(mask_arr == ''))[0]
    ulq,ulqindex = np.unique(pm[wlq],return_index=True)
    ulplates = data['PLATE'][wlq[ulqindex]]
    ulmjd = data['MJD'][wlq[ulqindex]]

    #Creates a new list of spZall files, for the leftover QSOs.
    file_lq = np.chararray(len(ulplates),itemsize=23,unicode=True)
    for j in range(len(ulplates)):
        file_lq[j] = 'spZall-{}-{}.fits'.format(ulplates[j],ulmjd[j])

    file_lq_count = len(file_lq)

    #Reinitialize all of the object address information. Don't want old data
    #messing us up.
    plate = 0
    mjd = 0
    fiber = 0
    run2d = 0
    print('\n')
    print('----------------------------------------')
    print('LEFTOVER QSOs')
    print('----------------------------------------')

    for i in range(file_lq_count):
        plate = ulplates[i]
        mjd = ulmjd[i]
        run2d = data['RUN2D'][wlq[ulqindex[i]]]

        z_file = '{}/{}/{}/{}/{}'.format(uloc0,run2d,plate,run2d,file_lq[i])

        dall = fits.open(z_file)[1].data

        objsQ = np.where((data['PLATE'][wlq] == plate)&(data['MJD'][wlq] == mjd))[0]
        fibQ,fibQind = np.unique(data['FIBERID'][wlq[objsQ]],return_index=True)

        for k in range(len(fibQ)):
            fiber_q = fibQ[k]
            z_warn = data['ZWARNING'][wlq[objsQ[fibQind[k]]]]
            mask_arr[wlq[objsQ[fibQind[k]]]] = qso_confirm(dall,plate,mjd,fiber_q,z_warn)

        st1 = '\rFile: {} | spZall Complete: {}/{}'.format(file_lq[i],i+1,file_lq_count)
        sys.stdout.write(st1)
        sys.stdout.flush()

    #TESTING FEEDBACK
    #These lines are only for feedback before and after for quick reference on how
    #many objects need VI. We're hoping for no more than 7% neededing followup.
    wflagged = np.where((mask_arr == 'QFLAG') | (mask_arr == ''))[0]
    nflagged = len(wflagged)
    wt2 = np.where(mask_arr == 'STAR')[0]
    wt3 = np.where(mask_arr == 'GALAXY')[0]
    wt4 = np.where(mask_arr == 'QSO')[0]
    prct_vi = (float(nflagged) / float(dnum)) * 100

    print('\n')
    print('----------------------------------------')
    print('Number of Stars after      : '+str(len(wt2)))
    print('Number of Galaxies after   : '+str(len(wt3)))
    print('Number of QSOs after       : '+str(len(wt4)))
    print('----------------------------------------')
    print('Number to be Visually inspected : '+str(nflagged))
    print('Percent to be Visually inspected: {0:.2f}%'.format(prct_vi))
    print('\n')

    #After classification flags are completed, find any objects still left Unclassified
    #or that have a QFLAG (which is an unclassified QSO). Mark all objects for
    #visual inspection followup.
    mask_arr[wflagged] = 'VI'
    #We also want to visually inspect QSOs with pipeline Z > 3.5.
    wzu = np.where((mask_arr=='QSO')&(data['Z']>3.5))[0]
    mask_arr[wzu] = 'VI'

    min_mjd = np.amin(data['MJD'])
    max_mjd = np.amax(data['MJD'])
    mjd_range = '{0:05d}-{1:05d}'.format(min_mjd,max_mjd)
    spvname = 'spAll-v5_13_0'
    #Write out a fits file with this new classification column. Also modifies
    #the primary HDU header (extension 0) to keep information on what the Input
    #spAll file was, what DR this is for, and what program ran the damn thing.
    prim_hdrc = fits.Header()
    prim_hdrc['ALG_FILE']=('spAll_class_full.py','Algorithm file used')
    prim_hdrc['DATA_REL']=(spvname,'spAll file used')
    prim_hdrc['MJD_RNGE']=(mjd_range,'MJD Range of objects')
    prim_hduc = fits.PrimaryHDU(header=prim_hdrc)

    #Tack this new classification column onto the columns from the cutdown catalog.
    mask_col = fits.ColDefs([fits.Column(name='AUTOCLASS_DR14Q',format='6A',array=mask_arr)])
    data_cols = data.columns
    data_hduc = fits.BinTableHDU.from_columns(mask_col + data_cols)

    ofile_dtag = time.strftime('%Y%m%d')
    data_ofc = fits.HDUList([prim_hduc,data_hduc])
    out_file_namec = 'DR16Q_autoclass_{}'.format(ofile_dtag)
    classified_name = ct.fet(data_ofc,out_file_namec)

    return classified_name

#############################CATALOG COMBINE####################################

def cat_combine(inrec_name):
    
    #These are just the names of the columns that appear in the spAll files.
    spcolkeep = np.array(['RA','DEC','AUTOCLASS_DR14Q','THING_ID','PLATE','MJD','FIBERID',
                        'Z','ZWARNING','BOSS_TARGET1','EBOSS_TARGET0','EBOSS_TARGET1',
                        'EBOSS_TARGET2','ANCILLARY_TARGET1','ANCILLARY_TARGET2',
                        'OBJID','RUN','RERUN','CAMCOL','SKYVERSION','LAMBDA_EFF',
                        'FIELD','PSFFLUX','PSFFLUX_IVAR','PSFMAG','PSFMAGERR',
                        'EXTINCTION','SN_MEDIAN_ALL','ZOFFSET','XFOCAL','YFOCAL','CHUNK',
                        'TILE','PLATESN2'])
    vdcopycols = np.array(['CLASS_PERSON','Z_CONF','Z_VI'])

    #Load the three files I need.
    virec_name = '../data/pre16_vi_database.fits'
    inrec = fits.open(inrec_name)[1].data #My classified FITS record
    virec = fits.open(virec_name)[1].data #The VI database dump from Patrick.

    vnum = 'v5_13_0'
    #Load the full current spAll file too.
    loc = os.environ.get('BOSS_SPECTRO_REDUX')
    full_file = '{0}/{1}/spAll-{2}.fits'.format(loc,vnum,vnum)
    sprec = fits.open(full_file)[1].data

    #Make the hash arrays.
    inhash = ct.mk_hash(inrec)
    vihash = ct.mk_hash(virec)
    sphash = ct.mk_hash(sprec)

    #Find the common objects from my FITS file and the vi database.
    inargs,viargs = ct.rec_match_srt(inhash,vihash)

    #Find the objects in the vi database that aren't in my FITS file.
    vi_mask = np.zeros(len(virec),dtype='i2')
    vi_mask[viargs] = 1
    wv0 = np.where(vi_mask==0)[0]
    vinhash = ct.mk_hash(virec[wv0])
    #Find those objects that aren't in my FITS file that ARE in the spAll file.
    vinargs,spargs = ct.rec_match_srt(vinhash,sphash)

    #The total number of records the output will hold. This is without duplicate
    #reduction.
    num_in = len(inrec)
    num_vin = len(vinhash)
    num_tot = len(inrec) + len(vinargs)

    #Create the structured array that will hold my record first.
    inarr = ecm.supercat_maker(num_in)
    vinarr = ecm.supercat_maker(num_vin)

    #Copy everything from my record to the structured array.
    for cname in spcolkeep:
        if cname == 'Z':
            cname1 = 'Z_PIPE'
        elif cname == 'RUN':
            cname1 = 'RUN_NUMBER'
        elif cname == 'RERUN':
            cname1 = 'RERUN_NUMBER'
        elif cname == 'CAMCOL':
            cname1 = 'CAMCOL_NUMBER'
        elif cname == 'FIELD':
            cname1 = 'FIELD_NUMBER'
        else:
            cname1 = cname
        inarr[cname1] = inrec[cname]

    #Copy the VI information from the VI database to the struct.
    for cname in vdcopycols:
        inarr[cname][inargs] = virec[cname][viargs]

    #Load everything from the spAll file for the VI objects that didn't match my record.
    for cname in spcolkeep:
        if cname == 'Z':
            cname1 = 'Z_PIPE'
        elif cname == 'RUN':
            cname1 = 'RUN_NUMBER'
        elif cname == 'RERUN':
            cname1 = 'RERUN_NUMBER'
        elif cname == 'CAMCOL':
            cname1 = 'CAMCOL_NUMBER'
        elif cname == 'FIELD':
            cname1 = 'FIELD_NUMBER'
        elif cname == 'AUTOCLASS_DR14Q':
            vinarr['AUTOCLASS_DR14Q'][vinargs] = 'UNK'
            continue
        else:
            cname1 = cname
        vinarr[cname1][vinargs] = sprec[cname][spargs]

    #Copy the VI information from the VI database.
    for cname in vdcopycols:
        vinarr[cname][vinargs] = virec[cname][wv0[vinargs]]

    #Copy both placeholder structs into one struct.
    dsarr = ecm.supercat_maker(num_tot)
    colnames = np.array(dsarr.dtype.names)
    for cname in colnames:
        dsarr[cname][0:num_in] = inarr[cname][:]
        dsarr[cname][num_in:num_tot] = vinarr[cname][vinargs]

    #Now we need to generate the SDSS names that go with each quasar
    name_tarr = sdn.s_name(dsarr['RA'],dsarr['DEC'])
    dsarr['SDSS_NAME'] = name_tarr

    #And sort the catalog by SDSS name, which is standard
    name_args = np.argsort(dsarr['SDSS_NAME'])
    dsarr = dsarr[name_args]


    #We might want all of the records, before any removals. This does that.
    prim_hdr = fits.open(inrec_name)[0].header
    prim_hdu = fits.PrimaryHDU(header=prim_hdr)

    #We need to make the output file name for the catalog that includes
    #the records with bad ZWARNING flags.
    outfile_dtag = time.strftime('%Y%m%d')
    outfile_name = 'DR16Q_vdb_zw_{}'.format(outfile_dtag)

    #Make the HDU1 catalog and write it out.
    data_hdu = fits.BinTableHDU.from_columns(dsarr)
    data_of = fits.HDUList([prim_hdu,data_hdu])
    vdb_out_name = ct.fet(data_of,outfile_name)

    return vdb_out_name

##################REMOVE BAD ZWARNING FLAGS#####################################

def zw_remove(ifile):
    #We also want to remove all records with bad ZWARNING flags.
    zwarn_arr = ct.file_load(ifile)
    
    zwarn_mask = np.zeros(len(zwarn_arr),dtype='i2')
    zwarn_mask[:] = 1
    unplug_val = flagval('ZWARNING',['UNPLUGGED','SKY','LITTLE_COVERAGE','NODATA',
                         'BAD_TARGET'])
    wunplugged = np.where(zwarn_arr['ZWARNING']&unplug_val)[0]
    zwarn_mask[wunplugged] = 0
    wplugged = np.where(zwarn_mask == 1)[0]

    #Again, we need to grab the HDU0 header information    
    prim_hdr = fits.open(ifile)[0].header
    prim_hdu = fits.PrimaryHDU(header=prim_hdr)

    #We also need to make another output file name
    outfile_dtag = time.strftime('%Y%m%d')
    outfile_name = 'DR16Q_vdb_rem_{}'.format(outfile_dtag)

    #And make the output data HDU1 and HDU list
    data_out = np.array(zwarn_arr[wplugged])
    data_hdu = fits.BinTableHDU.from_columns(data_out)
    data_hdulist = fits.HDUList([prim_hdu,data_hdu])
    
    #Write out that file. This is the last write out from this program
    vdb_zw_name = ct.fet(data_hdulist,outfile_name)

    #Return the name of the file that contains VI database data.
    #This is not the superset yet.
    #Note to me: I have gotten this far in git-prep.
    return vdb_zw_name

#############COMMAND LINE FUNCTION CALLS########################################

#This requires the utilities scripts:
#    progressBar.py, sdss_name_v2.py, cat_tools.py, and empty_cat_maker.py
#This requires the following data file in ../data/
#    pre16_vi_database.fits

#Run from command line with:
#  python spAll_class_full.py
#  and wait.
if __name__=='__main__':
    supername = cat_cut() #This will run the superset cutdown.
    clsnamed = obj_class(supername) #This will run the Classification Algorithm.
    vdb_combo_name = cat_combine(clsnamed) #This combines the superset with the VI db.
    final_outname = zw_remove(vdb_combo_name) #This removes records with bad ZWARNINGs.
