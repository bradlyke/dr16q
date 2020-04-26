"""
A calculator to find the absolute i-band magnitudes for quasars
given a redshift, the K-correction table, and apparent i-band
magnitudes, and i-band galactic extinction. Comoving radial
distance is calculated using Simpson's Method for integration.
This program uses a "general" cosmology with a flat universe.

Dependencies
----------
csv file : richards_kcorr_table.dat (in data folder)
git repository: https://github.com/bradlyke/utilities


Input file Requirements
----------
Input file must be a FITS file with u,g,r,i,z arrays for 
  apparent magnitudes and extinctions. Must have the 
  same column names that appear in 
  utilties/empty_cat_maker.multiwave_maker()


Parameters
----------
input_file : :class:'str'
             The file name/path of the FITS catalog.

Output
----------
A FITS table with the calculated absolute i-band magnitudes.

"""

from astropy.io import fits
import numpy as np
import tmark
import sys
import cat_tools as ct

#Cosmological Constants are taken from SDSS-IV DR16 fiducial cosmology
#from Planck2018+BAO.
#At time of calculation these were:
#  H0 = 67.6 km/s/Mpc, Omega_M = 0.31, Omega_L = 0.69, 
#  Omega_R = 4.165e-5/(H0*H0)/10000, c = 299792.458 km/s

#Calculates the f(z) for finding comoving radial distance using integral form.
def z_func(z):
    bM = omegaM * (1 + z)**3 #Matter term
    bL = 1 - omegaM - omegaR #Dark energy (vacuum term)
    bR = omegaR * (1 + z)**4 #Radiation term, includes relativistic neutrinos.
    denom = h0 * np.sqrt(bM + bL + bR) #denominator of proper distance
    dr_temp = c  * denom**(-1.0) #differential proper distance for current z_now
    return dr_temp

#Calculates the comoving radial distance using Simpson's Rule.
#Testing with Midpoint Rule and Trapezoid rule showed that all three methods,
#at 10,000 steps, agreed to 1e-4 Mpc.
def coradial_simp(z):
    num_steps = 10000 #number of steps from 0 to z
    dz = z / num_steps #step size.
    z_now = 0 #initlized the placeholder for partial proper distance
    dr0 = z_func(0) #Evaluate the start point
    drN = z_func(z) #Evaluate the end point
    drS = dr0 + drN #Add these together to initialize dr
    #Adds the middle terms in the Simpson's method
    for i in range(1,num_steps):
        z_0 = i * dz #current z_i
        #If we are on an even value for i, use 2*f(z)
        if (i % 2 == 0):
            p_now = 2 * z_func(z_0)
        #If we are on an odd value for i, use 4*f(z)
        else:
            p_now = 4 * z_func(z_0)
        drS = drS + p_now#Cumulative sum
    drSF = (dz / 3.0) * drS #Last step of Simpson's Method
    return drSF #Output comoving radial distance in Mpc

#Calculate the Luminosity distance from the comoving radial distance and redshift.
def lum_dist(dr,z):
    #dr must input as Mpc
    return dr * (1 + z) #Output luminosity distance in Mpc

#Calculate the distance modulus using the luminosity distance.
def dist_mod(dl):
    #dl must be input as Mpc
    return 5*np.log10(dl) + 25 #unitless

#This loads the Richards et al. 2006 K-Correction table. Loading done separately from 
#finding the K-Correction factor so that list comprehension can be used in get_mags().
#The table must have the header removed and converted to a csv format (no spaces).
def kcorr_load():
    k_table = np.loadtxt('../data/richards_kcorr_table.dat',dtype='f4',delimiter=',') #Load table
    ktout = np.zeros(len(k_table),dtype=[('Z','f4'),('KCORR','f4')]) #I prefer named columns
    ktout['Z']=k_table[:,0]
    ktout['KCORR']=k_table[:,1]
    return ktout #return the K-correction table
    
#This function finds the K-Correction factor from the closest matching redshift. This
#is done singly, so that list comprehension can be used in get_mags() like coradial_simp()
def kcorr_find(z,karr):
    diff_arr = np.absolute(z - karr['Z']) #find the differences between z and table.
    min_arg = np.argmin(diff_arr) #find the row with the smallest difference
    kval = karr['KCORR'][min_arg] #Get the K-Correction factor
    kvalF = float('{:.3f}'.format(kval)) #Strip down to the 3 decimals it is in the table.
    return kvalF #return the K-correction value for that object

#Find the absolute magnitude given the previously calculated values and data
#taken from the fits file.
def abs_Mag(imag,ext,dm,kc):
    #imag is the apparent i-band magnitude, ext is i-band galactic extinction
    #dm is distance modulus from dist_mod(), and kc is k correction from kcorr_find().
    return imag - ext - dm - kc #unitless

#Primary program for reading the FITS file and calculating redshift for all records.
#ifile is the input fits file.
def get_mags(ifile):
    #Set up the cosmological paramters as global variables for use in other functions.
    global h0
    global omegaM
    global h100
    global omegaR
    global c
    global omegaL
    
    drfile = ct.file_load(ifile) #Loads the FITS catalog as a numpy structued array
    h0 = 67.6 #Hubble constant at current time in km/s/Mpc
    omegaM = 0.310 #Critical density of all matter at current time
    omegaL = 0.690 #Critical density of dark energy at current time
    h100 = h0/100. #Standard unitless representation of h0
    omegaR = 4.165e-5/(h100*h100) #Critical density of radiation at current time
    c = 299792.458 #The speed of light in km/s

    num_rec = len(drfile) #Find the number of objects
    drfile['M_I'][:] = 1 #Set to placeholder for objects with bad photometry

    #We only want objects with good photometry and a good Z_PCA value
    #The Richards table is limited to z_max of 5.49 anyways
    #PSFMAG and EXTINCTION arrays are stored in u, g, r, i, z order
    wg = np.where((drfile['PSFMAG'][:,3]>-1)&(drfile['EXTINCTION'][:,3]>-1)&
                  (drfile['Z_PCA']>0)&(drfile['Z_PCA']<=5.0))[0]
    #If an object has good photometry but a bad Z_PCA value, we want to know that
    wb = np.where((drfile['PSFMAG'][:,3]>-1)&(drfile['EXTINCTION'][:,3]>-1)&
                   ((drfile['Z_PCA']<=0)|(drfile['Z_PCA']>5.0)))[0]
    drfile['M_I'][wb] = 2 #If the object has good photometry but bad Z_PCA I want these marked
    # Output: M_I = 1 -- bad photometry, M_I = 2 -- good photometry, bad Z_PCA. 
    # Otherwise records M_I.

    tmark.tm('Starting Absolute Magnitude Calculations')
    #Iterate through all objects.
    #Calculating M_I for a large number of records may take a while.
    #When ~750,000 records were used, it took ~12 hours. Number of
    #steps in coradial_dist() greatly affects this, but also reduces
    #distance accuracy. Simpson's Rule is accurate on O(dz**4).
    rec_z = drfile['Z_PCA'][wg] #Get the redshifts for the good objects
    iMag = drfile['PSFMAG'][wg,3] #Get the apparent i-band magnitudes
    A_i = drfile['EXTINCTION'][wg,3] #Get the i-band extinctions
    Dr = np.array([coradial_simp(x) for x in rec_z]) #Find the coradial distance with list comp
    Dl = lum_dist(Dr,rec_z) #Get the luminosity distance
    Dm = dist_mod(Dl) #Get the distance modulus
    ktab = kcorr_load() #Load the K-correction table.
    kvals = np.array([kcorr_find(x,ktab) for x in rec_z]) #Find K-correction values.
    absMag = abs_Mag(iMag,A_i,Dm,kvals) #find the absolute i-band magnitudes
    drfile['M_I'][wg] = absMag #And save them back in the primary array

    #User feedback for when the program completes.
    print('\n')
    print('Absolute Magnitudes Calculated')
    
    #The following writes out the new fits record with the absolute magnitude column.
    prim_hdr = fits.open(ifile)[0].header #Grad HDU0 header from input file
    prim_hdu = fits.PrimaryHDU(header=prim_hdr) #And make it this output file HDU0 header

    #Convert the data structued array into a FITS data HDU named "CATALOG"
    data_hdu = fits.BinTableHDU.from_columns(drfile,name='CATALOG')
    
    #Generate a unique output name so there are no failures on file write.
    ofile_dtag = time.strftime('%Y%m%dT%H%M%S') #Date and time tag
    outfile_name = '../data/DR16Q_absmag_{}.fits'.format(ofile_dtag) #Output file name.
    
    #Put together the primary HDU (HDU0) and data HDU (HDU1). Write it all out.
    data_out = fits.HDUList([prim_hdu,data_hdu])
    data_out.writeto(outfile_name)
    
    return outfile_name #Returns the written file name
    

if __name__=='__main__':
    #If the input catalog file is not in the data folder, include the file path.
    input_file_name = sys.argv[1] #Give the name of the input file
    input_file = '../data/{}'.format(input_file_name)
    mi_name = get_mags(input_file) #Used to run from command line
    print('\nCatalog File written out as: {}'.format(mi_name))
