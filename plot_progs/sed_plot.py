"""
This creates the plot that is used as Fig. 8 in Lyke et al. 2020. This spectrum
file and the catalog file is needed to make this plot. This will not work (and
has no error-trapping) for records that are missing any data from:
UKIDSS, ROSAT, FIRST, GALEX, or WISE. The spectrum file must have a positive
value for flux and ivar at every point. Spectra with bad pixels are NOT supported.

As UKIDSS and 2MASS overlap in wavelength coverage, only one should be used. 2MASS
data conversion is not supported here. This pretty much leaves UKIDSS only (for
this script).

There will be NO updates to this script to include 2MASS conversions.

Gaia data is not converted or plotted as the large time gap between SDSS optical and
Gaia optical observations cannot be plotted together in a meaningful way due to
known quasar flux variability.

The terms 'flux' and 'flux density' are used interchangeably throughout. Context
is key.

Dependencies
----------
spec file : requires the SDSS full spectrum file in ../data/
github repository : https://github.com/bradlyke/utilities
Note: To get these plots, LaTeX must be installed and
      matplotlib must be able to compile LaTeX commands


Input file
----------
The most recent version of the DR16Q quasar-only catalog.


Parameters
----------
input_file : the file name for the DR16Q FITS file in the
             ../data folder. No path needed.
pt,mt,ft : Plate, MJD, and FiberID for a record that has all multiwavelength data
           The values used to make the paper plot are hardcoded in __main__
           No guarantees are made about script viability or plot viability
           for any spectrum other than the hardcoded one.
output_plot_name : the name of the plot written out
plot_fontsize : The fontsize for everything in the plot
               For a 10x4 plot spanning twocolumn, 11 works best.
save_check : 0 - don't save, just plot
             1 - don't plot, just save into ../plots/


Output
----------
If selected, an EPS file of the plot.

"""

from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rc('text',usetex=True)
import progressBar_fancy as pbf
import cat_tools as ct
from sciCon import mks
import sys

#This function will smooth the SDSS spectrum using a boxcar window of a
#given pixel size (smooth_pct).
#This is a copy of the smoothing function from another of my repositories:
#  https://github.com/bradlyke/ica_5010/rolling_boxcar.ipynb
#Copied over so users don't have to clone another repository for one function.
def boxcar(flux_arr,flux_var,smooth_pct,weighted=False):
    spc = smooth_pct
    num_dpoints = len(flux_arr) #how many data points
    box_flux = np.zeros(num_dpoints,dtype='f8') #This is the smoothed flux data
    box_err = np.zeros(num_dpoints,dtype='f8') #This is the smoother flux error

    #This does the box. 10 pixels wide, so choose 5 on each side of current pixel
    for i in range(num_dpoints):
        #Define range of values to smooth
        if i < int(spc/2):
            lower = 0
        else:
            lower = -int(spc/2) + i
        if ((int(spc/2) + i) > num_dpoints):
            upper = num_dpoints
        else:
            upper = int(spc/2) + i
        #Smooth the values depending on smoothing type
        #Weighted takes each pixels error into account and the pixel flux is inversely
        #proportional to the pixel uncertainty
        if weighted==True:
            noise_temp = np.sqrt(flux_var[lower:upper])
            signal_temp = flux_arr[lower:upper]
            flux_temp = np.average(signal_temp,weights=noise_temp)
            ivar_temp = (np.average((signal_temp-flux_temp)**2, weights=noise_temp))**(-1.0)
        #Unweighted version. Why would you want to use this?
        else:
            flux_temp = np.median(flux_arr[lower:upper])
            ivar_temp = flux_var[i]

        #Save the NEW smoothed flux value
        box_flux[i] = flux_temp
        box_err[i] = ivar_temp

    return box_flux,box_err

"""
To make a Spectral Energy Distribution (SED), all of the different sources
of data have to be on the same flux scale (y-axis). These following functions
will convert the different surveys' data in DR16Q to the standard SDSS flux
units of 10^(-17) erg/s/cm^2/Angstrom.
uk_convert(): UKIDSS data
jy_convert(): Anything in Janskys.
nano_convert(): Anything in nanomaggies (GALEX or WISE data)
xray_convert(): Anything in flux, F (erg/s/cm^2), but NOT flux density, F_lambda (erg/s/cm^2/Ang)
                ROSAT and XMM both use F.

"""
#Remember that the ouput on all of these needs to be:
# 10^(-17) erg/s/cm^2/Ang
def uk_convert(influx,inlam):
    #I will convert W/m^2/Hz from UKIDSS here
    flux_temp = influx*1000 #converts W/m^2/Hz to erg/s/cm^2/Hz
    flux_temp = flux_temp * (mks.c*10**(10)) * inlam**(-2.0) #converts /Hz to /Ang
    flux_temp = flux_temp * 1e17 #convert to SDSS version
    return flux_temp

def jy_convert(fjy,inlam):
    #I will convert Jy or mJy here
    flux_temp = fjy / 1000 #converts mJy to Jy
    flux_temp = flux_temp * 1e-26 #converts Jy to W/m^2/Hz
    flux_temp = uk_convert(flux_temp,inlam) #converts W/m^2/Hz to erg/s/cm^2/Ang
    #uk_convert() already reports in SDSS version
    return flux_temp

def nano_convert(influx,inlam):
    #I will convert nanomaggies from GALEX and WISE here
    #flux_source will be 'GALEX' or 'WISE' as they may be on separate systems
    flux_temp = influx * 3.631e-6 #converts to Jy
    flux_temp = flux_temp * 1e-26 #converts to W/m^2/Hz
    flux_temp = uk_convert(flux_temp,inlam) #converts W/m^2/Hz to erg/s/cm^2/Ang
    #uk_convert() already reports in SDSS version

    return flux_temp

def xray_convert(influx,inlam):
    #I will convert ROSAT/XMM here. They are in Flux, not flux density
    #ROSAT(2RXS) will have one data point. XMM will have 2 (soft/hard) or
    #3 (soft/hard/total). The inlam should be the center wavelength for the range
    #and must be input in Angstroms. Conversion from keV to Ang in sed_plotter()
    flux_temp = influx / inlam
    flux_temp = flux_temp * 1e17 #convert to SDSS scaling

    return flux_temp

#def tmass_convert(influx,inlam):
    #Convert 2MASS magnitudes here. An online program does this, check there:
    # http://ssc.spitzer.caltech.edu/warmmission/propkit/pet/magtojy/
    #2MASS conversions are not supported for DR16Q. The spectrum in the paper
    #used UKIDSS data for the wavelenghts in question.

def sed_plotter(infile,pt,mt,ft,fname,fntsize,write_check):
    dr = fits.open(infile)[1].data #Load the catalog file
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    drloc = ct.find_rec(dr,pt,mt,ft) #Find record for the specified quasar

    #Load the SDSS spectrum
    spec_iname = '../data/spec-{0}-{1}-{2:04d}.fits'.format(pt,mt,ft)
    spec_data = fits.open(spec_iname)[1].data
    spec_lam = 10**spec_data['loglam'] #wavelength stored as log10(lambda) in file.
    spec_err = spec_data['ivar'] #error stored as inverse variance per pixel
    wspec = np.where(spec_lam <= 10000)[0] #We don't want UKIDSS to overlap SDSS
    #Note that SDSS spectra above lambda=10,000 Ang are typically dominated by sky emission
    spec_flux = spec_data['flux']
    spec_fluxS, spec_errS = boxcar(spec_flux,spec_err,10,weighted=True) #10 pixel window size

    #Converting the GALEX data
    gal_data = np.zeros(2,dtype=[('LAM','float64'),('FLUX_RAW','float64'),('FLUX_CONV','float64'),
                                            ('FLUX_ERR_RAW','float64'),('FLUX_ERR_CONV','float64')])
    gal_data['LAM'][0] = 1550 #Angstroms, FUV bandpass = 1350 - 1750 Ang
    gal_data['LAM'][1] = 2275 #Angstroms, NUV bandpass = 1750 - 2800 Ang
    gal_data['FLUX_RAW'][0] = dr['FUV'][drloc]
    gal_data['FLUX_RAW'][1] = dr['NUV'][drloc]
    #Convert inverse variance to a 1-sigma uncertainty for error bars later
    gal_data['FLUX_ERR_RAW'][0] = 1/np.sqrt(dr['FUV_IVAR'][drloc])
    gal_data['FLUX_ERR_RAW'][1] = 1/np.sqrt(dr['NUV_IVAR'][drloc])
    #Convert GALEX flux and uncertainty to SDSS units
    gal_data['FLUX_CONV'] = nano_convert(gal_data['FLUX_RAW'],gal_data['LAM'])
    gal_data['FLUX_ERR_CONV'] = nano_convert(gal_data['FLUX_ERR_RAW'],gal_data['LAM'])

    #Converting the UKIDSS data
    ukid_data = np.zeros(4,dtype=[('LAM','float64'),('FLUX_RAW','float64'),('FLUX_CONV','float64'),
                                        ('FLUX_ERR_RAW','float64'),('FLUX_ERR_CONV','float64')])
    #Get passband centers in Angstroms (comments are in microns)
    ukid_data['LAM'][0] = 10200 #Y-band, Angstroms, 50% cuton/cutoff = 0.97/1.07 microns Y
    ukid_data['LAM'][1] = 12500 #J 1.17/1.33
    ukid_data['LAM'][2] = 16350 #H 1.49/1.78
    ukid_data['LAM'][3] = 22000 #K 2.03/2.37
    ukid_data['FLUX_RAW'][0] = dr['YFLUX'][drloc]
    ukid_data['FLUX_RAW'][1] = dr['JFLUX'][drloc]
    ukid_data['FLUX_RAW'][2] = dr['HFLUX'][drloc]
    ukid_data['FLUX_RAW'][3] = dr['KFLUX'][drloc]
    ukid_data['FLUX_ERR_RAW'][0] = dr['YFLUX_ERR'][drloc]
    ukid_data['FLUX_ERR_RAW'][1] = dr['JFLUX_ERR'][drloc]
    ukid_data['FLUX_ERR_RAW'][2] = dr['HFLUX_ERR'][drloc]
    ukid_data['FLUX_ERR_RAW'][3] = dr['KFLUX_ERR'][drloc]
    #Convert UKIDSS flux and error to SDSS units
    ukid_data['FLUX_CONV'] = uk_convert(ukid_data['FLUX_RAW'],ukid_data['LAM'])
    ukid_data['FLUX_ERR_CONV'] = uk_convert(ukid_data['FLUX_ERR_RAW'],ukid_data['LAM'])

    #Radio, X-ray, and WISE (infrared) data. Radio/X-ray does not appear on the plot
    #as these are so far out in wavelength space they are difficult to plot even on
    #a log scale. The flux ranges for these are also of significantly
    #lesser (radio)/greater (X-ray) magnitude to SDSS, that the y-axis also won't
    #work well. Converted here for instructional purposes, but not used in the plot.
    radxw_data = np.zeros(4,dtype=[('LAM','float64'),('FLUX_RAW','float64'),('FLUX_CONV','float64'),
                                    ('FLUX_ERR_RAW','float64'),('FLUX_ERR_CONV','float64')])
    #Convert ROSAT data (not used)
    radxw_data['LAM'][0] = ((mks.h * mks.c)/(1250*mks.eV))*(10**10) #Angstroms, ROSAT is 0.5 - 2.0 keV
    radxw_data['FLUX_RAW'][0] = dr['2RXS_SRC_FLUX'][drloc]
    radxw_data['FLUX_CONV'][0] = xray_convert(radxw_data['FLUX_RAW'][0],radxw_data['LAM'][0])
    radxw_data['FLUX_ERR_RAW'][0] = dr['2RXS_SRC_FLUX_ERR'][drloc]
    radxw_data['FLUX_ERR_CONV'][0] = xray_convert(radxw_data['FLUX_ERR_RAW'][0],radxw_data['LAM'][0])
    #Convert WISE infrared data (USED in plot)
    radxw_data['LAM'][1] = 34000 #Angstroms, W1 band center
    radxw_data['LAM'][2] = 46000 #Angstroms, W2 band center
    radxw_data['FLUX_RAW'][1] = dr['W1_FLUX'][drloc]
    radxw_data['FLUX_RAW'][2] = dr['W2_FLUX'][drloc]
    radxw_data['FLUX_CONV'][1:3] = nano_convert(radxw_data['FLUX_RAW'][1:3],radxw_data['LAM'][1:3])
    radxw_data['FLUX_ERR_RAW'][1] = 1/np.sqrt(dr['W1_FLUX_IVAR'][drloc])
    radxw_data['FLUX_ERR_RAW'][2] = 1/np.sqrt(dr['W2_FLUX_IVAR'][drloc])
    radxw_data['FLUX_ERR_CONV'][1:3] = nano_convert(radxw_data['FLUX_ERR_RAW'][1:3],radxw_data['LAM'][1:3])
    #Convert FIRST radio data (not used)
    radxw_data['LAM'][3] = 20e8 #Angstroms, 20cm for FIRST
    radxw_data['FLUX_RAW'][3] = dr['FIRST_FLUX'][drloc]
    radxw_data['FLUX_CONV'][3] = jy_convert(radxw_data['FLUX_RAW'][3],radxw_data['LAM'][3])
    radxw_data['FLUX_ERR_RAW'][3] = (dr['FIRST_FLUX'][drloc]-0.25)/dr['FIRST_SNR'][drloc]
    radxw_data['FLUX_ERR_CONV'][3] = jy_convert(radxw_data['FLUX_ERR_RAW'][3],radxw_data['LAM'][3])

    #Print the converted flux values for each band pass. Using the default spectrum
    #in the paper, this will show why FIRST and ROSAT were not plotted.
    print('\n')
    print('LAMBDA: FUV | FLUX: {:.3e} | ERR: {:.3e}'.format(gal_data['FLUX_CONV'][0],gal_data['FLUX_ERR_CONV'][0]))
    print('LAMBDA: NUV | FLUX: {:.3e} | ERR: {:.3e}'.format(gal_data['FLUX_CONV'][1],gal_data['FLUX_ERR_CONV'][1]))

    print('LAMBDA: Y | FLUX: {:.3e}'.format(ukid_data['FLUX_CONV'][0]))
    print('LAMBDA: J | FLUX: {:.3e}'.format(ukid_data['FLUX_CONV'][1]))
    print('LAMBDA: H | FLUX: {:.3e}'.format(ukid_data['FLUX_CONV'][2]))
    print('LAMBDA: K | FLUX: {:.3e}'.format(ukid_data['FLUX_CONV'][3]))

    print('LAMBDA: ROSAT | FLUX: {:.3e}'.format(radxw_data['FLUX_CONV'][0]))
    print('LAMBDA: W1 | FLUX: {:.3e}'.format(radxw_data['FLUX_CONV'][1]))
    print('LAMBDA: W2 | FLUX: {:.3e}'.format(radxw_data['FLUX_CONV'][2]))
    print('LAMBDA: FIRST | FLUX: {:.3e}'.format(radxw_data['FLUX_CONV'][3]))

    print('\nFIRST RAW: {:.3e}'.format(radxw_data['FLUX_RAW'][3]))

    #The bottom x-axis will be in the observed frame, the top will have the
    #wavelengths in the rest frame. These need different tick labels.
    xtick_obsLabs = np.array([1400,2000,4000,10000,20000,40000])
    xtick_restLabs = np.array([1000,2000,4000,10000,20000])

    #The SDSS name of the quasar and redshift to include in a text box.
    name_str = r'\textbf{SDSS J002912.35+022549.5, z = 0.576}'

    #Now we plot. Error bars for multiwave data may not display as they are
    #very small compared to the flux values (1 to 2 orders of magnitude smaller)
    fig,ax = plt.subplots(figsize=(10,4))
    axT = ax.twiny() #This makes a set of tick labels along the top (twin ALONG y-axis)
    #ax.errorbar(radxw_data['LAM'][0],radxw_data['FLUX_CONV'][0],yerr=radxw_data['FLUX_ERR_CONV'][0],label='ROSAT') #ROSAT DATA
    ax.errorbar(gal_data['LAM'],gal_data['FLUX_CONV'],yerr=gal_data['FLUX_ERR_CONV'],color='blue',alpha=0.7,fmt='d',label=r'\textbf{GALEX}') #GALEX DATA with ERROR
    ax.plot(spec_lam[wspec],spec_flux[wspec],linewidth=0.8,color='0.70',alpha=1.0,label=r'\textbf{SDSS Raw}') #SDSS DATA
    ax.plot(spec_lam[wspec],spec_fluxS[wspec],linewidth=0.6,color='black',alpha=1.0,label=r'\textbf{SDSS Smoothed}') #SDSS DATA
    ax.errorbar(ukid_data['LAM'],ukid_data['FLUX_CONV'],yerr=ukid_data['FLUX_ERR_CONV'],color='darkorange',alpha=1.0,fmt='^',label=r'\textbf{UKIDSS}') #UKIDSS DATA with ERROR
    ax.errorbar(radxw_data['LAM'][1:3],radxw_data['FLUX_CONV'][1:3],yerr=radxw_data['FLUX_ERR_CONV'][1:3],color='red',fmt='o',label=r'\textbf{WISE}') #WISE DATA with ERROR
    #ax.errorbar(radxw_data['LAM'][3],radxw_data['FLUX_CONV'][3],yerr=radxw_data['FLUX_ERR_CONV'][3],label='FIRST') #FIRST DATA

    ax.set_xscale('log') #Lower x-axis scale
    #Convert observed wavelengths to rest by lam_obs / (1+z)
    top_lim = np.array(ax.get_xlim())/1.576
    print('Rest Limits: ',top_lim) #Print these for verification by user
    #Top x-axis labels and scale
    axT.set_xlim(top_lim)
    axT.set_xscale('log')
    #Everything done to the bottom x-axis ticks needs to be done to the top (axT)
    ax.set_xticks(xtick_obsLabs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axT.set_xticks(xtick_restLabs)
    axT.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r'\textbf{Observed Frame Wavelength (\AA)}')
    axT.set_xlabel(r'\textbf{Rest Frame Wavelength (\AA)}')
    ax.set_ylabel(r'$f_{\lambda}$ ($10^{-17}$ \textbf{erg s}$^{-1}$\, \textbf{cm}$^{-2}$\, \textbf{\AA}$^{-1}$)')

    ##Bottom X-axis and Left/Right Y-axis minor ticks
    ax.tick_params(axis='both',direction='in')
    ax.tick_params(axis='both',which='minor',direction='in')

    #Right-side ticks
    ax.tick_params(right=True)
    ax.tick_params(which='minor',right=True)

    #Top minor ticks
    axT.tick_params(direction='in')
    axT.tick_params(which='minor',direction='in')
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10)) #Y-axis minor ticks locations
    ax.legend(loc='upper right',facecolor='white',edgecolor='white')

    #This calls the name_str from earlier. Puts it in the bottom left
    ax.text(0.01,0.1,name_str,transform=ax.transAxes, verticalalignment='top',bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0))

    #Write out the plot, or just view it
    if write_check == 1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#Call from the command line with:
#    python sed_plot.py <DR16Q file name> <plot name> <font size> <save flag>
# Example:
#    python sed_plot.py DR16Q_v3.fits sed.eps 11 1
if __name__=='__main__':
    input_file = '../data/{}'.format(sys.argv[1])
    output_plot_name = '../plots/{}'.format(sys.argv[2])
    plot_fontsize = int(sys.argv[3])
    save_check = int(sys.argv[4])
    pt,mt,ft = 7855,57011,530
    sed_plotter(input_file,pt,mt,ft,output_plot_name,plot_fontsize,save_check)
