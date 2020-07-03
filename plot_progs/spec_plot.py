"""
This creates FOUR spectral plots used used in the appendix as Fig. 10-13.
This will only work for the four pre-selected spectra listed in __main__.

Dependencies
----------
github repository : https://github.com/bradlyke/utilities
Note: To get these plots, LaTeX must be installed and
      matplotlib must be able to compile LaTeX commands


Input file
----------
The most recent version of the DR16Q quasar-only catalog.

Parameters
----------

p : [10,11,12,13] The figure number from the paper

-h : Output help text to terminal (spec_plot.py -h)
-e : Plot the error spectrum in red
-k : Plot the sky spectrum in green
-x : Scale the sky flux to the quasar flux (so it's readable)
-s : Save the figure (does not display)

Output
----------
If selected, an EPS file of the plot.

"""

#Import the tools necessary to work with spectra files.
import cat_tools as ct #This is a tool I wrote, so you'll need the script in your path.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import sys
matplotlib.rc('text',usetex=True)

#A spectrum is a class. The object has 3 attributes right now: self, boxcar,
#paper_plot_small.
class spectrum:
    #initialize the object by loading the file. Needs a file name for infile.
    def __init__(self,infile):
        self.infile = infile #Load the filename as a thing for later use in plot saving.
        self.data = ct.file_load(infile) #Load the file as a numpy structured array.
        self.loglam = self.data['loglam'] #SDSS stores wavelengths as log10 values.
        self.lam = 10**self.loglam #Convert this to a decimal wavelength in Angstroms.
        self.flux = self.data['flux'] #Load the flux. Absolute, units are in plot.
        self.ivar = self.data['ivar'] #Load the inverse variance. Needed for plotting and boxcar.
        self.ferr = self.data['ivar']**(-0.5)

    '''
    We want to be able to smooth it fairly easily. Smooth_pct = 10 is good for
    emphasizing broad emission lines through noise. Smooth_pct = 1000 will
    remove (almost) all features and noise, leaving a useful continuum

    This uses a square window of width smooth_pct to perform a modified
    moving average on a spectrum using a boxcar (or "square" or "tophat")
    window. The box calculates a new flux for a point using the original
    signal flux, NOT the updated (smoothed) flux. This should only be used
    to make plots more readable, not for scientific analysis.
    '''
    def boxcar(self,flux_arr,flux_var,smooth_pct,weighted=False):
        spc = smooth_pct
        num_dpoints = len(flux_arr)
        #Initialize the new flux and error vectors.
        box_flux = np.zeros(num_dpoints,dtype='f8')
        box_err = np.zeros(num_dpoints,dtype='f8')

        #The window should be symmetrical, but won't be near the ends of the
        #spectrum, so it needs to be truncated properly, as padding won't
        #allow the edges to be properly averaged.
        for i in range(num_dpoints):
            #Define range of values to smooth
            #We have to check if the window can be symmetrical. If it is closer
            #to the edge of spectrum than half of smooth_pct it can't be
            #symmetrical, so truncate the denominator when averaging.
            if i < int(spc/2):
                lower = 0
            else:
                lower = -int(spc/2) + i
            if ((int(spc/2) + i) > num_dpoints):
                upper = num_dpoints
            else:
                upper = int(spc/2) + i
            #Smooth the values depending on smoothing type
            #Weighting use the flux error to do a weighted average, but if the
            #spectrum flux is smoothed, then so must the error spectrum, or else
            #it will be the wrong scale.
            #Note that this always calls a box of the ORIGINAL signal, not taking
            #previously smoothing pixels into account.
            if weighted==True:
                noise_temp = np.sqrt(flux_var[lower:upper])
                signal_temp = flux_arr[lower:upper]
                flux_temp = np.average(signal_temp,weights=noise_temp)
                ivar_temp = (np.average((signal_temp-flux_temp)**2, weights=noise_temp))**(-1.0)
            else:
                flux_temp = np.median(flux_arr[lower:upper])
                ivar_temp = flux_var[i]
            #If the smoothing is not weighted by error, the error isn't smoothed
            #and thus will no longer be properly scaled.

            #These hold the smoothed flux and (possibly) smoothed error.
            box_flux[i] = flux_temp
            box_err[i] = ivar_temp

        return box_flux,box_err

    #A "small paper plot" is just one of the attributes taken from the
    #specClass in the utilities repository. The emission line labels had
    #to be added by hand for each spectrum, so the original class wouldn't work
    #here.
    def paper_plot_small(self,z_in,spec_num,smooth=False,err=False,sky=False,scale_sky=False,save=False,rest=True):
        #These look at the data domain and range (wavelength and flux) and
        #find limits for the plot window.
        wobs = np.where((self.lam>=3700)&(self.lam<=10000))[0]
        flux_range = np.amax(self.flux[wobs]) - np.amin(self.flux[wobs])
        flux_pad = float(flux_range)/10
        y_lower = np.amin(self.flux[wobs]) - flux_pad
        y_upper = np.amax(self.flux[wobs]) + flux_pad
        x_lower = np.amin(self.lam)
        x_upper = np.amax(self.lam)

        '''
        For the four following spectra, the rest wavelength of the emission lines
        are taken from Vanden Berk et al. 2001 paper. That paper lists rest
        wavelengths in the lab and observed in spectra. The "official" SDSS list
        uses a blend of these two sources, so we chose lab or observed based on
        whichever SDSS chose.

        The following if/elif/else chain just builds the line_table array for
        the user-selected spectrum.
        '''

        #This is the FeLoBAL spectrum, Fig. 10.
        if spec_num==10:
            name_str = r'\textbf{SDSS J235134.38+031757.6, z = 2.230}'
            line_table = np.zeros(6,dtype=[('NAME','U26'),('LAM_REST','f8'),('LAM_OBS','f8'),('X_DISP','f8'),('Y_DISP','f8')])
            line_table['NAME'][0],line_table['LAM_REST'][0] = r'\textbf{Ly}$\alpha$',1216.25
            line_table['NAME'][1],line_table['LAM_REST'][1] = r'\textbf{N V}',1239.85
            line_table['NAME'][2],line_table['LAM_REST'][2] = r'\textbf{Si IV+O IV]}',1398.33
            line_table['NAME'][3],line_table['LAM_REST'][3] = r'\textbf{C IV}',1546.15
            line_table['NAME'][4],line_table['LAM_REST'][4] = r'\textbf{C III]}',1905.97
            line_table['NAME'][5],line_table['LAM_REST'][5] = r'\textbf{Mg II}',2800.26
            for i in range(6):
                #From the defined rest wavelength and given redshift, find the
                #observed wavelength.
                line_table['LAM_OBS'][i] = line_table['LAM_REST'][i] * (1+ z_in)
                #The exact calculated observed wavelength likely won't be in the data
                #so we need to find the closest point that is included.
                idx = np.argmin(np.absolute(line_table['LAM_OBS'][i] - self.lam)) #
                line_table['X_DISP'][i] = self.lam[idx] #The selected wavelength for the line
                line_table['Y_DISP'][i] = self.flux[idx] #The flux at that wavelength (hopefully the peak)

        #This is the BAL example spectrum, Fig. 11
        elif spec_num==11:
            name_str = r'\textbf{SDSS J003713.64+241121.5, z = 3.495}'
            line_table = np.zeros(6,dtype=[('NAME','U26'),('LAM_REST','f8'),('LAM_OBS','f8'),('X_DISP','f8'),('Y_DISP','f8')])
            line_table['NAME'][0],line_table['LAM_REST'][0] = r'\textbf{Ly}$\beta$',1033.03
            line_table['NAME'][1],line_table['LAM_REST'][1] = r'\textbf{Ly}$\alpha$',1216.25
            line_table['NAME'][2],line_table['LAM_REST'][2] = r'\textbf{O I}',1305.42
            line_table['NAME'][3],line_table['LAM_REST'][3] = r'\textbf{Si IV+O IV]}',1398.33
            line_table['NAME'][4],line_table['LAM_REST'][4] = r'\textbf{C IV}',1546.15
            line_table['NAME'][5],line_table['LAM_REST'][5] = r'\textbf{C III]}',1905.97
            for i in range(6):
                line_table['LAM_OBS'][i] = line_table['LAM_REST'][i] * (1+ z_in)
                idx = np.argmin(np.absolute(line_table['LAM_OBS'][i] - self.lam))
                line_table['X_DISP'][i] = self.lam[idx]
                line_table['Y_DISP'][i] = self.flux[idx]

        #This is the Mg II BAL spectrum, Fig. 12
        elif spec_num==12:
            name_str = r'\textbf{SDSS J212627.22+012321.0, z = 0.944}'
            line_table = np.zeros(8,dtype=[('NAME','U26'),('LAM_REST','f8'),('LAM_OBS','f8'),('X_DISP','f8'),('Y_DISP','f8')])
            line_table['NAME'][0],line_table['LAM_REST'][0] = r'\textbf{Fe II}',2626.92
            line_table['NAME'][1],line_table['LAM_REST'][1] = r'\textbf{Mg II}',2800.26
            line_table['NAME'][2],line_table['LAM_REST'][2] = r'\textbf{[O II]}',3729.66
            line_table['NAME'][3],line_table['LAM_REST'][3] = r'\textbf{H}$\delta$',4102.73
            line_table['NAME'][4],line_table['LAM_REST'][4] = r'\textbf{H}$\gamma$',4346.42
            line_table['NAME'][5],line_table['LAM_REST'][5] = r'\textbf{H}$\beta$',4853.13
            line_table['NAME'][6],line_table['LAM_REST'][6] = r'\textbf{[O III]}',4960.36
            line_table['NAME'][7],line_table['LAM_REST'][7] = r'\textbf{[O III]}',5008.22
            for i in range(8):
                line_table['LAM_OBS'][i] = line_table['LAM_REST'][i] * (1+ z_in)
                idx = np.argmin(np.absolute(line_table['LAM_OBS'][i] - self.lam))
                line_table['X_DISP'][i] = self.lam[idx]
                line_table['Y_DISP'][i] = self.flux[idx]

        #This is the LyA forest spectrum, Fig. 13
        elif spec_num==13:
            name_str = r'\textbf{SDSS J161016.71+411753.7, z = 5.005}'
            line_table = np.zeros(6,dtype=[('NAME','U26'),('LAM_REST','f8'),('LAM_OBS','f8'),('X_DISP','f8'),('Y_DISP','f8')])
            line_table['NAME'][0],line_table['LAM_REST'][0] = r'\textbf{Ly Limit}',912
            line_table['NAME'][1],line_table['LAM_REST'][1] = r'\textbf{Ly}$\beta$',1033.03
            line_table['NAME'][2],line_table['LAM_REST'][2] = r'\textbf{Ly}$\alpha$',1216.25
            line_table['NAME'][3],line_table['LAM_REST'][3] = r'\textbf{O I}',1305.42
            line_table['NAME'][4],line_table['LAM_REST'][4] = r'\textbf{Si IV+O IV]}',1398.33
            line_table['NAME'][5],line_table['LAM_REST'][5] = r'\textbf{C IV}',1546.15
            for i in range(6):
                line_table['LAM_OBS'][i] = line_table['LAM_REST'][i] * (1+ z_in)
                idx = np.argmin(np.absolute(line_table['LAM_OBS'][i] - self.lam))
                line_table['X_DISP'][i] = self.lam[idx]
                line_table['Y_DISP'][i] = self.flux[idx]


        #Do the "make it pretty" stuff
        matplotlib.rc('font',size=11)
        matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
        fig1,ax1 = plt.subplots(figsize=(10,4))
        #We want the top and bottom x-axis to be wavelength, but different
        #reference frames (rest and observed respectively). Twiny() does this.
        ax1T = ax1.twiny() #This will be the rest wavelength x-axis object
        if smooth==True: #If smoothed, plot the smoothed on top of the raw flux.
            self.bflux,self.berr = self.boxcar(self.flux,self.ivar,10,weighted=True)
            ax1.plot(self.lam,self.flux,color='0.70',linewidth=0.8)
            ax1.plot(self.lam,self.bflux,color='black',linewidth=0.6)
        else:
            ax1.plot(self.lam,self.flux,color='black') #Or just raw flux if unsmoothed.
        if sky==True: #We can also plot the sky spectrum in green, if selected.
            if scale_sky==True:
                #Sky flux is usually much greater than the source flux, so we
                #can choose to scale it. The sky flux is less important than the
                #shape of the sky spectrum (to find patterns), so scaling is useful.
                self.sky = self.data['sky'] / 10.0
            else:
                self.sky = self.data['sky']
            ax1.plot(self.lam,self.sky,color='green',linewidth=0.7,alpha=0.5)
        if err==True: #And we can also plot the error spectrum in red if selected
            ax1.plot(self.lam,self.ferr,color='red',linewidth=0.6)
        ax1.set_xlim((x_lower,x_upper)) #Observerd wavelength x-axis
        ax1T.set_xlim((x_lower/(1+z_in),x_upper/(1+z_in))) #Rest wavelength x-axis
        ax1.set_ylim((y_lower,y_upper)) #The y-axis doesn't change.
        ax1.set_xlabel(r'\textbf{Observed Frame Wavelength (\AA)}')
        ax1T.set_xlabel(r'\textbf{Rest Frame Wavelength (\AA)}')
        ax1.set_ylabel(r'$f_{\lambda}$ ($10^{-17}$ \textbf{ergs s}$^{-1}$ \textbf{cm}$^{-2}$\,\textbf{\AA}$^{-1}$)')
        #The emission lines defined in the tables above need to be plotted with
        #marker lines and text. Because emission line redshift and quasar
        #redshift are always the same due to physical processes, these lines
        #needed to be hand-modified until they point to the right thing.
        #EXCEPT in Fig. 10. These lines are plotted to demonstrate how FeLoBAL
        #features do NOT align with common emission lines.
        if spec_num==10:
            ax1.annotate(line_table['NAME'][0],xy=(line_table['X_DISP'][0],line_table['Y_DISP'][0]),
                         xytext=(-10,30),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            #N V (line_table[1]) is usually blended with LyA, so the labels overlapped
            #ax1.annotate(line_table['NAME'][1],xy=(line_table['X_DISP'][1],line_table['Y_DISP'][1]),
                         #xytext=(-31,60),textcoords='offset points',
                         #bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         #arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][2],xy=(line_table['X_DISP'][2],line_table['Y_DISP'][2]),
                         xytext=(-31,60),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][3],xy=(line_table['X_DISP'][3],line_table['Y_DISP'][3]),
                         xytext=(-12,25),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][4],xy=(line_table['X_DISP'][4]+10,line_table['Y_DISP'][4]+1),
                         xytext=(-13,30),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][5],xy=(line_table['X_DISP'][5],line_table['Y_DISP'][5]+0.5),
                         xytext=(-14,50),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            #annotate calls mark the lines. The text call is what puts the
            #quasar name ('SDSS JHH:MM:SS.SS+DD:MM:SS.S) on the spectrum
            ax1.text(0.01,0.95,name_str,transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0))
        elif spec_num==11:
            ax1.annotate(line_table['NAME'][0],xy=(line_table['X_DISP'][0],line_table['Y_DISP'][0]),
                         xytext=(-10,60),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            #The +35,+2 is the hand-modifying part, so the label matches the observed
            #position of the line in the plot. LyA and C IV are commonly shifted
            #a small amount from the value calculated using the "host" redshift
            ax1.annotate(line_table['NAME'][1],xy=(line_table['X_DISP'][1]+35,line_table['Y_DISP'][1]+2),
                         xytext=(-10,18),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][2],xy=(line_table['X_DISP'][2],line_table['Y_DISP'][2]),
                         xytext=(-8,20),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][3],xy=(line_table['X_DISP'][3],line_table['Y_DISP'][3]),
                         xytext=(-31,60),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][4],xy=(line_table['X_DISP'][4],line_table['Y_DISP'][4]-2),
                         xytext=(-12,25),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][5],xy=(line_table['X_DISP'][5]+10,line_table['Y_DISP'][5]),
                         xytext=(-13,20),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.text(0.6,0.95,name_str,transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0))
        elif spec_num==12:
            ax1.annotate(line_table['NAME'][0],xy=(line_table['X_DISP'][0],line_table['Y_DISP'][0]),
                         xytext=(-11,18),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][1],xy=(line_table['X_DISP'][1]+10,line_table['Y_DISP'][1]+6),
                         xytext=(-14,20),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][2],xy=(line_table['X_DISP'][2],line_table['Y_DISP'][2]),
                         xytext=(-14,20),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            #H delta (line_table[3]) should be visible in the wavelength range,
            #but the flux was too small to see (looks like noise)
            #ax1.annotate(line_table['NAME'][3],xy=(line_table['X_DISP'][3],line_table['Y_DISP'][3]),
                         #xytext=(-31,60),textcoords='offset points',
                         #bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         #arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][4],xy=(line_table['X_DISP'][4],line_table['Y_DISP'][4]),
                         xytext=(-7,25),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][5],xy=(line_table['X_DISP'][5]+22,line_table['Y_DISP'][5]+2),
                         xytext=(-8,30),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate('',xy=(line_table['X_DISP'][6],line_table['Y_DISP'][6]),
                         xytext=(0,89),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][7],xy=(line_table['X_DISP'][7],line_table['Y_DISP'][7]-2.5),
                         xytext=(-16,25),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.text(0.01,0.95,name_str,transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0))
        elif spec_num==13:
            ax1.annotate(line_table['NAME'][0],xy=(line_table['X_DISP'][0],line_table['Y_DISP'][0]),
                         xytext=(-22,30),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][1],xy=(line_table['X_DISP'][1]+35,line_table['Y_DISP'][1]+2),
                         xytext=(-10,18),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][2],xy=(line_table['X_DISP'][2]-10,line_table['Y_DISP'][2]+2),
                         xytext=(-10,20),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][3],xy=(line_table['X_DISP'][3],line_table['Y_DISP'][3]),
                         xytext=(-8,30),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][4],xy=(line_table['X_DISP'][4]-30,line_table['Y_DISP'][4]+0.5),
                         xytext=(-31,45),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.annotate(line_table['NAME'][5],xy=(line_table['X_DISP'][5],line_table['Y_DISP'][5]+0.5),
                         xytext=(-12,30),textcoords='offset points',
                         bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0),
                         arrowprops=dict(arrowstyle='-',connectionstyle='arc3,rad=0',color='blue'))
            ax1.text(0.01,0.95,name_str,transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='square,pad=0.2',fc='magenta',alpha=0.0))
        #Ticks point inward in physics and astronomy
        #And minor ticks should be displayed
        ax1.tick_params(axis='both',direction='in')
        ax1.tick_params(axis='both',which='minor',direction='in')
        ax1.tick_params(right=True)
        ax1.tick_params(which='minor',right=True)
        ax1T.tick_params(direction='in')
        ax1T.tick_params(which='minor',direction='in')
        ax1.xaxis.set_minor_locator(ticker.MultipleLocator(100))
        ax1T.xaxis.set_minor_locator(ticker.MultipleLocator(50))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        #Show or save?
        if save==True:
            fname_out = self.infile.replace('.fits','.eps')
            fname_out = fname_out.replace('../data/','../plots/')
            fig1.savefig(fname_out,bbox_inches='tight',pad_inches=0.03,format='eps')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

# There are a lot of options here. Use python spec_plot.py -h to see them.
# This automatically generates the name of the eps plot using the spectrum file
# name.
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot one of the spectra in Lyke et al. 2020')
    parser.add_argument('p', type=int, choices=[10,11,12,13], help='Paper figure number')
    parser.add_argument('-e', '--error', action='store_true',
                        help='Include the error spectrum in red')
    parser.add_argument('-k', '--sky', action='store_true',
                        help='Include the sky spectrum in green')
    parser.add_argument('-x', '--scale_sky', action='store_true',
                        help='Scale the sky flux to the quasar flux')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save the plot, without display')

    args = parser.parse_args()

    #Set up the plate-mjd-fiberid combinations with redshifts.
    spec_to_plot = np.zeros(4,dtype=[('FIG_NUM','i2'),('PMF','U16'),('Z','f8')])
    spec_to_plot['FIG_NUM'][0],spec_to_plot['PMF'][0],spec_to_plot['Z'][0] = 10,'11278-58395-0576',2.23
    spec_to_plot['FIG_NUM'][1],spec_to_plot['PMF'][1],spec_to_plot['Z'][1] = 11,'7672-57339-0394',3.495
    spec_to_plot['FIG_NUM'][2],spec_to_plot['PMF'][2],spec_to_plot['Z'][2] = 12,'9162-58040-0354',0.944
    spec_to_plot['FIG_NUM'][3],spec_to_plot['PMF'][3],spec_to_plot['Z'][3] = 13,'8528-57896-0104',5.005
    '''
    NOTE: The spectrum for 8528-57896-0104 is not the PRIMARY record in DR16Q. That's listed under: 6044-56090-0418
          The spectra, however, are pretty identical.
    NOTE: The spectrum for 11278-58395-0576 is not the PRIMARY record in DR16Q. That's listed under: 8741-57390-0450
          The spectra is identical, but that record has a Z_VI of 2.43. We are going to trust Vivek at Z = 2.23
    '''
    #This chooses which of the four spectra to use, then grabs the redshift.
    w = np.where(spec_to_plot['FIG_NUM']==args.p)[0]
    red = spec_to_plot['Z'][w]
    #Make the spectrum filename.
    spec_name = '../data/spec-{}.fits'.format(spec_to_plot['PMF'][w[0]])
    spec = spectrum(spec_name) #Load the spectrum using the class functionality.
    #Use the class attribute for spectra to make a 10x4 plot.
    spec.paper_plot_small(red,args.p,smooth=True,err=args.error,sky=args.sky,
                        scale_sky=args.scale_sky,save=args.save)
