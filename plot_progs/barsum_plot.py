"""
This creates the plot that is used as Fig. 2 in Lyke et al. 2020.

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
input_file : the file name for the DR16Q FITS file in the
             ../data folder. No path needed.
output_plot_name : the name of the plot written out
plot_fontsize : The fontsize for everything in the plot
               For a 5x4 plot in twocolumn, 11 works best.
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
import tmark
import sys

#This function creates a table of MJDs and the cumulative number of quasars
#observed as of that date. This overcomes the errors that might occur
#when some dates have 0 observations, or aren't represented in DR16Q.
def mjd_maker(dr):
    dr = fits.open(infile)[1].data
    tmark.tm('Making MJD Arrays')
    mjd_min = np.amin(dr['MJD']) #Starting MJD
    mjd_mpo = np.amax(dr['MJD'])+1 #Non-inclusive maximum MJD
    mjd_arr = np.arange(mjd_min,mjd_mpo,1)
    num_mjd = len(mjd_arr)
    
    #This will hold the table of date vs. the cumulative number by that date.
    data = np.zeros(num_mjd,dtype=[('MJD','int32'),('NUM_UNIQUE','int64')])
    data['MJD'] = mjd_arr
    
    #This array holds the first MJD that each quasar in DR16Q was observed on.
    first_mjd = np.zeros(len(sd),dtype='int64')
    first_mjd = sd['MJD']
    
    #MJD can be stored either in the MJD column OR in the MJD_DUPLICATE column
    #as a more recent observation may have been selected as the primary for
    #a given quasar.
    wdupe = np.where(sd['NSPEC']>0)[0] #Find quasars with more than one obs
    
    tmark.tm('Finding Earlier MJDs')
    #For all of the multiply-observed quasars, overwrite the primary obs MJD
    #with the earliest
    for i in range(len(wdupe)):
        mjd_arr_temp = sd['MJD_DUPLICATE'][wdupe[i]]
        wn1 = np.where(mjd_arr_temp>-1)[0]
        mjd_min_temp = np.amin(mjd_arr_temp[wn1])
        if mjd_min_temp < first_mjd[wdupe[i]]:
            first_mjd[wdupe[i]] = mjd_min_temp
        pbf.pbar(i,len(wdupe))
    
    #Find the unique MJDs from the first_mjd array and count them
    tmark.tm('Making Data Array')
    for i in range(num_mjd):
        w = np.where(first_mjd<=data['MJD'][i])[0]
        data['NUM_UNIQUE'][i] = len(w) #Write the count to the output table
        pbf.pbar(i,num_mjd)
    
    #Write the output table of cumulative number of quasars per MJD.
    data_hdu = fits.BinTableHDU.from_columns(data,name='TABLE')
    outname = '../data/unique_mjd.fits'
    data_hdu.writeto(outname)
    return outname
    
#This function makes the plot. The user can choose to save it as an eps
#or have it plot on screen.
def sum_plot(infile,fname,fntsize,write_check):
    data = fits.open(infile)[1].data

    #Set up the plot parameters to be pretty
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
    
    #Set up the X and Y axis ticks.
    max_mjd = np.amax(data['MJD'])
    mjd_ticks = np.array([52000,53000,54000,55000,56000,57000,58000])
    sum_ticks = np.array([0,1,2,3,4,5,6,7])
    min_mjd = np.amin(data['MJD'])
    ydata = data['NUM_UNIQUE']*1e-5
    max_num = np.amax(ydata)
    
    #Color code the plot. Using standard colors means other plots that
    #use the same groups can be standardized across the paper.
    sd12color = 'blue'
    sd3color='pink'
    sd4color='red'
    
    #Making the plot
    fig,ax = plt.subplots(figsize=(5,4))
    ax.plot(data['MJD'],ydata,color='black',linewidth=0.8) #Plot the top line
    #These fill in the area below the plot line for each SDSS observation campaign
    ax.fill_between(data['MJD'],0,ydata,where=data['MJD']<54663,facecolor=sd12color)
    ax.fill_between(data['MJD'],0,ydata,where=data['MJD']>=54663,facecolor=sd3color)
    ax.fill_between(data['MJD'],0,ydata,where=data['MJD']>=56898,facecolor=sd4color)
    #These set up the legend color blocks for labels later.
    p12 = ax.fill(np.NaN,np.NaN,sd12color)
    p3 = ax.fill(np.NaN,np.NaN,sd3color)
    p4 = ax.fill(np.NaN,np.NaN,sd4color)
    #Axes labels, ticks, and tick limits.
    ax.set_xlabel(r'\textbf{Modified Julian Date}')
    ax.set_ylabel(r'\textbf{Cumulative number of quasars (}$10^{5}$\textbf{)}')
    ax.set_xlim([min_mjd,max_mjd])
    ax.set_xticks(mjd_ticks)
    ax.set_ylim([0,max_num])
    ax.set_yticks(sum_ticks)
    ax.tick_params(axis='both',direction='in')
    ax.tick_params(top=True,right=True)
    #This sets up the a legend box with no border or background color with the
    #fill colors attached to the proper labels.
    ax.legend([p12[0],p3[0],p4[0]],[r'\textbf{SDSS-I/II}',r'\textbf{SDSS-III}',r'\textbf{SDSS-IV}'],loc='upper left',facecolor='white',edgecolor='white')

    #Write or view?
    if write_check==1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        
    #YOU HAVE TO PUT THIS INTO TWOCOLUMN AT EPSSCALE 1.18 AND IT'S PERFECT!!!

#Call from the command line with:
#    python barsum_plot.py <DR16Q file name> <plot name> <font size> <save flag>
# Example:
#    python barsum_plot.py DR16Q_v3.fits barsum.eps 11 1

if __name__=='__main__':
    input_file = '../data/{}'.format(sys.argv[1])
    output_plot_name = '../plots/{}'.format(sys.argv[2])
    plot_fontsize = int(sys.argv[3])
    save_check = int(sys.argv[4])
    unique_mjd_filename = mjd_maker(input_file)
    sum_plot(unique_mjd_filename,output_plot_name,plot_fontsize,save_check)
