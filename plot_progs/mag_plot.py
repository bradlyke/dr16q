"""
This creates the plot that is used as Fig. 7 in Lyke et al. 2020.

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
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rc('text',usetex=True)
import progressBar_fancy as pbf

def mk_plot(infile,fname,fntsize,write_check):
    dr = fits.open(infile)[1].data
    #Set the plot paramaters for more readable numbers.
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    #We only want to plot quasars that had a detected at the 1-sigma level.
    fluxsig = dr['PSFFLUX'][:,3]*np.sqrt(dr['PSFFLUX_IVAR'][:,3])
    #Find quasars with reasonable absolute magnitudes [-30,-18], a reasonable
    #redshift [0,5], and a 1-sigma detection (flux*sqrt(ivar)>1.0)
    w = np.where((dr['M_I']<=-18)&(dr['M_I']>=-30)&(dr['Z_PCA']>=0)&
        (dr['Z_PCA']<=5.0)&(fluxsig>1.0))[0]

    #After the cut, how many quasars are we plotting?
    print('Num Sources: ',len(w))

    #Set up the bins manually for the absolute magnitude.
    mag_binleft = np.arange(-30.0,-18.0,0.01) #define left bin edges
    mag_binright = np.arange(-29.99,-17.99,0.01) #define right bin edges
    mag_arr = np.zeros((len(mag_binleft),2),dtype='float32') #magnitude bin edges array
    mag_arr[:,0] = mag_binleft
    mag_arr[:,1] = mag_binright
    num_mags = len(mag_binleft)

    #Do the same for redshift.
    z_binleft = np.arange(0,5.0,0.01)
    z_binright = np.arange(0.01,5.01,0.01)
    z_arr = np.zeros((len(z_binleft),2),dtype='float32')
    z_arr[:,0] = z_binleft
    z_arr[:,1] = z_binright
    num_zs = len(z_binleft)

    #Now make the plot.
    fig,ax = plt.subplots(figsize=(5,4))

    #2d histograms for this need a user-defined colorbar so that a bin populated
    #with 0 quasars is not black. Otherwise the majority of the plot is black
    #with no clearly defined edges. cmin=1 and we can attach the colorbar to h.
    h = ax.hist2d(dr['Z_PCA'][w],dr['M_I'][w],bins=[num_zs,num_mags],range=[[0,5.01],[-30.0,-17.99]],cmin=1,cmap=plt.cm.inferno)
    #viridis,jet, hsv - other color options that look okay.

    z_ticks = np.arange(0,6,1) #Ticks at every integer redshift.
    mag_ticks = np.arange(-30,-17,2) #Ticks at every other integer magnitude.

    ax.set_xlabel(r'\textbf{Z}$_{\textrm{\textbf{pca}}}$')
    ax.set_ylabel(r'\textbf{M}$_{i}$')
    ax.set_xlim([-0.1,5.1])
    ax.set_ylim([-30,-18])
    ax.set_xticks(z_ticks)
    ax.set_yticks(mag_ticks)
    ax.tick_params(axis='both',direction='in')
    ax.tick_params(axis='both',which='minor',direction='in')
    ax.tick_params(top=True,right=True)
    ax.tick_params(which='minor',top=True,right=True)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.invert_yaxis() #Magnitude goes backwards: the more negative the brighter
    cb = plt.colorbar(h[3],ax=ax) #attach colorbar as object so that we can add a label.
    cb.set_label(r'\textbf{Number of quasars, bins:} $\Delta$\textbf{z}, $\Delta$\textbf{M}$_{i}=0.01$',labelpad=-40,x=0.85)

    #Choose whether to save the plot or just display it.
    if write_check==1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#Call from the command line with:
#    python mag_plot.py <DR16Q file name> <plot name> <font size> <save flag>
# Example:
#    python mag_plot.py DR16Q_v3.fits mag_test.eps 11 1
if __name__=='__main__':
    input_file = '../data/{}'.format(sys.argv[1])
    output_plot_name = '../plots/{}'.format(sys.argv[2])
    plot_fontsize = int(sys.argv[3])
    save_check = int(sys.argv[4])
    mk_plot(input_file,output_plot_name,plot_fontsize,save_check)
