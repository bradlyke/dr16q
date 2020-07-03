"""
This creates TWO plots used in Fig. 6 of in Lyke et al. 2020. The first is a
histogram of the number of quasars for each SDSS observation campaign (I,II,etc.)
as a function of redshift. The second is a histogram of the number of quasars
in each eBOSS subprogram as a function of redshift.

Dependencies
----------
github repository : https://github.com/bradlyke/utilities
pydl utilities : https://pypi.org/project/pydl/
Note: To get these plots, LaTeX must be installed and
      matplotlib must be able to compile LaTeX commands


Input file
----------
The most recent version of the DR16Q quasar-only catalog.

Parameters
----------
input_file : the file name for the DR16Q FITS file in the
             ../data folder. No path needed.
output_plot_name_z : the name of the histogram plot of SDSS-X written out
output_plot_name_t : the name of the histogram plot of Targets written out
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
from pydl.pydlutils.sdss import sdss_flagval as flagval
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rc('text',usetex=True)
import progressBar_fancy as pbf
import tmark
import sys

#This creates the lefthand plot of all quasars grouped by SDSS campaign.
def zhisto_plot(infile,fname,fntsize,write_check):
    dr = fits.open(infile)[1].data

    #Make the numbers pretty and bold.
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    #Only select quasars with reasonable redshifts. Break them up into sets
    #based on the MJD ranges of the different SDSS campaigns.
    #Making them subsets lets us easily plot the total as well.
    wgood = np.where((sd['Z']<=5.0)&(sd['Z']>=0))[0]
    w2 = np.where(dr['MJD'][wgood]<54663)[0]
    w3 = np.where((dr['MJD'][wgood]>=54663)&(dr['MJD'][wgood]<56898))[0]
    w4 = np.where(dr['MJD'][wgood]>=56898)[0]

    #Establish a standard color set for this set breakdown so that we can
    #be consistent with other plots in the paper.
    sd12color = 'blue'
    sd3color = 'pink'
    sd4color = 'red'

    #Plot each set as a histogram. To make each histogram visible when stacked
    #they need to be in order III, I/II, IV. They are saved to a variable so we
    #can manually set the order in the legend so that it's logically I/II, III, IV.
    fig,ax = plt.subplots(figsize=(5,4))
    p1 = ax.hist(dr['Z'][wgood],bins=50,histtype='step',color='black',label=r'\textbf{Total}')
    p2 = ax.hist(dr['Z'][wgood[w3]],bins=50,histtype='stepfilled',color=sd3color,label=r'\textbf{SDSS-III}')
    p3 = ax.hist(dr['Z'][wgood[w2]],bins=50,histtype='step',color=sd12color,label=r'\textbf{SDSS-I/II}')
    p4 = ax.hist(dr['Z'][wgood[w4]],bins=50,histtype='step',color=sd4color,label=r'\textbf{SDSS-IV}')
    ax.set_xlabel(r'\textbf{Redshift}')
    ax.set_ylabel(r'\textbf{Number of quasars}')
    ax.set_xlim([0,5])
    ax.set_xticks(np.array([0,1,2,3,4,5]))
    ax.tick_params(axis='both',direction='in')
    ax.tick_params(top=True,right=True)
    #Now we need to fix the legend which defaults to out of order.
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[0],handles[2],handles[1],handles[3]]
    labels = [labels[0],labels[2],labels[1],labels[3]]
    ax.legend(handles,labels,loc='upper right',facecolor='white',edgecolor='white')

    #Do you want a file, or just the plot displayed?
    if write_check == 1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#This creates the righthand plot of eBOSS quasars broken down by targeting
#subprogram: CORE, Palomar Transient Factory (PTF), TDSS, SPIDERS
def targ_histo(infile,fname,fntsize,write_check):
    dr = fits.open(infile)[1].data
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    #We use the same redshift cut, but the bitwise and (&) operator doesn't
    #work with the EBOSS_TARGET field AND the redshift cut at the same time.
    wgood = np.where((sd['Z']<=5.0)&(sd['Z']>=0)&(sd['MJD']>=56898))[0]
    #Find the bit flag that corresponds to each from the pydl utility.
    corebit = flagval('EBOSS_TARGET1',['QSO1_EBOSS_CORE'])
    ptfbit = flagval('EBOSS_TARGET1',['QSO1_PTF'])
    tdssbit = flagval('EBOSS_TARGET1',['TDSS_TARGET'])
    spdrbit = flagval('EBOSS_TARGET1',['SPIDERS_TARGET'])

    #Cut down the total set of good redshifts (and MJD) by subprogram.
    wcore = np.where(dr['EBOSS_TARGET1'][wgood]&corebit)[0]
    wptf = np.where(dr['EBOSS_TARGET1'][wgood]&ptfbit)[0]
    wtdss = np.where(dr['EBOSS_TARGET1'][wgood]&tdssbit)[0]
    wspdr = np.where(dr['EBOSS_TARGET1'][wgood]&spdrbit)[0]

    #We don't want to reuse the SDSS-I/II/III/IV color definitions as these
    #are all SDSS-IV/eBOSS quasars.
    corecolor = 'black'
    ptfcolor = 'blue'
    tdsscolor = 'darkorange'
    spdrcolor = 'red'

    #Stack the histograms. The legend for these doesn't have to be in a particular
    #order, so we can use the default order it sets. But we need to stack them
    #this way for maximum readability.
    fig,ax = plt.subplots(figsize=(5,4))
    ax.hist(dr['Z'][wgood[wcore]],bins=50,histtype='step',color=corecolor,label=r'\textbf{CORE}')
    ax.hist(dr['Z'][wgood[wtdss]],bins=50,histtype='stepfilled',color=tdsscolor,alpha=0.6,label=r'\textbf{TDSS}')
    ax.hist(dr['Z'][wgood[wptf]],bins=50,histtype='step',color=ptfcolor,label=r'\textbf{PTF}')
    ax.hist(dr['Z'][wgood[wspdr]],bins=50,histtype='step',color=spdrcolor,label=r'\textbf{SPIDERS}')
    ax.set_xlabel(r'\textbf{Redshift}')
    ax.set_ylabel(r'\textbf{Number of quasars}')
    ax.set_xlim([0,5])
    ax.set_xticks(np.array([0,1,2,3,4,5]))
    ax.tick_params(axis='both',direction='in')
    ax.tick_params(top=True,right=True)
    #Again, we can use the default legend order.
    ax.legend(loc='upper right',facecolor='white',edgecolor='white')

    #Save or show?
    if write_check == 1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#Call from the command line with:
#    python zhisto.py <DR16Q file name> <plot name Z> <plot name T> <font size> <save flag>
# Example:
#    python zhisto.py DR16Q_v3.fits zhisto.eps targhisto.eps 11 1
if __name__=='__main__':
    input_file = '../data/{}'.format(sys.argv[1])
    output_plot_name_z = '../plots/{}'.format(sys.argv[2])
    output_plot_name_t = '../plots/{}'.format(sys.argv[3])
    plot_fontsize = int(sys.argv[4])
    save_check = int(sys.argv[5])
    zhisto_plot(input_file,output_plot_name_z,plot_fontsize,save_check)
    targ_histo(input_file,output_plot_name_t,plot_fontsize,save_check)
