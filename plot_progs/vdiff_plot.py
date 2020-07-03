"""
This creates TWO plots used in Fig. 5 of in Lyke et al. 2020. The first is a
histogram of redshift failures broken down by Z_SOURCE as a function of the
redshift difference in velocity space. The second is a scatter plot of
catastrophic redshift failures (delta_v > 3000 km/s) as a function of redshift.

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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rc('text',usetex=True)
import progressBar_fancy as pbf
import sys
import cat_tools as ct

#This calculates the relative redshift difference in velocity space, where zt
#is the "true" redshift.
def zdiff(z1,zt):
    ckms = 299792.458 #speed of light in km/s
    dv = (np.abs(z1-zt)/(1+zt))*ckms
    return dv

#This is the Median Absolute Deviation (MAD) for the table that outputs to the
#terminal. This is used in vdiff_plot().
def mad(data):
    dmed = np.median(data)
    madder = np.median(np.abs(data - dmed))
    return madder

#This generates the lefthand histogram plot
def vdiff_plot(infile,fname,fntsize,write_check):
    dr = fits.open(infile)[1].data
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    #We need only the quasars with a good PCA redshift AND a good Z column redshift.
    w = np.where((dr['Z_PCA']>0)&(dr['Z_PCA']<=5.0)&(dr['Z']>0)&(dr['Z']<=5.0))[0]

    #Break down the good quasars by the source of the redshift in Z.
    w7s = np.where(dr['SOURCE_Z'][w]=='DR7QV_SCH')[0]
    w7h = np.where(dr['SOURCE_Z'][w]=='DR6Q_HW')[0]
    w12 = np.where(dr['SOURCE_Z'][w]=='DR12QV')[0]
    wvi = np.where(dr['SOURCE_Z'][w]=='VI')[0]
    wpipe = np.where(dr['SOURCE_Z'][w]=='PIPE')[0]

    #Calculate the redfshift difference for all good quasars of each Z_SOURCE.
    #The set breakdown is used in Table 4 of the paper to characterize
    #catastrophic redshift failures for each source. zdiff() is defined above.
    vdp7s = zdiff(dr['Z_DR7Q_SCH'][w[w7s]],dr['Z_PCA'][w[w7s]])
    vdp7h = zdiff(dr['Z_DR6Q_HW'][w[w7h]],dr['Z_PCA'][w[w7h]])
    vdp12 = zdiff(dr['Z_DR12Q'][w[w12]],dr['Z_PCA'][w[w12]])
    vdpv = zdiff(dr['Z_VI'][w[wvi]],dr['Z_PCA'][w[wvi]])
    vdpp = zdiff(dr['Z_PIPE'][w[wpipe]],dr['Z_PCA'][w[wpipe]])
    vdp_total = zdiff(dr['Z'][w],dr['Z_PCA'][w])

    #The next few blocks are for making Table 4.
    #Yes, I know the "copy/paste" method is bad practice, but I just needed
    #some numbers quickly, not optimally.
    #Count all of the quasars in each set and in the total.
    #"Sample Size" column
    num7s = len(vdp7s)
    num7h = len(vdp7h)
    num12 = len(vdp12)
    numvi = len(vdpv)
    numpp = len(vdpp)
    num_total = num7s + num7h + num12 + numvi + numpp

    #Only count the number of quasars with acceptable redshift differences
    #where delta_v < 3000.
    #in each set (and the total).
    #"Acceptable delta_v" column
    zg7s = len(np.where(vdp7s<3000)[0])
    zg7h = len(np.where(vdp7h<3000)[0])
    zg12 = len(np.where(vdp12<3000)[0])
    zgvi = len(np.where(vdpv<3000)[0])
    zgpp = len(np.where(vdpp<3000)[0])
    zg_total = zg7s + zg7h + zg12 + zgvi + zgpp

    #What NUMBER were bad?
    #"Catastrophic Failures" column
    zb7s = num7s - zg7s
    zb7h = num7h - zg7h
    zb12 = num12 - zg12
    zbvi = numvi - zgvi
    zbpp = numpp - zgpp
    zb_total = zb7s + zb7h + zb12 + zbvi + zbpp

    #What percentage was bad?
    #"Catastrophic Failure Rate" column
    pct7s = (zb7s/num7s)*100
    pct7h = (zb7h/num7h)*100
    pct12 = (zb12/num12)*100
    pctvi = (zbvi/numvi)*100
    pctpp = (zbpp/numpp)*100
    pct_total = (zb_total/num_total)*100

    #Find the median delta_v for each set.
    #"Median delta_v" column
    med7s = np.median(vdp7s)
    med7h = np.median(vdp7h)
    med12 = np.median(vdp12)
    medvi = np.median(vdpv)
    medpp = np.median(vdpp)

    #Find the Median Absolute Deviation (MAD) for each set.
    #"MAD delta_v" column.
    mad7s = mad(vdp7s)
    mad7h = mad(vdp7h)
    mad12 = mad(vdp12)
    madvi = mad(vdpv)
    madpp = mad(vdpp)

    #Do median and MAD for the total as well.
    med_total = np.median(vdp_total)
    mad_total = mad(vdp_total)

    #Print the calculated values to the terminal.
    print()
    print(' SOURCE   |   MEDIAN   |   Median Absolute Deviation')
    print('----------------------------------------------------')
    print(' DR7Q_S   |   {:5.1f}    |   {:5.1f}'.format(med7s,mad7s))
    print(' DR6Q_H   |   {:5.1f}    |   {:5.1f}'.format(med7h,mad7h))
    print(' DR12Q    |   {:5.1f}    |   {:5.1f}'.format(med12,mad12))
    print('   VI     |   {:5.1f}    |   {:5.1f}'.format(medvi,madvi))
    print('  PIPE    |   {:5.1f}    |   {:5.1f}'.format(medpp,madpp))
    print('----------------------------------------------------')
    print('  TOTAL   |   {:5.1f}    |   {:5.1f}'.format(med_total,mad_total))

    print('\n')
    print(' SOURCE   |   TOTAL   |   NUM_GOOD   |   PBAD    |   NUM_BAD   ')
    print('---------------------------------------------------------------')
    print(' DR7Q_S   |  {:,d}   |   {:,d}     |   {:5.2f}%  |   {:,d}'.format(num7s,zg7s,pct7s,zb7s))
    print(' DR6Q_H   |  {:,d}   |   {:,d}     |   {:5.2f}%  |   {:,d}'.format(num7h,zg7h,pct7h,zb7h))
    print(' DR12Q    |  {:,d}       |   {:,d}         |   {:5.2f}%  |   {:,d}'.format(num12,zg12,pct12,zb12))
    print('   VI     |  {:,d}  |   {:,d}    |   {:5.2f}%  |   {:,d}'.format(numvi,zgvi,pctvi,zbvi))
    print('  PIPE    |  {:,d}  |   {:,d}    |   {:5.2f}%  |   {:,d}'.format(numpp,zgpp,pctpp,zbpp))
    print('---------------------------------------------------------------')
    print('  TOTAL   |  {:,d}  |   {:,d}    |   {:5.2f}%  |   {:,d}'.format(num_total,zg_total,pct_total,zb_total))


    #Now we make the histogram plot. Stacked so each is visible.
    fig,ax = plt.subplots(figsize=(5,4))
    pvi = ax.hist(vdpv[np.where(vdpv<10600)[0]],bins=50,histtype='stepfilled',
            color='darkslateblue',alpha=1.0,
            label=r'$Z_{\textrm{\footnotesize{VI}}}-Z_{\textrm{\footnotesize{PCA}}}$')
    ppi = ax.hist(vdpp[np.where(vdpp<10600)[0]],bins=50,histtype='stepfilled',
            color='black',alpha=1.0,
            label=r'$Z_{\textrm{\footnotesize{PIPE}}}-Z_{\textrm{\footnotesize{PCA}}}$')
    p7h = ax.hist(vdp7h[np.where(vdp7h<10600)[0]],bins=50,histtype='stepfilled',
            color='maroon',alpha=1.0,
            label=r'$Z_{\textrm{\footnotesize{HW}}}-Z_{\textrm{\footnotesize{PCA}}}$')
    p7s = ax.hist(vdp7s[np.where(vdp7s<10600)[0]],bins=50,histtype='stepfilled',
            color='chocolate',alpha=1.0,
            label=r'$Z_{\textrm{\footnotesize{SCH}}}-Z_{\textrm{\footnotesize{PCA}}}$')
    p12 = ax.hist(vdp12[np.where(vdp12<10600)[0]],bins=50,histtype='stepfilled',
            color='purple',alpha=1.0,
            label=r'$Z_{\textrm{\footnotesize{12Q}}}-Z_{\textrm{\footnotesize{PCA}}}$')
    ax.set_xlabel(r'$\Delta$\textbf{v (km s}$^{-1}$\textbf{)}')
    ax.set_ylabel(r'\textbf{Number of quasars}')
    ax.set_yscale('log') #We can't plot this on a linear y-axis
    ax.tick_params(axis='both',direction='in')
    ax.tick_params(top=True,right=True)
    ax.tick_params(which='minor',axis='x',direction='in')
    ax.tick_params(which='minor',top=True)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1000)) #minor tick every 1000 km/s
    ax.legend(loc='upper right',facecolor='white',edgecolor='white')

    if write_check == 1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#This generates the righthand scatter plot.
def vdz(infile,fname,fntsize,write_check):
    dr = fits.open(infile)[1].data
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    #Find the quasars with reasonable redshifts in Z_PCA and Z.
    w = np.where((dr['Z_PCA']>0)&(dr['Z_PCA']<=5.0)&(dr['Z']>0)&(dr['Z']<=5.0))[0]
    #Find the redshift difference for the full set.
    dv = zdiff(dr['Z'][w],dr['Z_PCA'][w])

    #Make the scatter plot. Note the scaling done on dv to make the y-axis
    #more readable.
    fig,ax = plt.subplots(figsize=(5,4))
    ax.scatter(dr['Z_PCA'][w],dv/1000,marker='.',color='black',s=10)
    ax.set_xlabel(r'\textbf{Redshift}')
    ax.set_ylabel(r'$\Delta$\textbf{v (}$10^{3}$ \textbf{km s}$^{-1}$\textbf{)}')
    ax.set_xlim([0,5.0])
    ax.set_ylim([3,10.6])
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_yticks([4,6,8,10])
    ax.tick_params(axis='both',direction='in')
    ax.tick_params(axis='both',which='minor',direction='in')
    ax.tick_params(top=True,right=True)
    ax.tick_params(which='minor',top=True,right=True)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    #Write or show?
    if write_check == 1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#Call from the command line with:
#    python vdiff_plot.py <DR16Q file name> <plot name histo> <plot name scatter> <font size> <save flag>
# Example:
#    python vdiff_plot.py DR16Q_v3.fits vdiff.eps vdiff_vs_z.eps 11 1
if __name__=='__main__':
    input_file = '../data/{}'.format(sys.argv[1])
    output_plot_name_h = '../plots/{}'.format(sys.argv[2])
    output_plot_name_s = '../plots/{}'.format(sys.argv[3])
    plot_fontsize = int(sys.argv[4])
    save_check = int(sys.argv[5])
    vdiff_plot(input_file,output_plot_name_h,plot_fontsize,save_check)
    vdz(input_file,output_plot_name_s,plot_fontsize,save_check)
