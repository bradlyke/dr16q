"""
This creates the plot that is used as Fig. 1 in Lyke et al. 2020. Matplotlib will
not allow a 2d histogram to be mapped onto a Mollweidge project. We're going
to use numpy to trick it into putting a "colormap" onto the projection.

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
               For a 10x4 plot spanning twocolumn, 11 works best.
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
import astropy.coordinates as coord
import astropy.units as u
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.colors as colors

#This function creates the sky map of observed quasars.
def skyplot_heat(infile,fname,fntsize,write_check):
    dr = fits.open(infile)[1].data

    #Set up the numbering to be more readable.
    matplotlib.rc('font',size=fntsize)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    #Define a new color map so that bins with zero objects are white, but the rest is fine.
    inferno = cm.get_cmap('inferno',256)
    newcolors = inferno(np.linspace(0,1,256))
    white = np.array([1,1,1,1])
    newcolors[0,:] = white
    newcmp = ListedColormap(newcolors)

    #Pull the RA and DEC out.
    #These are stored in decimal degrees. We will set the RA to sexegessimal
    #in the axis label.
    ra = dr['RA']
    dec = dr['DEC']

    #In astronomy, East (or RA) increases to the left. You are looking out, not
    #down, so it needs to be mirrored to retain parity.
    #We also have a choice of where to center the projection. 8h is used in the
    #paper, but 6h also worked.
    #The projection wants to plot -180 to 180 degrees. It can't plot 0 to 360.
    #We need to wrap the data and force the axes labels/shift to match.

    #This will shift the data to plot centered on 8h.
    #ra = ra - 90 #centered on 6h
    ra = ra - 120 #centered on 8h
    wp = np.where(ra<0)[0]
    ra[wp] = 360 + ra[wp] #

    #This will wrap the shifted RA values to get -180 to 180
    w = np.where(ra>180)[0]
    ra[w] = -(360-ra[w])

    #Now we need to use numpy to make the histogram to trick matplotlib.
    #I want nstep bins in each axis.
    nstep = 100
    #Now we need to set up the bins and histogram
    ra_step = (np.amax(ra)-np.amin(ra))/nstep #set the RA bin step size
    dec_step = (np.amax(dec)-np.amin(dec))/nstep #Set the DEC bin step size

    #Make a linear space for the bins based on the above stepsize. You must
    #have the +1 on nstep or you have the wrong number of values.
    ra_bins = np.linspace(np.amin(ra)-ra_step,np.amax(ra)+ra_step,nstep+1)
    dec_bins = np.linspace(np.amin(dec)-dec_step,np.amax(dec)+dec_step,nstep+1)

    #Someone in numpy didn't transpose the 2d histogram in the backend, so we have
    #to do it. This must be in DEC/RA order or it's flipped.
    dd, _, _ = np.histogram2d(dec,ra,[dec_bins,ra_bins]) #THIS HAS TO BE IN DEC/RA ORDER BECAUSE NUMPY IS STUPID

    #And this one has to be in RA/DEC order. Thanks numpy.
    ra2d,dec2d = np.meshgrid(ra_bins,dec_bins)
    #We need to turn the floating degree values in RA/DEC into astropy
    #coordinate objects in degrees.
    ra2d = coord.Angle(ra2d*u.degree)
    ra2d = -ra2d #This does the East/West flip so that East points to the left.
    #Remember, you are looking at a projected map of the sky, not of the Earth.
    dec2d = coord.Angle(dec2d*u.degree)

    #Now we set up the plot as a pcolormesh with each color in the mesh set
    #by the value in the numpy 2d histogram by using the colormap we defined.
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111,projection='mollweide')
    psm = ax.pcolormesh(ra2d.radian,dec2d.radian,dd,cmap=newcmp)

    #We have to manually define the axes labels as we shifted the real data to
    #a different origin. This is because matplotlib won't let you set the origin
    #in a mollweide project. The 6h-centered set of labels is left as a
    #comment in case you want to try something different.
    #ax.set_xticklabels(['16h','14h','12h','10h','8h','6h','4h','2h','0h','22h','20h']) #centered on 6h

    ax.set_xticklabels(['18h','16h','14h','12h','10h','8h','6h','4h','2h','0h','22h']) #centered on 8h
    cb = fig.colorbar(psm,ax=ax,orientation='horizontal') #put a colorbar on the bottom
    cb.set_label(r'\textbf{Number of quasars, bins:} $\Delta \alpha = 3.6^{\circ}$, $\Delta \delta = 1.0^{\circ}$',labelpad=-45,y=0.85)
    ax.grid(True)

    #Do you want to display, or save the file?
    if write_check==1:
        fig.savefig(fname,bbox_inches='tight',pad_inches=0.03,format='eps')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#Call from the command line with:
#    python sky_plot.py <DR16Q file name> <plot name> <font size> <save flag>
# Example:
#    python sky_plot.py DR16Q_v3.fits sky_map.eps 11 1
if __name__=='__main__':
    input_file = '../data/{}'.format(sys.argv[1])
    output_plot_name = '../plots/{}'.format(sys.argv[2])
    plot_fontsize = int(sys.argv[3])
    save_check = int(sys.argv[4])
    skyplot_heat(input_file,output_plot_name,plot_fontsize,save_check)
