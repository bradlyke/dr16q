## The Sloan Digitial Sky Survey Quasar Catalog: Sixteenth Data Release (DR16Q)
The python 3 code that can generate parts of DR16Q and the plots used in the attendant paper (Lyke et al. 2020).

##### Dependencies: 

My other repository called "utilities"

##### Folder Descriptions:

- data : This folder should hold the data files input to, or written out by programs in the parent folder.
- plot_progs : The scripts that create the plots used in the DR16Q paper.
- plots : Where the plots written out by the scripts in plot_progs are kept.

### File Descriptions:

#### Main files:
- spAll_class_full.py : The initial script that creates a proto-superset from the spAll file.
- qnet_afterburn.py : The script to reclassify records flagged for visual inspection using the QuasarNET catalog.
- dr16q_duplicate_removal.py : The script that generates the quasar-only catalog from the superset catalog.
- abs_mag.py : The absolute i-band magnitude calculator for DR16Q.
- Coming soon : a recipe script to use the above scripts, in order, to generate DR16Q.

#### Other files:
richards_kcorr_table.dat : K-correction table 4 from [Richards et al. 2006.](https://ui.adsabs.harvard.edu/abs/2006AJ....131.2766R/abstract)
