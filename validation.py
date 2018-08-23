"""
This code computes the validation statistics
and makes nice plots for analysis...
"""
import glob
import gdal
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy

valDir ="/home/users/jbrennan01/DATA2/RRGlob/masks_marc/"
valSites = glob.glob(valDir+"*.tif")


for site in valSites:
    # get boundaries of the site for the processing extents...
    site_strings = site.split("/")[-1].split("_")
    tile = site_strings[1]
    # get dates
    start_date = datetime.datetime.strptime(site_strings[-3], "%Y%m%d")
    end_date = datetime.datetime.strptime(site_strings[-2], "%Y%m%d")
    t0 = copy.deepcopy(start_date)
    t1 = copy.deepcopy(end_date)
    # add a bit of border to be sure we can get a prior solution...
    start_date -= datetime.timedelta(32)
    end_date += datetime.timedelta(32)


    analysis_length = (end_date - start_date).days


    if analysis_length < 120:
        end_date += datetime.timedelta(int((120-analysis_length)/2))
        start_date -= datetime.timedelta(int((120-analysis_length)/2))
        analysis_length = (end_date - start_date).days
    # convert to strings too
    str_start_date = start_date.strftime("%j")
    str_end_date = end_date.strftime("%j")



    # get processing extent
    data = gdal.Open(site).ReadAsArray()[0]
    locs = np.where(data>-1)
    ymin = locs[0].min()
    ymax = locs[0].max()
    xmin = locs[1].min()
    xmax = locs[1].max()

    # limit to this extent
    data = data[ymin:ymax, xmin:xmax]

    """
    Now locate and load the relevant BADA run
    """
    base_dir = "/group_workspaces/cems2/nceo_generic/users/jbrennan01/BADA/tmp/"
    base_dir += "/%s/" % tile
    base_dir += "%s_%s/" %(start_date.strftime("%j"), end_date.strftime("%j"))

    """
    Load Pb
    """

    BADA_files = glob.glob(base_dir+"*pb4*npy")
    #BADA_files = glob.glob("*pb2*npy")
    #print BADA_files
    # fix dates back to Landsat pre and post
    #start_date += datetime.timedelta(32)
    #end_date -= datetime.timedelta(32)

    extra_dates = int((120-analysis_length)/2)

    pb = np.zeros((2400, 2400))

    for f in BADA_files:
        arr = np.load(f)
        """
        The fire must occur between the two dates..
        """
        #import pdb; pdb.set_trace()
        arr = arr[(t0-start_date).days:(t1-end_date).days]
        pp = np.nanmax(arr, axis=0) #arr.max(axis=0)
        y0 = int(f.split("_")[-4])
        x0 = int(f.split("_")[-3])
        y1 = y0 + arr.shape[1]
        x1 = x0 + arr.shape[2]
        pb[y0:y1, x0:x1]=pp

    pb = pb[ymin:ymax, xmin:xmax]


    """
    calculate the stats...

    do a matrix
                    Map Burnt       Map Unbur
    true Burnt          a1              a2
    true Unburnt        a3              a4

    """
    #import pdb; pdb.set_trace()

    a1 = np.logical_and(data>0.05, pb>0.5).sum()
    a2 = np.logical_and(data>0.05, pb<0.5).sum()
    a3 = np.logical_and(data==0.0, pb>0.5).sum()
    a4 = np.logical_and(data==0.0, pb<0.5).sum()

    try:
        Ce = 100 * np.float(a3) / (a1+a3)
        Oe = 100* np.float(a2) / (a2+a4)
    except:
        Ce= -1
        Oe = -1
    """
    plot these two things...
    """
    #import pdb; pdb.set_trace()
    print site.split("/")[-1].strip(".tif"), Oe, Ce, (pb>0.5).sum(), (data>0.05).sum()
    # calculate aspect ratio

    figsize=np.array([arr.shape[0], arr.shape[1]])

    # re-scale by the max to be 12?
    re_scale = 12.0/ figsize.max()
    figsize = re_scale * figsize
    # double x to plot two beside each other
    figsize[0]*=2

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=2)
    ax[0].imshow(data, vmin=0, vmax=1)
    ax[0].axis('off')
    ax[1].imshow(pb, vmin=0, vmax=1)
    ax[1].axis('off')
    plt.tight_layout()

    val_r_dir = "/group_workspaces/cems2/nceo_generic/users/jbrennan01/BADA/validation/"
    fname = "2_%s_%s.png" % (tile, site.split("/")[-1].strip(".tif") )
    plt.savefig(val_r_dir + fname, dpi=200, bbox_inches='tight')
    plt.close()
    #print site_strings
