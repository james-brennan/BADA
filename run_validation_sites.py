"""
run_validation_sites.py

Run the algorithm over the FireCCI validation sites

"""



valDir ="/home/users/jbrennan01/DATA2/RRGlob/masks_marc/"


valSites = glob.glob(valDir+"*.tif")


for site in valSites:

    # get boundaries of the site for the processing extents...
    site_strings = site.split("/")[-1].split("_")
    tile = site_strings[1]
    # get dates
    start_date = datetime.datetime.strptime(site_strings[-3], "%Y%m%d")
    end_date = datetime.datetime.strptime(site_strings[-2], "%Y%m%d")
    # add a bit of border to be sure we can get a prior solution...
    start_date -= datetime.timedelta(32)
    end_date += datetime.timedelta(32)

    # correctly re-format these 
    str_start_date = start_date.strftime("%Y-%m-%d")
    str_end_date = end_date.strftime("%Y-%m-%d")


    # get processing extent
    data = gdal.Open(site).ReadAsArray()[0]
    locs = np.where(data>-1)
    ymin = locs[0].min()
    ymax = locs[0].max()
    xmin = locs[1].min()
    xmax = locs[1].max()

    """
    partition this block up for different jobs
    """
    filenamexy = './xys/xy_val_%s_%s_%s.dat' % (tile, str_start_date, str_end_date)
    myfile = open(filenamexy, 'w')
    ii = 0
    for y0 in xrange(ymin, ymax, 120):
        for x0 in xrange(xmin, xmax, 120):
            x1 = x0 + 120
            y1 = y0 + 120
            x1 = np.minimum(x1, xmax)
            y1 = np.minimum(y1, ymax)
            xs = x1-x0
            ys = y1-y0
            #print y0, y1, x0, x1,
            myfile.write("%i %i %i %i\n" % (y0, x0, y1, x1))
            ii += 1
    myfile.close()

    # lsf
    lsf_tmpl = """#!/bin/bash
#BSUB -J BADA_%s[1-%i]
#BSUB -o ./job_logs/job_%s_%s_%s.o
#BSUB -e ./job_logs/job_%s_%s_%s.e
#BSUB -q short-serial
#BSUB -W 02:45
line=`sed -n ${LSB_JOBINDEX}p < %s`
y0=`awk '{print $1}' <<< $line`
x0=`awk '{print $2}' <<< $line`
y1=`awk '{print $3}' <<< $line`
x1=`awk '{print $4}' <<< $line`
python par_main.py $y0 $x0 --ymax $y1 --xmax $x1 --tile %s --start_date %s --end_date %s"""
    
    print lsf_tmpl % (tile, ii,
        tile, str_start_date, str_end_date,
        tile, str_start_date, str_end_date,
        filenamexy,
     tile, str_start_date, str_end_date)

    myfile = open('run_it.lsf', 'w')
    myfile.write(lsf_tmpl % (tile, ii,
        tile, str_start_date, str_end_date,
        tile, str_start_date, str_end_date,
        filenamexy, tile, str_start_date, str_end_date))
    myfile.close()

    # submit it?
    os.system("bsub < run_it.lsf")
