import glob 
import gdal
import scipy.interpolate
import sys
import os
import copy 
import numpy as np 
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# old code
#points = np.vstack((data[0], data[1])).T
#values = data[3]
#grid = np.mgrid[0:2400, 0:2400]
#grid = grid.reshape((2, -1)).T
#test = scipy.interpolate.griddata(points, values, grid, method='cubic', )


if __name__ == "__main__":

    tile = sys.argv[1]

    outdir = '/home/users/jbrennan01/DATA2/BADA/priors/%s/' % (tile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # location of the fcc priors
    base_dir = '/group_workspaces/cems2/nceo_generic/users/jlgomezdans/output_tmp/%s/' % tile

    data = []
    for year in xrange(2001, 2015):
        files = glob.glob(base_dir + str(year) +'/' + "fcc.MCD14A1*tif")
        for f in files:
            # open it
            ar = gdal.Open(f).ReadAsArray()[:3]
            """
            What we want to keep is locations and values 
            where fcc is detected etc.
            """
            y, x = np.where(ar[0]>0)
            dt = np.vstack([ y, x,  ar[:, y, x]])
            print f, year 
            data.append(dt)
    data = np.hstack(data)

    # get geo info too
    inf = gdal.Open(f)


    """
    Now try 
    for every 60 pixels
    a unique fcc, a0,a1 prior
    """ 
    yx = np.vstack((data[0], data[1])).T

    means = -999*np.ones((3, 40, 40))
    sums = -999*np.ones((40, 40))
    covs = -999*np.ones((3, 3, 40, 40))

    j = 0
    for y0 in xrange(0, 2400, 60):
        i = 0
        for x0 in xrange(0, 2400, 60):
            x1 = x0 + 60
            y1 = y0 + 60
            x1 = np.minimum(x1, 2400)
            y1 = np.minimum(y1, 2400)
            print y0, x0
            # get fires covered by this...
            """
            probably need to use the nearest fires in some 
            places...
            """ 
            yC = np.logical_and(yx[:, 0] >= y0, yx[:, 0] < y1)
            xC = np.logical_and(yx[:, 1] >= x0, yx[:, 1] < x1)
            idx = np.logical_and(yC, xC)
            local_data = data[:, idx]
            fcc_test = np.logical_and(local_data[2] > 0, local_data[2] < 1.1)
            local_data = local_data[:, fcc_test]
            if local_data.shape[1]>10:

                """
                Do robust covariance
                
                [description]
                """
                robust_cov = MinCovDet().fit(local_data[2:].T)
                covs[:, :, j, i] = robust_cov.covariance_
                means[:, j, i] = robust_cov.location_
                sums[j, i] = robust_cov.support_.sum()
            i+=1
        j+=1

    """
    Write priors to files
    """
    # first figure out the new geotransform
    oldGeo =  inf.GetGeoTransform()
    newGeo = list(copy.deepcopy(oldGeo))
    totmetres = 463.3127165279165 * 2400
    cell_metres = totmetres / 40.0
    newGeo[1]=cell_metres
    newGeo[-1]=-cell_metres
    newGeo = tuple(newGeo)

    # create the files
    dst_filename = "%s_fcc_means.tif" % tile
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        outdir+dst_filename,
        40,
        40,
        3,
        gdal.GDT_Float32, )
    dataset.SetGeoTransform(newGeo)
    dataset.SetProjection(inf.GetProjection())
    dataset.GetRasterBand(1).SetNoDataValue(-999)
    dataset.GetRasterBand(2).SetNoDataValue(-999)
    dataset.GetRasterBand(3).SetNoDataValue(-999)

    dataset.GetRasterBand(1).WriteArray(means[0])
    dataset.GetRasterBand(2).WriteArray(means[1])
    dataset.GetRasterBand(3).WriteArray(means[2])
    dataset.FlushCache()  # Write to disk.
    dataset = None

    # and cov
    covs = covs.reshape((9, 40, 40))
    dst_filename = "%s_fcc_covs.tif" % tile
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        outdir+dst_filename,
        40,
        40,
        9,
        gdal.GDT_Float32, )
    dataset.SetGeoTransform(newGeo)
    dataset.SetProjection(inf.GetProjection())
    re = [dataset.GetRasterBand(i).WriteArray(covs[i-1]) for i in xrange(1,10)]
    re = [dataset.GetRasterBand(i).SetNoDataValue(-999) for i in xrange(1,10)]
    dataset.FlushCache()  # Write to disk.
    dataset = None

    # and sum
    dst_filename = "%s_fcc_nAf.tif" % tile
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        outdir+dst_filename,
        40,
        40,
        1,
        gdal.GDT_Float32, )
    dataset.SetGeoTransform(newGeo)
    dataset.SetProjection(inf.GetProjection())
    dataset.GetRasterBand(1).SetNoDataValue(-999)
    dataset.GetRasterBand(1).WriteArray(sums)
    dataset.FlushCache()  # Write to disk.
    dataset = None
