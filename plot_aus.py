"""
plot_aus
"""
import glob
import os
import numpy as np
import gdal
import copy
base_dir = '/group_workspaces/cems2/nceo_generic/users/jbrennan01/BADA/ausTest/'
auTiles = np.genfromtxt("AusTiles.txt", dtype=str)
auTiles.sort()

tile_datas = []
for tile in auTiles:
    # load data
    # change to dir
    data = np.zeros((2400, 2400))
    try:
        _files =glob.glob(base_dir+tile+'/'+"*iters*npy")
        for f in _files:
            ar = np.load(f)
            y0 = int(f.split("_")[-4])
            x0 = int(f.split("_")[-3])
            y1 = y0 + ar.shape[0]
            x1 = x0 + ar.shape[1]
            data[y0:y1, x0:x1]=ar
    except:
        pass
    print tile

    """
    save as geotiff

    [description]
    """
    try:
        prDir = "/home/users/jbrennan01/DATA2/BADA/priors/%s/" % tile
        inf = gdal.Open(prDir+"%s_fcc_means.tif"%tile)
        oldGeo =  inf.GetGeoTransform()
        newGeo = list(copy.deepcopy(oldGeo))
        totmetres = 463.3127165279165 * 2400
        cell_metres = totmetres / 2400
        newGeo[1]=cell_metres
        newGeo[-1]=-cell_metres
        newGeo = tuple(newGeo)

        outdir = base_dir

        # create the files
        dst_filename = "%s_iter.tif" % tile
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            outdir+dst_filename,
            2400,
            2400,
            1,
            gdal.GDT_Float32, )
        dataset.SetGeoTransform(newGeo)
        dataset.SetProjection(inf.GetProjection())
        dataset.GetRasterBand(1).SetNoDataValue(-999)

        dataset.GetRasterBand(1).WriteArray(data)
        dataset.FlushCache()  # Write to disk.
        dataset = None
    except:
        pass


        """
        run these commands on the shell voila!
        gdalbuildvrt pb.vrt /home/users/jbrennan01/DATA2/BADA/ausTest/*pb*tif
        gdalbuildvrt fcc.vrt /home/users/jbrennan01/DATA2/BADA/ausTest/*fcc*tif
        gdaldem color-relief /home/users/jbrennan01/DATA2/BADA/ausTest/pb.vrt cols.txt pb2.png -of PNG
        gdaldem color-relief /home/users/jbrennan01/DATA2/BADA/ausTest/fcc.vrt  cols.txt fcc2.png -of PNG

        !gdalbuildvrt iters.vrt /home/users/jbrennan01/DATA2/BADA/ausTest/*iter*tif
        !gdaldem color-relief /home/users/jbrennan01/DATA2/BADA/ausTest/iters.vrt  cols.txt iter.png -of PNG


        """
