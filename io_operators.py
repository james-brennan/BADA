"""
IO.py

Assorted code for handling IO of observations...

Given the m month processing idea this will probably be rather
complex. But can be aided by making lots of vrt files for:

1. MODIS reflectance
2. MODIS active fires
3. VIIRS etc reflectance too...

each organise by year and doy.

"""
import numpy as np
from kernels import *
import glob
import gdal
import h5py
import datetime

"""
*-- MODIS related IO functions --*
"""
class MODIS_refl(object):
    """
    This is a class to hold and load MODIS reflectance data
    """
    def __init__(self, tile, start_date, end_date, xmin, ymin, xmax, ymax):
        self.tile = tile
        self.start_date = start_date
        self.end_date = end_date
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def apply_qa(self, qa):
        QA_OK = np.array([8, 72, 136, 200, 1032, 1288, 2056, 2120, 2184, 2248])
        qa2 = np.in1d(qa, QA_OK).reshape(qa.shape)
        return qa2

    def vrt_loader(self, tile, date, xmin, ymin, xmax, ymax):
        """
        this loads a
        """
        vrt_dir = '/home/users/jbrennan01/mod09_vrts/'
        dire = vrt_dir+tile+'/'
        # get the right band
        # load it
        yr = date.year
        xsize = xmax-xmin
        ysize = ymax-ymin
        data = {}
        files = ["brdf_%s_%s_b01.vrt" % (yr, tile ), "brdf_%s_%s_b02.vrt" % (yr, tile),
                 "brdf_%s_%s_b03.vrt" % (yr, tile), "brdf_%s_%s_b04.vrt" % (yr, tile),
                 "brdf_%s_%s_b05.vrt" % (yr, tile), "brdf_%s_%s_b06.vrt" % (yr, tile),
                 "brdf_%s_%s_b07.vrt" % (yr, tile),
                 "statekm_%s_%s.vrt" % (yr, tile),
                 "SensorAzimuth_%s_%s.vrt" % (yr, tile),
                 "SensorZenith_%s_%s.vrt" % (yr, tile),
                 "SolarAzimuth_%s_%s.vrt" % (yr, tile),
                 "SolarZenith_%s_%s.vrt" % (yr, tile),]
        dNames = ['brdf1', 'brdf2', 'brdf3', 'brdf4', 'brdf5',
                  'brdf6', 'brdf7', 'qa', 'vaa', 'vza', 'saa', 'sza', ]
        qainfo = gdal.Open(dire+"statekm_%s_%s.vrt" % (yr, tile))
        doy = np.array([int(qainfo.GetRasterBand(b+1).GetMetadataItem("DoY")) for b in xrange(qainfo.RasterCount)])
        year_doy = np.array([int(qainfo.GetRasterBand(b+1).GetMetadataItem("Year")) for b in xrange(qainfo.RasterCount)])
        dates = np.array([datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1) for year, days in zip(year_doy, doy)])
        sens = np.array([qainfo.GetRasterBand(b+1).GetMetadataItem("Platform") for b in xrange(qainfo.RasterCount)])
        # select correct date
        #import pdb; pdb.set_trace()
        idx = np.where(dates==date)[0]+1 # add 1 for GDAL
        # load these bands
        for nm, p in zip(dNames, files):
            datastack = []
            for band in idx:
                pp = gdal.Open(dire+p)
                data_p = pp.GetRasterBand(band)
                data_ = data_p.ReadAsArray(xoff=xmin, yoff=ymin, win_xsize=xsize, win_ysize=ysize)
                datastack.append(data_)
                #print nm, p, band
            data[nm]=np.array(datastack)
        data['dates'] = dates[idx-1]
        data['sensor']=sens[idx-1]
        return data

    def loadData(self):
        """
        Loads the modis refl and active fires
        for the time-span
        """
        # cba re-writing...
        tile = self.tile
        xmin =self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        xs = xmax - xmin 
        ys = ymax - ymin
        beginning = self.start_date
        ending = self.end_date
        # figure what dates we need
        ndays = (ending-beginning).days
        dates = np.array([beginning + datetime.timedelta(days=x) for x in range(0, ndays)])
        """
        for a date
        load the necessary band
        """
        datas = {}
        # add stuff
        datas['qa'] = []
        datas['refl'] = []
        datas['date'] = []
        datas['vza'] = []
        datas['sza'] = []
        datas['raa'] = []
        datas['sensor']=[]
        ida = 0
        n = len(dates)
        for date in dates:
            data = self.vrt_loader(tile, date, xmin, ymin, xmax, ymax)
            # do qa
            qa = self.apply_qa(data['qa'])
            datas['qa'].append(qa)
            refl = np.stack(([data['brdf%i' % b] for b in xrange(1,8)]))
            refl = np.swapaxes(refl, 0,1 )
            refl = refl.astype(float)
            refl *= 0.0001
            """
            Fix aqua band
            """
            band6 = refl[:, 5]>0.0
            qa = np.logical_and(qa, band6)
            datas['date'].append(data['dates'])
            """
            fix mask errors
            due to band6
            """
            datas['sensor'].append(data['sensor'])
            datas['refl'].append( refl )
            datas['vza'].append( data['vza']*0.01 )
            datas['sza'].append( data['sza']*0.01 )
            datas['raa'].append( (data['vaa']*0.01 - data['saa']*0.01).astype( np.float32 ))
            ida += 1
        # fix structure
        datas['refl'] = np.vstack(datas['refl'])
        datas['qa'] = np.vstack(datas['qa'])
        datas['vza'] = np.vstack(datas['vza']).astype(float)
        datas['sza'] = np.vstack(datas['sza']).astype(float)
        datas['raa'] = np.vstack(datas['raa']).astype(float)
        datas['date'] = np.hstack(datas['date'])
        datas['sensor'] = np.hstack(datas['sensor'])

        """
        Make kernels too
        """
        ts = datas['vza'].shape[0]
        kerns =  Kernels(datas['vza'], datas['sza'], datas['raa'],
                        LiType='Sparse', doIntegrals=False,
                        normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                        RossType='Thick',nbar=0.0)
        kerns.Ross = kerns.Ross.reshape((ts, ys, xs))
        kerns.Li = kerns.Li.reshape((ts, ys, xs))
        kerns.Isotropic = kerns.Isotropic.reshape((ts, ys, xs))
        datas['kernels'] = kerns
        self.data = datas
        return None







class observations(object):
    """
    The idea of this class is to store some time-space block of observations
    which the algorithm uses.

    This should therefore make the propagation through time easier
    because this can just be updated each n months with the new obs etc

    """
    def __init__(self,
                tile = 'h30v10',
                xmin = 0,
                ymin = 0,
                xmax = 240,
                ymax = 240,
                zero_date=None,
                analysis_length=120,
                    pre_wing_days=60, post_wing_days=60):
        """
        """
        self.tile = tile
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.zero_date = zero_date
        self.analysis_length = analysis_length
        self.pre_wing_days = pre_wing_days
        self.post_wing_days = post_wing_days
        # calculate pre wing
        self.wing_pre = zero_date - datetime.timedelta(pre_wing_days)
        self.wing_post = None
        # start date
        self.stat_date = zero_date
        # calculate end date
        self.end_date = zero_date + datetime.timedelta(analysis_length)
        self.first = True

        """
        Load the MODIS observations
        """
        self.MOD09 = MODIS_refl(self.tile, self.wing_pre,
                                self.end_date, self.xmin,
                                self.ymin, self.xmax, self.ymax)
        self.MOD09.loadData()

    def advance(self):
        """
        this tells the observations to advance to the next analysis
        window
        """
        # slide the window along
        self.start_date += datetime.timedelta(self.analysis_length)
        self.end_date += datetime.timedelta(self.analysis_length)
        """
        Do we want a pre or post wing for extra obs?
        """
        if not self.first:
            self.wing_pre = None
            self.wing_post =  self.end_date + datetime.timedelta(post_wing_days)



def create_outfile(tile, ):
    AUXDIRECTORY = "/group_workspaces/cems2/nceo_generic/users/jbrennan01/BADA/aux/"
    """
    make dirs
    """
    outfile = h5py.File( AUXDIRECTORY +'/'+'%s_BRDFstate_.hdf5' % (tile),'w')
    ks = outfile.create_dataset("state", (136, 7, 3, 2400, 2400), dtype="f4", chunks=(136, 7, 3, 256, 256))
    ks_unc = outfile.create_dataset("unc", (136, 7, 3, 2400, 2400), dtype="f4", chunks=(136, 7, 3, 256, 256))
    file = outfile
    return file, ks, ks_unc


class output_handler(object):
    """
    This class helps the output writing stuff
    eg it creates output files and writes to them
    """
    def __init__(self, tile, zero_date ):
        """
        """
        self.OUTDIRECTORY = "/group_workspaces/cems2/nceo_generic/users/jbrennan01/BADA/outputs/"
        self.AUXDIRECTORY = "/group_workspaces/cems2/nceo_generic/users/jbrennan01/BADA/aux/"
        """
        make dirs
        """
        outfile = h5py.File( self.AUXDIRECTORY +'/'+'%s_BRDFstate_.hdf5' % (tile),'w')
        # and for now store evols
        self.ks = outfile.create_dataset("state", (136, 7, 3, 2400, 2400), dtype="f4", chunks=(136, 7, 3, 256, 256))
        self.ks_unc = outfile.create_dataset("unc", (136, 7, 3, 2400, 2400), dtype="f4", chunks=(136, 7, 3, 256, 256))
        self.file = outfile

    def write(self, state, unc, x0, x1, y0, y1):
        """
        write a chunk of data
        """
        self.ks[:, :, :, y0:y1, x0:x1] = state
        self.ks_unc[:, :, :, y0:y1, x0:x1] = unc


