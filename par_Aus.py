#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_Aus.py

This runs a section as specified of assumed block size

"""
import sys
import os
import datetime
import time
import logging
import numpy as np
from fcc import fccModel
from io_operators import observations, output_handler
from kernels import Kernels
from KSw0_vNIR import *
from pb import Pb_MC, PB2, PB3, PB4
import argparse
import textwrap as _textwrap
import pylab as plt

class MultilineFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        paragraphs = text.split('|n ')
        multiline_text = ''
        for paragraph in paragraphs:
            formatted_paragraph = _textwrap.fill(paragraph, width, initial_indent=indent,
                                                    subsequent_indent=indent) + '\n\n'
            multiline_text = multiline_text + formatted_paragraph
        return multiline_text


def mkdate(datestr):
    return datetime.datetime.strptime(datestr, '%Y-%m-%d')


logoTxt = """\
██████╗  █████╗ ██████╗  █████╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗
██████╔╝███████║██║  ██║███████║
██╔══██╗██╔══██║██║  ██║██╔══██║
██████╔╝██║  ██║██████╔╝██║  ██║
╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝"""


print logoTxt


helpTxt = """
BADA version 0.1
|n
----------------
|n
This a first implementation of the algorithm for uncertainty characterised
BA retrieval.
"""


if __name__ == "__main__":

    """
    get options
    """
    parser = argparse.ArgumentParser(description=helpTxt,  formatter_class=MultilineFormatter)
    parser.add_argument('ymin',  type=int,
                        help='ymin of the processing')
    parser.add_argument('xmin', type=int,
                        help='xmin of the processing')
    parser.add_argument('--ymax',  type=int, default=None,
                        help='ymax of the processing')
    parser.add_argument('--xmax',  type=int, default=None,
                        help='xmax of the processing')
    parser.add_argument('--tile', default='h30v10')
    parser.add_argument('--start_date', type=mkdate, default='2008-03-05')
    parser.add_argument('--end_date', type=mkdate, default='2008-07-03')
    parser.add_argument('--outdir', dest='outdir',
                        default=None,
                        help="""specify a custom output directory for the files.
                             Otherwise a sub-directory determined by the tile. """)
    options = parser.parse_args()

    """
    specify extents
    """
    tile = options.tile
    y0 = options.ymin
    x0 = options.xmin
    x1 = options.xmax
    y1 = options.ymax
    if options.ymax == None:
        y1 = y0 + 120
    if options.xmax == None:
        x1 = x0 + 120
    # force limit just incase
    x1 = np.minimum(x1, 2400)
    y1 = np.minimum(y1, 2400)
    xs = x1 - x0
    ys = y1 - y0
    # and dates
    date_0 = options.start_date
    date_1 = options.end_date
    analysis_length = (date_1 - date_0).days
    """
    If the analysis length is less than 120 increase to 120
    """
    #import pdb; pdb.set_trace()
    if analysis_length < 120:
        date_1 += datetime.timedelta(int((120-analysis_length)/2))
        date_0 -= datetime.timedelta(int((120-analysis_length)/2))
        analysis_length = (date_1 - date_0).days
    # convert to strings too
    doy0 = date_0.strftime("%j")
    doy1 = date_1.strftime("%j")

    """
    Sort out logging file
    """
    logger = logging.getLogger(__name__)
    syslog = logging.StreamHandler()
    formatter = logging.Formatter('%(extra)s %(asctime)s - %(message)s')
    #syslog.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    #logger.addHandler(syslog)
    # create a file handler
    handler = logging.FileHandler('aus_log.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)


    # extra info into the logger
    extra = {'extra':'[%s | t0-t1: %s %s | xy: %i %i %i %i]' % (tile, doy0, doy1, y0, x0, y1, x1)}
    logger = logging.LoggerAdapter(logger, extra)


    # record logging
    logger.info("Initiated processing. Load observations...")

    """
    Load observations
    """
    obs = observations( tile = tile,
                        xmin = x0,
                        ymin = y0,
                        xmax = x1,
                        ymax = y1,
                        zero_date = date_0,
                        analysis_length = analysis_length ,
                        pre_wing_days = 16,
                        post_wing_days = 16)

    logger.info("Loaded observations")
    """
    make some local storage

    -- figure out the necessary size of the arrays

    """
    nT = analysis_length + 16




    state = -999*np.ones((nT, 7, ys, xs))
    state_unc = -999* np.ones((nT, 7,  ys, xs))
    iters = -999 *np.ones((ys, xs))
    """
    Run BRDF correction algorithm
    """
    t0 = time.time()
    for y in xrange(ys):
        for x in xrange(xs):
            qa = obs.MOD09.data['qa'][:, y, x]
            if qa.sum() < 20:
                pass
            else:
                k = KSw_vNIR(obs.MOD09.data['date'],
                    obs.MOD09.data['qa'][:, y, x],
                    obs.MOD09.data['refl'][:, :, y, x],
                    obs.MOD09.data['kernels'].Isotropic[:, y, x],
                    obs.MOD09.data['kernels'].Ross[:, y, x],
                    obs.MOD09.data['kernels'].Li[:, y, x])
                k._prepare_matrices()
                k._do_initial_conditions()
                k.getEdges()
                k.solve()
                state[:, :, y, x] = k.xs[:, ::3]#
                iters[y, x] = k.itera
                if x == (xs- 1):
                   logging.info("Done a row...")
                for band in xrange(7):
                    state_unc[:, band, y,x] = k.Cs[:, 3*band:(3*band+3), :][:, 0, 0]
    t1 = time.time()
    logger.info("BRDF correction took %f seconds" % (t1-t0))

    #import pdb; pdb.set_trace()
    """
    try new model
    """
    pb2 = -999*np.ones((nT, ys, xs))
    fcc_params = -999*np.ones((nT, 3, ys, xs))
    a0 = -999*np.ones((nT, ys, xs))
    a1 = -999*np.ones((nT, ys, xs))

    t0 = time.time()
    for y in xrange(ys):
        for x in xrange(xs):
            for t in xrange(0, nT-2):
                pre_iso = state[t, :, y, x]
                post_iso = state[t+2, :, y, x]
                pre_unc =  state_unc[t, :, y, x]
                post_unc =  state_unc[t+2, :, y, x]
                try:
                    #import pdb; pdb.set_trace()
                    _pb, _fcc, _a0, _a1 = PB2(pre_iso, post_iso, pre_unc, post_unc)
                    pb2[t+2, y, x] = _pb
                    fcc_params[t+2, 0, y, x] = _fcc
                    fcc_params[t+2, 1, y, x] = _a0
                    fcc_params[t+2, 2, y, x] = _a1
                except:
                    pb2[t+2, y, x] = -998
                    fcc_params[t+2, 0, y, x] =  -998
                    fcc_params[t+2, 1,y, x] =  -998
                    fcc_params[t+2, 2,y, x] =  -998

    t1 = time.time()
    logger.info("fcc/pb calculation took %f seconds" % (t1-t0))
    #import pdb; pdb.set_trace()


    # also put into a subdirectory based on dates for multiple tiles...
    logger.info("Going to write files...")

    odir = '/home/users/jbrennan01/DATA2/BADA/ausTest/%s/' % (tile)
    if not os.path.exists(odir):
        os.makedirs(odir)

    """
    We only want to write the max pb outcome
    and the fcc, a0, a1 values for these...
    """

    pbMax = np.nanmax(pb2, axis=0)
    pbMax = np.nanmax(pb2, axis=0)
    cc = np.ma.array(data=pb2, mask=~np.isfinite(pb2) )
    idx = cc.argmax(axis=0)
    fcc_parB = -999*np.ones((3, ys, xs))
    for y in xrange(ys):
        for x in xrange(xs):
           fcc_parB[:, y, x] = fcc_params[idx[y,x], :, y, x]

    np.save(odir+"%s_iters_%i_%i_%s_%s" % (tile, y0, x0, doy0, doy1), iters)
    np.save(odir+"%s_fcc_%i_%i_%s_%s" % (tile, y0, x0, doy0, doy1), fcc_parB)
    np.save(odir+"%s_pb2_%i_%i_%s_%s" % (tile, y0, x0, doy0, doy1), pbMax)
    # finish logging
    logger.info("Files written. BADA finished. ")
