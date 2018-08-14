#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_par.py

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
from pb import Pb_MC
import argparse

import textwrap as _textwrap
class MultilineFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        paragraphs = text.split('|n ')
        multiline_text = ''
        for paragraph in paragraphs:
            formatted_paragraph = _textwrap.fill(paragraph, width, initial_indent=indent, subsequent_indent=indent) + '\n\n'
            multiline_text = multiline_text + formatted_paragraph
        return multiline_text


def mkdate(datestr):
    return datetime.datetime.strptime(datestr, '%Y-%m-%d')




logoTxt = """\
 _______  _______  ______   _______
|  _    ||   _   ||      | |   _   |
| |_|   ||  |_|  ||  _    ||  |_|  |
|       ||       || | |   ||       |
|  _   | |       || |_|   ||       |
| |_|   ||   _   ||       ||   _   |
|_______||__| |__||______| |__| |__|
"""

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


    parser = argparse.ArgumentParser(description=helpTxt,  formatter_class=MultilineFormatter)
    parser.add_argument('xmin', type=int,
                        help='xmin of the processing')
    parser.add_argument('ymin',  type=int,
                        help='ymin of the processing')
    parser.add_argument('--xmax',  type=int, default=None,
                        help='xmax of the processing')
    parser.add_argument('--ymax',  type=int, default=None,
                        help='ymax of the processing')
    parser.add_argument('--start_date', type=mkdate, default='2008-03-05')
    parser.add_argument('--end_date', type=mkdate, default='2008-07-03')


    parser.add_argument('--outdir', dest='outdir',
                        default=None,
                        help='specify a custom output directory for the files')
    options = parser.parse_args()




    """
    Sort out logging file
    """
    logging.basicConfig(filename="log.log",
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    """
    specify extents
    """
    y0 = options.ymin
    x0 = options.xmin
    x1 = options.xmax
    y1 = options.ymax
    if options.ymax == None:
        y1 = y0 + 20
    if options.xmax == None:
        x1 = x0 + 20
    # force limit just incase
    x1 = np.minimum(x1, 2400)
    y1 = np.minimum(y1, 2400)
    xs = x1 - x0
    ys = y1 - y0

    # and dates

    date_0 = options.start_date
    date_1 = options.end_date
    analysis_length = (date_1 - date_0).days

    logging.info("Processing %i %i" % (y0, x0))

    """
    Load observations
    """
    tile = "h30v10"

    obs = observations( tile=tile,
                        xmin = x0,
                        ymin = y0,
                        xmax = x1,
                        ymax = y1,
                        zero_date = date_0,
                        analysis_length = analysis_length ,
                        pre_wing_days = 16,
                        post_wing_days = 16)

    logging.info("Loaded obs for %i %i" % (y0, x0))
    """
    make some local storage
    """
    state = -999*np.ones((136, 7, ys, xs))
    state_unc = -999* np.ones((136, 7,  ys, xs))
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
    logging.info("That took %f seconds" % (t1-t0))


    """
    And fcc storage
    """
    fcc_params = -999*np.ones((136, 3, ys, xs))
    fcc_uncs = -999*np.ones((136, 3, 3, ys, xs))
    """
    Now do fcc
    """
    t0 = time.time()
    for y in xrange(ys):
        for x in xrange(xs):
            for t in xrange(0, 136-2):
                pre_iso = state[t, :, y, x]
                post_iso = state[t+2, :, y, x]
                pre_unc =  state_unc[t, :, y, x]
                post_unc =  state_unc[t+2, :, y, x]
                try:
                    _fcc, _a0, _a1, _fcc_uncs = fccModel(pre_iso, post_iso, pre_unc, post_unc)
                    fcc_params[t+2, 0, y, x] = _fcc
                    fcc_params[t+2, 1, y, x] = _a0
                    fcc_params[t+2, 2, y, x] = _a1
                    fcc_uncs[t+2, :, :,y, x] = _fcc_uncs
                except:
                    fcc_params[t+2, 0, y, x] = -998.
                    fcc_params[t+2, 1, y, x] = -998.
                    fcc_params[t+2, 2, y, x] = -998.
                    fcc_uncs[t+2, :, :,y, x] = -998.
    t1 = time.time()
    logging.info("fcc calculation took %f seconds" % (t1-t0))


    """
    do pb classifier
    """
    pb = -999*np.ones((136, ys, xs))
    t0 = time.time()
    for y in xrange(ys):
        for x in xrange(xs):
            for t in xrange(0, 136-2):
                fcc_p   = fcc_params[t+2, :, y, x]
                fcc_unc = fcc_uncs[t+2, :, :,y, x]
                if fcc_p[0] == -999 or fcc_p[0] == -998:
                    pb[t+2, y, x] = -999
                else:
                    _pb = Pb_MC(fcc_p, fcc_unc)
                    pb[t+2, y, x] = _pb
        print y
    t1 = time.time()
    logging.info("pb calculation took %f seconds" % (t1-t0))
    import pdb; pdb.set_trace()

    """
    Save outputs
    """
    odir = '/home/users/jbrennan01/DATA2/BADA/tmp/%s/' % tile
    if not os.path.exists(odir):
        os.makedirs(odir)
    np.save(odir+"%s_state_%f_%f" % (tile, y0, x0), state)
    np.save(odir+"%s_iters_%f_%f" % (tile, y0, x0), iters)
    np.save(odir+"%s_fcc_%f_%f" % (tile, y0, x0), fcc_params)
    np.save(odir+"%s_pb_%f_%f" % (tile, y0, x0), pb)
