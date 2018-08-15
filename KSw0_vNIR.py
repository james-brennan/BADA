"""
Speed ups for the original per-pixel kalman smoother

This version solves iteratively only the NIR band
and then refines the others based on the w for the NIR

"""
import numpy as np
from kernels import *
import scipy.linalg
import glob
import datetime
from numba import jit


"""
These are numba compiled utility functions
"""
@jit(nopython=True)
def _detCov(Cov):
    """
    spatial determinant of cov in space...
    """
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = Cov.ravel()
    DET  = m1*m5*m9 + m4*m8*m3 + m7*m2*m6 - m1*m6*m8 - m3*m5*m7 - m2*m4*m9
    return DET

@jit(nopython=True)
def _inverseCov(Cov, outArr):
    """
    This does a inverse of a set of spatial 3x3 cov matrices
    Adapted from
    https://stackoverflow.com/questions/42489310/
    """
    # reshape it
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = Cov.ravel()
    outArr[0,0]= m5*m9-m6*m8
    outArr[0,1]= m3*m8-m2*m9
    outArr[0,2]= m2*m6-m3*m5
    outArr[1,0]= m6*m7-m4*m9
    outArr[1,1]= m1*m9-m3*m7
    outArr[1,2]= m3*m4-m1*m6
    outArr[2,0]= m4*m8-m5*m7
    outArr[2,1]=m2*m7-m1*m8
    outArr[2,2]= m1*m5-m2*m4
    """
    Now need the 3x3 det
    """
    DET  = m1*m5*m9 + m4*m8*m3 + m7*m2*m6 - m1*m6*m8 - m3*m5*m7 - m2*m4*m9
    return outArr/DET


class KSw_vNIR(object):
    def __init__(self, dates, qa, rho, isoK, volK, geoK):
        """
        This version just does the iteration on the NIR
        and then applies the w to other bands

        """
        # convert dates to doys
        self.dates = dates
        self.doy = np.array([int(d.strftime("%j")) for d in self.dates])
        self.rho = rho
        self.qa = qa
        self.qa = np.logical_and(self.qa, self.rho[:, 5]>0)
        self.isoK = isoK
        self.volK = volK
        self.geoK = geoK
        """
        make life easier by keeping only one ob per day
        eventually write this to do all obs per day...
        """
        #import pdb; pdb.set_trace()
        _, keep = np.unique(self.dates, return_index=True)
        self.doy = self.doy[keep]
        self.dates = self.dates[keep]
        self.qa = self.qa[keep]
        self.isoK = self.isoK[keep]
        self.volK = self.volK[keep]
        self.geoK = self.geoK[keep]
        self.rho = self.rho[keep]

        # adjust fake doy
        self.year = np.array([int(d.strftime("%Y")) for d in self.dates])
        self.year -= self.year.min()
        # this should approximately work...
        self.doy = self.year * 365 + self.doy
        self.doy -= self.doy.min()


    def _prepare_matrices(self):
        """
        This produces all the necessary matrix storage
        to run the smoother
        """
        # Dimensions information
        self.nT  = self.doy.max() + 1
        self.nB  = 7
        self.nK = 3
        # Prior estimates
        self.x_p = np.zeros((self.nT,  self.nB * self.nK))
        self.C_p = np.zeros((self.nT,self.nB * self.nK, 3))
        # Posterior estimates
        self.x = np.zeros((self.nT,self.nB * self.nK))
        self.C = np.zeros((self.nT,self.nB * self.nK, 3))
        """
        And for the smoothed estimate
        """
        self.xs = np.zeros((self.nT, self.nB * self.nK))
        self.Cs = np.zeros((self.nT, self.nB * self.nK, 3))
        """
        edge preserving functional
        """
        self.w = np.ones(self.nT)
        """
        Specify the observation error covariance
        """
        self.C_obs  = ([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003])
        self.R = np.diag(self.C_obs)
        """
        specify the observation operator matrix (eg the kernels)
        """
        self.H = np.zeros((self.nT, self.nK, ))
        self.H[:, 0 ] = self.isoK
        self.H[:, 1 ] = self.volK
        self.H[:, 2 ] = self.geoK

    def _do_initial_conditions(self):
        """
        Produce the initial condition estimates
        This involves iterating
        over each pixel to produce an initial BRDF inversion
        """
        # number of obs used to produce initial estimate
        n0 = 16
        qa  = self.qa[:]
        """
        make kernel matrix
        """
        KK = np.vstack(( self.isoK[qa][:n0],
                         self.volK[qa][:n0],
                         self.geoK[qa][:n0] )).T
        xx = []
        cc = []
        for band in xrange(self.nB):
            # get band uncertainty
            unc = self.C_obs[band]
            """
            Do solution
            """
            rho_b = self.rho[qa, band][:n0]
            _x = np.linalg.solve(KK.T.dot(KK)/(unc),
                                    KK.T.dot(rho_b)/(unc))
            # and cov matrix
            _c = np.linalg.inv(KK.T.dot(KK)/(unc))
            xx.append(_x)
            cc.append(_c)
        """
        Add the initial condition to time-step 1
        """
        #import pdb; pdb.set_trace()
        self.x[0, :] = np.hstack(xx)
        self.C[0, :] = np.vstack(cc)

    def getEdges(self):
        """
        *-- Run the iterative edge preserving filter --*

        This runs the kalman smoother several times and updates
        the edge process weighting

        BUT JUST on the NIR band
        """
        # localised
        rho = self.rho
        qa = self.qa
        doy = self.doy
        """
        Some constants
        """
        TOL = 1e-2
        MAX_ITER = 10
        """
        This controls convergence for different pixels
        eg if a pixel has converged we change the flag and therefore
        do no more edge preserving on this pixel
        """
        converged = False
        err_0 = 1e10
        err_1 = (1e10-1)
        SSE = 0.0
        # The process model
        alphaIso = 0.94 # 0.94
        alphaGV =  0.97 # 0.94
        # This is the shared process model for all pixels
        self.Qq = np.ones((3,3 ))
        self.Qq[0,0]=1/alphaIso
        self.Qq[1,1]=1/alphaGV
        self.Qq[2,2]=1/alphaGV
        self.itera = 0
        I21_3 = [np.eye(3)]

        # This stores an inverse for the backward step
        invC = np.ones((3, 3))

        C_nir = self.C[:, 3:6,:]
        x_nir = self.x[:, 3:6]
        x_p_nir = self.x_p[:, 3:6]
        C_p_Nir = self.C_p[:, 3:6,:]
        rho_nir = self.rho[:, 1]
        Cs_nir = self.Cs[:, 3:6,:]
        xs_nir = self.xs[:, 3:6]


        xs_0 = np.copy(xs_nir[:, 0])*0.0


        self.badDets = np.zeros(self.nT).astype(np.bool)

        while converged==False and self.itera < MAX_ITER:
            # reset the SSE counter
            SSE *= 0.0
            self.itera +=1
            #print iter
            """
            run the fwd kalman filter
            """
            for t in xrange(doy.min()+1, doy.max()+1):
                """
                The prediction of x is the same as the
                previous time-step which corresponds to
                a 0-th order process model
                """
                #import pdb; pdb.set_trace()
                x_t = x_nir[t-1]
                C_t = C_nir[t-1]
                # apply the stand process mode noise
                C_t = C_t * self.Qq
                """
                Apply the egde preserving functional
                do band by band as it seems easier...

                For pixels that have already converged stop
                these being edge preserved..

                """
                # force w back to 1 in these cases
                C_t[0, 0] *= self.w[t]
                # save prior estimates
                x_p_nir[t] = x_t
                C_p_Nir[t]= C_t
                """
                Check whether we have an observation on this
                day for each pixel

                If we have no observation for this day
                the prior becomes the posterior...
                """
                if ~self.qa[t]:
                    x_nir[t]= x_t
                    C_nir[t]= C_t
                else:
                    """
                    Now for the other pixels
                    we want to perform the Kalman update equations
                    """
                    # Get the observation operator
                    Ht = self.H[t]
                    # Predict reflectance
                    pred = Ht.dot(x_t)# (Ht * x_t).sum()
                    """
                    Get innovation error
                    """
                    residual = rho_nir[t] - pred
                    """
                    Update sum of squared errors
                    """
                    SSE+= (residual**2)
                    """
                    The innovation covariance matrix S
                    usual
                    S =  R + H C H^T
                    equ. R + ((H * C.T).sum(axis=1) * H).sum()
                    Do band by band
                    """
                    HC = np.matmul(Ht, C_t.T)
                    HCHT = HC.dot(Ht.T)
                    S = self.C_obs[1] + HCHT
                    """
                    'invert' S
                    """
                    invS = 1.0/S # wooh!
                    """
                    Kalman Gain!
                    K = C H invS
                    """
                    K = C_t.dot(Ht.T) * invS
                    """
                    update
                    """
                    x_up = x_t + (K * residual)
                    x_nir[t] = x_up
                    _c = (I21_3 - np.outer(K, Ht)).dot(C_t)
                    C_nir[t] =  _c
                    """
                    check the determinant
                    """
                    dett = _detCov(_c)
                    if (dett <= 0):
                        C_nir[t] = C_nir[t-1]
                        self.badDets[t]=True
            """
            *-- Now do the RTS smoother! --*
            """
            Cs_nir[-1]=C_nir[-1]
            xs_nir[-1]=x_nir[-1]

            """
            *-- prototype outlier --*

            Just compute from the smoothed
            estimate the z-score residual
            """
            self.z_score = np.zeros(self.nT)

            for t in np.arange(doy.min(), doy.max())[::-1]:
                C_p_t1 = C_p_Nir[t+1]
                C_tt = C_nir[t]
                """
                Need the inverse of np.linalg.inv(C_p_t1)
                """
                invC = _inverseCov(C_p_t1, invC)
                #invC = np.linalg.inv(C_p_t1)
                K_t = C_tt.dot(invC)
                """
                calculate adjustment
                """
                xo = x_nir[t]
                xst_1 = xs_nir[t+1]
                xp_t = x_p_nir[t+1]
                diff = xst_1 - xp_t
                adjustment = K_t.dot(diff)
                _x_t = xo + adjustment
                """
                and covariance update
                """
                _Cs = Cs_nir[t+1]
                diff = C_p_t1 - _Cs
                _C_t = C_tt - K_t.dot(diff).dot(K_t.T)
                xs_nir[t] = _x_t
                Cs_nir[t] = _C_t
                # outlier thing
                self.z_score[t] =  (self.H[t].dot(_x_t) - self.rho[t, 1])**2/self.C_obs[1]
            """
            Calculate edge-preserving functional
            """
            # note we don't use 1 over here but increase the unc instead...
            self.w[:] =  (1 + 4e3 * np.gradient(xs_nir[:, 0])**2)
            # update error
            err_0 = err_1
            err_1 = SSE
            # check per-pixel convergence
            #converged = np.abs(SSE - err_0) < TOL
            """
            different convergence...
            has the solution stopped changing much?
            [description]
            """
            #import pdb; pdb.set_trace()
            converged =  ( (np.abs(xs_nir[:, 0] - xs_0)).sum() < TOL)
            # propagate around the solution for next iteration
            xs_0 = np.copy(xs_nir[:, 0])
            """
            update the outlier mask for the qa...

            For now we specify a simple outlier model.

            An observation is an outlier if the zscore
            exceeds a 95\% threshold in the z_score values
            """
            if self.itera <= 2:
                outliersT = np.nanpercentile(self.z_score[self.qa], 98)
                outliers = self.z_score > outliersT
                # update the qa
                self.qa = np.logical_and(self.qa, ~outliers)


    def solve(self):
        """
        Now with the edge process calculated
        we can run the solver on the other bands
        with the specified w vector
        """
        # localised
        rho = self.rho
        qa = self.qa
        doy = self.doy

        alphaIso = 0.94 # 0.94
        alphaGV =  0.97 # 0.94
        # This is the shared process model for all pixels
        self.Qq = np.ones((3,3 ))
        self.Qq[0,0]=1/alphaIso
        self.Qq[1,1]=1/alphaGV
        self.Qq[2,2]=1/alphaGV
        self.Qq = np.vstack([self.Qq]*7)
        I21_3 = np.vstack([np.eye(3)]*7)
        xs_0 = np.copy(self.xs[:, ::3])*0.0

        # This stores an inverse for the backward step
        invC = np.ones((3, 3))

        """
        run the fwd kalman filter
        """
        for t in xrange(doy.min()+1, doy.max()+1):
            """
            The prediction of x is the same as the
            previous time-step which corresponds to
            a 0-th order process model
            """
            x_t = self.x[t-1]
            C_t = np.copy(self.C[t-1])
            # apply the stand process mode noise
            C_t = C_t[:, :] * self.Qq[:, :]
            """
            Apply the egde preserving functional
            do band by band as it seems easier...

            For pixels that have already converged stop
            these being edge preserved..

            """
            # force w back to 1 in these cases
            C_t[0, 0]  *= self.w[t]
            C_t[3, 0]  *= self.w[t]
            C_t[6, 0]  *= self.w[t]
            C_t[9, 0]  *= self.w[t]
            C_t[12, 0] *= self.w[t]
            C_t[15, 0] *= self.w[t]
            C_t[18, 0] *= self.w[t]
            # save prior estimates
            self.x_p[t] = x_t
            self.C_p[t]= C_t
            """
            Check whether we have an observation on this
            day for each pixel

            If we have no observation for this day
            the prior becomes the posterior...
            """
            if ~self.qa[t]:
                self.x[t]= x_t
                self.C[t]= C_t
            else:
                """
                Now for the other pixels
                we want to perform the Kalman update equations
                """
                # Get the observation operator
                Ht = self.H[t]
                # Predict reflectance
                pred = (Ht*x_t.reshape((self.nB, self.nK))).sum(axis=1)
                """
                Get innovation error
                """
                residual = self.rho[t] - pred
                """
                The innovation covariance matrix S
                usual
                S =  R + H C H^T
                equ. R + ((H * C.T).sum(axis=1) * H).sum()
                Do band by band
                """
                HC = np.matmul(Ht, C_t.T)
                HCHT = HC.reshape((7, 3)).dot(Ht.T)
                S = self.C_obs + HCHT
                """
                'invert' S
                """
                invS = 1.0/S # wooh!
                """
                Kalman Gain!
                K = C H invS
                """
                K = []
                for band in xrange(7):
                    K.append(C_t[3*band:(3*band+3), :].dot(Ht.T) * invS[band])
                K = np.array(K)
                """
                update
                """
                x_up = x_t + (K * residual[:, None]).ravel()
                self.x[t] = x_up
                brack = (I21_3 - np.outer(K, Ht))
                for band in xrange(7):
                    _c = brack[3*band:(3*band+3), :].dot(C_t[3*band:(3*band+3), :])
                    #C_.append(_)
                    self.C[t, 3*band:(3*band+3), :] = _c
                """
                check the determinant
                """
                if self.badDets[t]:
                    self.C[t] = self.C[t-1]
        """
        *-- Now do the RTS smoother! --*
        """
        self.Cs[-1]=self.C[-1]
        self.xs[-1]=self.x[-1]
        for t in np.arange(doy.min(), doy.max())[::-1]:
            for band in np.arange(7):
                C_p_t1 = self.C_p[t+1, 3*band:(3*band+3), :]
                C_tt = self.C[t,  3*band:(3*band+3), :]
                """
                Need the inverse of np.linalg.inv(C_p_t1)
                """
                invC = _inverseCov(C_p_t1, invC)
                #invC = np.linalg.inv(C_p_t1)
                K_t = C_tt.dot(invC)
                """
                calculate adjustment
                """
                xo = self.x[t,  3*band:(3*band+3)]
                xst_1 = self.xs[t+1, 3*band:(3*band+3)]
                xp_t = self.x_p[t+1, 3*band:(3*band+3)]
                diff = xst_1 - xp_t
                adjustment = K_t.dot(diff)
                _x_t = xo + adjustment
                """
                and covariance update
                """
                _Cs = self.Cs[t+1, 3*band:(3*band+3), :]
                diff = C_p_t1 - _Cs
                _C_t = C_tt - K_t.dot(diff).dot(K_t.T)
                self.xs[t, 3*band:(3*band+3)] = _x_t
                self.Cs[t, 3*band:(3*band+3), :] = _C_t
        #import pdb; pdb.set_trace()