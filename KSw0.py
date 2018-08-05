"""
first version of Kalman smoother with edge preserving functional...

This can actually be re-written to be really fast because 
the only true inverse covariance matrices are 3x3...
    --> so can do these analytically really easy!

Then other dot products can be done as array multiplications and summations...





"""
import numpy as np
from kernels import *
import scipy.linalg
import glob
%load_ext line_profiler


class KSw(object):
    def __init__(self, doy, qa, rho, kerns):
        self.doy = doy
        self.qa = qa
        self.rho = rho
        self.kerns = kerns


    def solve(self):
        """
        This actually runs the filter to solve it
        """
        # localised
        kerns = self.kerns
        rho = self.rho
        qa = self.qa
        doy = self.doy

        # useful 3x3 det -- faster...
        def det(m):
           m1, m2, m3, m4, m5, m6, m7, m8, m9 = m.ravel()
           return np.dot(m[:, 0], [m5*m9-m6*m8, m3*m8-m2*m9, m2*m6-m3*m5])




        C_obs  = ([0.005, 0.014, 0.008, 0.005, 0.012, 0.006, 0.003])
        R = np.diag(C_obs)

        """
        Get an initial solution
        """
        x0 = []
        c = []
        for band in xrange(7):
            unc = C_obs[band]
            n0 = 16
            KK = np.vstack((kerns.Isotropic[qa][:n0], kerns.Ross[qa][:n0], kerns.Li[qa][:n0])).T
            CK = np.diag(1.0/(1*unc) * np.ones(n0))
            x0.append( np.linalg.solve(KK.T.dot(CK).dot(KK), KK.T.dot(CK).dot(rho[qa, band][:n0])) )
            c.append( (np.linalg.inv(KK.T.dot(CK).dot(KK)))  )


        """
        make into proper matrices etc
        """
        arrs = [c[band].reshape((3,3)) for band in xrange(7)]
        C0 = scipy.linalg.block_diag(*arrs)
        """
        make storage
        """
        # Posterior estimates
        x = np.zeros((doy.max(), 7*3))
        C = np.zeros((doy.max(), 7*3, 7*3))
        # Prior estimates
        x_p = np.zeros((doy.max(), 7*3))
        C_p = np.zeros((doy.max(), 7*3, 7*3))
        # add initial condition
        x[0] = np.hstack(x0)
        C[0] = C0

        """
        And for the smoothed estimate
        """
        xs = np.zeros((doy.max(), 7*3))
        Cs = np.zeros((doy.max(), 7*3, 7*3))
        _x = np.zeros((doy.max(), 7*3))

        """
        Make the observation operator for each timestep
        """
        Ht = np.zeros((doy.max(), 7, 7*3))
        for t in xrange(doy.min()+1, doy.max()):
            idx = np.where(doy==t)[0]
            if idx.shape[0]>0:
                idx = np.take(idx, 0)
                ha = np.array([1, kerns.Ross[idx], kerns.Li[idx]])
                for b in xrange(7):
                    Ht[idx, b, 3*b:(3*b+3)]=ha



        """
        Specify process model
        """
        # phrase Q as forgetting filter..
        alphaIso = 0.94 # 0.94
        alphaGV =  0.97 # 0.94
        Q = (C0 !=0).astype(float)
        qDiag = np.array([ [1/alphaIso, 1/alphaGV, 1/alphaGV ] for i in xrange(7)]).flatten()
        np.fill_diagonal(Q, qDiag)
        iw = np.ones(21)
        I21 = np.eye(21)
        """
        *-- Run the iterative edge preserving filter --*

        This runs the kalman smoother several times and updates
        the edge process weighting
        """
        w = np.ones((doy.max(), 7))
        err_0 = 1e10
        err_1 = 1e10-1
        TOL = 1e-3
        MAX_ITER = 10
        iter =0
        while np.abs(err_1 - err_0) > TOL and iter < MAX_ITER:
            SSE = 0.0
            iter +=1
            #print iter
            """
            run the fwd kalman filter
            """
            for t in xrange(doy.min()+1, doy.max()):
                """
                predict state and covariance

                Get w value...

                """
                w_t = w[t]
                iw[::3] = 1.0/w_t
                np.fill_diagonal(Q, qDiag * iw)
                x_t = x[t-1]
                C_t = C[t-1] * Q
                # same prior estimates]
                x_p[t] = x_t
                C_p[t]=C_t
                """
                update if we have an observation
                """
                idx = np.where(doy==t)[0]
                if idx.shape[0]==0:
                    """
                    no observation
                    So propagate the prediction
                    """
                    x[t]=x_t
                    C[t]=C_t
                else:
                    """
                    Reflectance observation do kalman update
                    """
                    if qa[idx][0]:
                        """
                        make the observation operator
                        """
                        idx = np.take(idx, 0)
                        #ha = np.array([1, kerns.Ross[idx], kerns.Li[idx]])
                        #for b in xrange(7):
                        #    H[b, 3*b:(3*b+3)]=ha
                        H = Ht[idx]
                        #import pdb; pdb.set_trace()
                        # innovation
                        r = rho[idx] - H.dot(x_t)
                        # innovation covariance
                        S = (R + H.dot(C_t).dot(H.T))
                        # kalman gain
                        invS = np.diag(1.0/np.diag(S))
                        K = C_t.dot(H.T).dot(invS)
                        """
                        update
                        """
                        x[t] = x_t + K.dot(np.squeeze(r))
                        C_ = (I21 - K.dot(H)).dot(C_t)
                        # If its gone weird...
                        if det(C_[:3, :3]) <= 0:
                            C[t]=C[t-1]
                        else:
                            C[t]=C_
                        SSE+= (r**2).sum()
                    else:
                        x[t]=x_t
                        C[t]=C_t
            """
            Now do backward pass of the RTS smoother

            https://cse.sc.edu/~terejanu/files/tutorialKS.pdf
            """
            Cs[-1]=C[-1]
            xs[-1]=x[-1]

            for t in np.arange(doy.min(), doy.max()-1)[::-1]:
                C_p_t1 = C_p[t+1]
                C_tt = C[t]
                K_t = C_tt.dot(np.linalg.inv(C_p_t1))
                x_t = x[t] + K_t.dot( xs[t+1]-x_p[t+1]  )
                C_t = C_tt - K_t.dot( C_p_t1 - Cs[t+1]).dot(K_t.T)
                xs[t]=x_t
                Cs[t]=C_t
            """
            update w for the next iteration
            """
            w = 1 / (1 + 5000 * np.gradient(xs[:, ::3], axis=0)**2)
            # update error
            err_0 = err_1
            err_1 = SSE
        self.xs = xs
        self.Cs = Cs
        self.w = w

def loader2(f = "pix.A2004153.r43.c34.d165.txt"):
    """
    load a fire from
    /data/geospatial_19/ucfajlg/fire/Angola/time_series
    """
    a = np.genfromtxt(f, names=True)
    refl = np.array([a['b%02i' %bb ] for bb in xrange(1, 8)]).T
    doy = a['BRDF']
    sza = a['SZA']
    vza = a['VZA']
    raa = a['RAA']
    qa = a['QA_PASS'].astype(bool)
    """
    make unique -- one a day
    """
    _, idx = np.unique(doy, return_index=True)

    qa = qa[idx]
    refl = refl[idx]
    vza = vza[idx]
    sza = sza[idx]
    raa = raa[idx]
    doy = (doy[idx]).astype(np.int)
    kerns = Kernels(vza, sza, raa,
        LiType='Sparse', doIntegrals=False,
        normalise=True, RecipFlag=True, RossHS=False, MODISSPARSE=True,
        RossType='Thick',nbar=0.0)
    return doy, qa, refl, kerns

"""
good
221 345 77 101 4444 448 485
540 is a big fire

4445 experiment
qa[30:50]=False

79 is an interesting one... signal is lost
"""
ff = glob.glob("/data/geospatial_19/ucfajlg/fire/Angola/time_series/*txt")

doy, qa, rho, kerns  = loader2(ff[111])
doy -= doy.min()
#qa[27:50]=False

k = KSw(doy, qa, rho, kerns)
k.solve()
xs = k.xs
Cs = k.Cs
w = k.w 


#plt.plot(Cs[:, 3*band,3*band])

#%lprun -f k.solve k.solve()




band = 3

plt.plot(xs[:, 3*band], 'r.-')
plt.plot(xs[:, 3*band] + 2*Cs[:, 3*band,3*band], 'gray', lw=1)
plt.plot(xs[:, 3*band] - 2*Cs[:, 3*band,3*band], 'gray', lw=1)
#plt.plot(xs[:, 4])
#plt.plot(xs[:, 5])
plt.plot(doy[qa], rho[qa,band], '+-', lw=0.5)
plt.plot(w[:, band], '.k')
plt.ylim(0,1)






