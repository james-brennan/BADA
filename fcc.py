"""
fcc.py
""" 
import numpy as np
import scipy.stats 


def fccModel(pre_fire, post_fire, pre_unc, post_unc):
    """
    Solve fcc model on the day to day difference
    to solve for solution. fcc model is:
        rho_post - rho_pre = fcc(a_0 + a_1 f(ll) - rho_post)
    where f(ll) is a function of wavelength ll to represent
    the soil line:
        f(ll) =  (2.0 / llmax) * (ll - ll*ll/(2.0*llmax))
    to solve fcc as a linear system we can write the system as
        rho_post - rho_pre  = fcc*a_0 + fcc*a_1*f(ll) - fcc*rho_pre
    """
    wavelengths = np.array([645., 858.5, 469., 555., \
                                1240., 1640., 2130.])
    """
    sort out uncertainty
    """
    # turn into standard devs
    #pre_unc = np.sqrt(pre_unc)
    #post_unc = np.sqrt(post_unc)
    #chngUnc = 2*np.sqrt(pre_unc**2 + post_unc**2) ## ((bu*pre_unc)**2 + (bu*post_unc)**2)
    chngUnc = np.sqrt(pre_unc**2 + post_unc**2)
    """
    calculate the values of f(ll)
    """
    loff = 400.
    lmax = 2000.
    ll =  wavelengths - loff
    llmax = lmax-loff
    lk = (2.0 / llmax) * (ll - ll*ll/(2.0*llmax))
    K = np.array(np.ones([7,3]))
    K[:, 1] = lk.transpose()#/np.sqrt(chngUnc)
    K[:, 0] = K[:, 0]#/np.sqrt(chngUnc)
    # change signal
    y = np.array(post_fire - pre_fire)#/np.sqrt(chngUnc)
                                        #Difference Post
                                        # and pre fire rhos
    """
    also need to treat change signal by including
    date uncertainty
    """
    # add third term
    K[:, 2] = pre_fire.squeeze()#/np.sqrt (chngUnc ) # K is the matrix with our linear
                                # system of equations (K*x = y)

    """
    Make covariance matrix
    """
    CInv = np.diag(1/chngUnc**2)

    #import pdb; pdb.set_trace()
    KTK = K.T.dot(CInv).dot(K)
    KTy = K.T.dot(CInv).dot(y)
    sP = np.linalg.solve(KTK, KTy)
    #sP,residual, rank,singular_vals = np.linalg.lstsq ( K, y )
    #Uncertainty
    inv = np.linalg.inv ( KTK )
    #    (fccUnc, a0Unc, a1Unc ) = \
    #        inv.diagonal().squeeze()
    #else:
    #    (fccUnc, a0Unc, a1Unc ) = -998, -998,-998
    # get indiv. elements
    fcc = -sP[2]
    a0 = sP[0]/fcc
    a1 = sP[1]/fcc
    #sBurn = a0 + lk*a1
    #sFWD = pre_fire*(1-fcc) + fcc*sBurn
    #import pdb; pdb.set_trace()
    return fcc, a0, a1, inv


