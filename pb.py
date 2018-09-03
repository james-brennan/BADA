"""
This module performs the pb calculation section...
making it the main part of the classfication algorithm
"""
import numpy as np
import scipy.stats


"""
An initial idea for calculating Pb

The parameeters could be learnt from
the active fires plus a global prior
on these...

maybe develop in a notebook first?
"""


# this will do for now...


def PB4(pre, post, pre_unc, post_unc):
    """

    Back to the original idea
    with a one class thing

    Gaussian for now

    marginalise over the uncertainty...

    include fcc bit for now
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
    y = np.array(post - pre)#/np.sqrt(chngUnc)
                                        #Difference Post
                                        # and pre fire rhos
    """
    also need to treat change signal by including
    date uncertainty
    """
    # add third term
    K[:, 2] = pre.squeeze()#/np.sqrt (chngUnc ) # K is the matrix with our linear
                                # system of equations (K*x = y)
    """
    Make covariance matrix
    """
    CInv = np.diag(1/chngUnc**2)
    C = np.diag(chngUnc**2)
    #import pdb; pdb.set_trace()
    KTK = K.T.dot(CInv).dot(K)
    KTy = K.T.dot(CInv).dot(y)
    sP = np.linalg.solve(KTK, KTy)
    #Uncertainty
    inv = np.linalg.inv ( KTK )
    fcc = -sP[2]
    a0 = sP[0]/fcc
    a1 = sP[1]/fcc

    theta_pars = np.array([a0,a1, fcc ])
    C_theta = inv
    model_err = np.exp(-0.5 * (K.dot(sP) - y).T.dot(CInv).dot(K.dot(sP) - y))

    """
    Move into the classifier model...
    """

    """
    Define constants
    """
    # fcc slope prior
    alpha = 10.0
    # global model for a0 and a1 for now
    a_mu = np.array([0.02, 0.17])
    #C_a = np.array([[0.05, 0.0], [0.0, 0.1]])
    C_a = np.array([[ 0.008,  0.0  ],
                 [ 0.0  ,  0.02 ]])
    a_max = scipy.stats.multivariate_normal(a_mu, C_a).pdf(a_mu)
    """
    *-- potential speed-up for now --*
    if the MAP of the classifier is very low
    eg no chance of a fire we just input this value...
    """
    # get map
    fcc_map = 1 - np.exp(-alpha * theta_pars[2])
    a_map = scipy.stats.multivariate_normal(a_mu, C_a).pdf(theta_pars[:2])
    a_map /= a_max
    if fcc_map * a_map < 0.05:
        # MAP estimates less than 5% chance of fire...
        # plug in this
        return a_map * fcc_map, model_err, fcc, a0, a1
    else:
        """
        Now want to marginalise over the
        the uncertainty in the parameters fcc, a0, a1

        + the quality of the model...?
        """
        theta_samps = scipy.stats.multivariate_normal(theta_pars, C_theta).rvs(5000)
        fcc_samps = theta_samps[:, 2]
        a_samps = theta_samps[:, :2]
        # get fcc likelihood
        fcc_eq = 1 - np.exp(-alpha * fcc_samps)
        fcc_eq[fcc_eq<0]=0
        # use scipy stats for now
        # a0 and a1 likelihood
        a_like = scipy.stats.multivariate_normal(a_mu, C_a).pdf(a_samps)
        # normalise to 0 and 1
        a_like /= a_max
        ppb = fcc_eq * a_like
        """
        And model 'fidelty' term

        eg y

        [description]
        """
        #import pdb; pdb.set_trace()
        return model_err * ppb.mean(), model_err, fcc, a0, a1





def PB3(pre, post, pre_unc, post_unc):
    """
    This follows a slightly different concept to before...

    now a two class modelling problem...
    """
    # do fcc calculation here...

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
    y = np.array(post - pre)#/np.sqrt(chngUnc)
                                        #Difference Post
                                        # and pre fire rhos
    """
    also need to treat change signal by including
    date uncertainty
    """
    # add third term
    K[:, 2] = pre.squeeze()#/np.sqrt (chngUnc ) # K is the matrix with our linear
                                # system of equations (K*x = y)
    """
    Make covariance matrix
    """
    CInv = np.diag(1/chngUnc**2)
    C = np.diag(chngUnc**2)
    #import pdb; pdb.set_trace()
    KTK = K.T.dot(CInv).dot(K)
    KTy = K.T.dot(CInv).dot(y)
    sP = np.linalg.solve(KTK, KTy)
    #Uncertainty
    inv = np.linalg.inv ( KTK )
    fcc = -sP[2]
    a0 = sP[0]/fcc
    a1 = sP[1]/fcc
    # get RMSE?

    """
    Now put in the two model part
    using two gaussians for the
    classification
    """
    B_mu =  np.array([ 1.1  ,  0.05,  0.15])
    B_prior = np.array([[ 0.3,  0.    ,  0.    ],
                         [ 0.    ,  0.008 ,  0.    ],
                         [ 0.    ,  0.    ,  0.02  ]])
    U_mu = np.array([ 0.  ,  0.00,  0.0])
    U_prior = np.array([[ 0.0625,  0.    ,  0.    ],
                        [ 0.    ,  1e1 ,  0.    ],
                        [ 0.    ,  0.    ,  1e1  ]])


    a_obs = np.array([fcc, a0, a1])
    _C_a = inv

    # first burnt class
    # make joint inverse
    CppInv = np.linalg.inv(B_prior +_C_a ) #+ np.linalg.inv(  _C_a )
    # assumed cost function for a
    const = np.sqrt( (2*np.pi)**3 * np.linalg.det(np.linalg.inv(CppInv)) )
    burnt_cost =  np.exp(-0.5 * (a_obs - B_mu).T.dot(CppInv).dot(a_obs-B_mu))

    # make joint inverse
    CppInv = np.linalg.inv(U_prior +_C_a  ) #+ np.linalg.inv(  _C_a )
    # assumed cost function for a
    const = np.sqrt( (2*np.pi)**3 * np.linalg.det(np.linalg.inv(CppInv)) )
    unburnt_cost =  np.exp(-0.5 * (a_obs - U_mu).T.dot(CppInv).dot(a_obs-U_mu))


    pb = burnt_cost/(burnt_cost + unburnt_cost)


    return pb




def Pb_MC(theta_mu, C_theta):
    """
    Define constants
    """
    alpha = 10.0
    # global model for a0 and a1 for now
    a_mu = np.array([0.08, 0.15])
    #C_a = np.array([[0.05, 0.0], [0.0, 0.1]])
    C_a = np.array([[ 0.008,  0.0  ],
                 [ 0.0  ,  0.02 ]])
    a_max = scipy.stats.multivariate_normal(a_mu, C_a).pdf(a_mu)
    """
    *-- potential speed-up for now --*
    if the MAP of the classifier is very low
    eg no chance of a fire we just input this value...
    """
    # get map
    fcc_map = 1 - np.exp(-alpha * theta_mu[0])
    a_map = scipy.stats.multivariate_normal(a_mu, C_a).pdf(theta_mu[1:])
    a_map /= a_max
    if fcc_map * a_map < 0.05:
        # MAP estimates less than 5% chance of fire...
        # plug in this
        return a_map * fcc_map
    else:
        # considerable uncertainty...???????
        # get samples from the distribution
        theta_samps = scipy.stats.multivariate_normal(theta_mu, C_theta).rvs(5000)
        fcc = theta_samps[:, 0]
        a = theta_samps[:, 1:]
        # get fcc likelihood
        fcc_eq = 1 - np.exp(-alpha * fcc)
        fcc_eq[fcc_eq<0]=0
        # use scipy stats for now
        # a0 and a1 likelihood
        a_samps = scipy.stats.multivariate_normal(a_mu, C_a).pdf(a)
        # normalise to 0 and 1
        a_samps /= a_max
        ppb = fcc_eq * a_samps
        return np.mean(ppb)





def PB2(pre, post, pre_unc, post_unc):
    """
    This follows a slightly different concept to before...

    now a two class modelling problem...
    """
    # do fcc calculation here...

    wavelengths = np.array([645., 858.5, 469., 555., \
                                1240., 1640., 2130.])
    """
    sort out uncertainty
    """
    # turn into standard devs
    #pre_unc = np.sqrt(pre_unc)
    #post_unc = np.sqrt(post_unc)
    #chngUnc = 2*np.sqrt(pre_unc**2 + post_unc**2) ## ((bu*pre_unc)**2 + (bu*post_unc)**2)
    chngUnc = pre_unc  + post_unc
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
    y = np.array(post - pre)#/np.sqrt(chngUnc)
                                        #Difference Post
                                        # and pre fire rhos
    """
    also need to treat change signal by including
    date uncertainty
    """
    # add third term
    K[:, 2] = pre.squeeze()#/np.sqrt (chngUnc ) # K is the matrix with our linear
                                # system of equations (K*x = y)
    """
    Make covariance matrix
    """
    CInv = np.diag(1/chngUnc)
    C = np.diag(chngUnc)
    #import pdb; pdb.set_trace()
    KTK = K.T.dot(CInv).dot(K)
    KTy = K.T.dot(CInv).dot(y)
    sP = np.linalg.solve(KTK, KTy)
    #Uncertainty
    inv = np.linalg.inv ( KTK )
    fcc = -sP[2]
    a0 = sP[0]/fcc
    a1 = sP[1]/fcc
    # get RMSE?


    """
    First compute the model fidelty

    This decides whether the fcc model provides
    a better fit to y than a model with no change...
    eg a0=0, fcc=0, a1=0
    """
    const = 1/np.sqrt((2*np.pi)**7 *np.linalg.det(C))
    noChangeModel = const * np.exp(-0.5 *  y.dot(CInv).dot(y))
    """
    Now the fcc model
    """
    const = 1/np.sqrt( (2*np.pi)**7 * np.linalg.det(C) )
    fcc_Model =const *  np.exp(-0.5 *(K.dot(sP) - y).dot(CInv).dot(K.dot(sP) - y))


    """
    given the assumption that these two models
    cover every possible condition...
    eg fcc works for non-fire related changes too
    the posterior probability for the hypothesis

    that the fcc model is the correct one for the data is

    """
    pModel = fcc_Model / (fcc_Model + noChangeModel)


    """
    Now consider the burn signal specified by the
    fcc model directly to see how good it is...

    use the one from before for now (eg PB4)
    """


    B_mu =  np.array([ 1.1  ,  0.05,  0.15])
    B_prior = np.array([[ 0.3,  0.    ,  0.    ],
                         [ 0.    ,  0.008 ,  0.    ],
                         [ 0.    ,  0.    ,  0.02  ]])
    U_mu = np.array([ 0.  ,  0.00,  0.0])
    U_prior = np.array([[ 0.0625,  0.    ,  0.    ],
                        [ 0.    ,  1e1 ,  0.    ],
                        [ 0.    ,  0.    ,  1e1  ]])


    a_obs = np.array([fcc, a0, a1])
    _C_a = inv

    # first burnt class
    # make joint inverse
    CppInv = np.linalg.inv(B_prior +_C_a ) #+ np.linalg.inv(  _C_a )
    # assumed cost function for a
    const = np.sqrt( (2*np.pi)**3 * np.linalg.det(np.linalg.inv(CppInv)) )
    burnt_cost =  np.exp(-0.5 * (a_obs - B_mu).T.dot(CppInv).dot(a_obs-B_mu))

    # make joint inverse
    CppInv = np.linalg.inv(U_prior +_C_a  ) #+ np.linalg.inv(  _C_a )
    # assumed cost function for a
    const = np.sqrt( (2*np.pi)**3 * np.linalg.det(np.linalg.inv(CppInv)) )
    unburnt_cost =  np.exp(-0.5 * (a_obs - U_mu).T.dot(CppInv).dot(a_obs-U_mu))
    #return noChangeModel, fcc_Model, fcc, a0, a1, burnt_cost, unburnt_cost
    # only return what we need
    pModel = fcc_Model / (fcc_Model + noChangeModel)
    pB = burnt_cost / (burnt_cost + unburnt_cost)
    return pB * pModel, fcc, a0, a1



class classifier(object):
    """
    A first generic implementation of the classifier
    algorithm

    Includes the computation of pb via integration...

    """
    def __init__(self):
        """
        Define global values for a0 and a1
        These can be considered an appropriate global
        prior for these values but they should be found probably..
        """
        self.a_mu = np.array([0.08, 0.15])
        self.C_a = np.array([[0.05, 0.01], [0.01, 0.1]])

        """
        Re-call that the maximum of this likelihood
        should be 1
        so find the max point from which to provide
        the normalisation constant
        """
        self.a_const = scipy.stats.multivariate_normal(self.a_mu, self.C_a).pdf(self.a_mu)


    def computation(self):
        pass

    def Pb(self, fcc, a0, a1, cov):
        """
        add docs
        """
        pass
