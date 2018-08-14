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

def Pb_MC(theta_mu, C_theta):
    """
    Define constants
    """
    alpha = 10.0
    # global model for a0 and a1 for now
    a_mu = np.array([0.08, 0.15])
    C_a = np.array([[0.05, 0.01], [0.01, 0.1]])
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
    a_max = scipy.stats.multivariate_normal(a_mu, C_a).pdf(a_mu)
    # normalise to 0 and 1
    a_samps /= a_max
    ppb = fcc_eq * a_samps
    return np.mean(ppb)





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
