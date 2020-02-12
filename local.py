from scipy.linalg import expm
import msmtools
import numpy as np


class Channel:
    """
    Implementation of the above 5 state channel.
    """
    
    statemap = {
    0 : 'C1',
    1 : 'C2',
    2 : 'C3',
    3 : 'C4',
    4 : 'O'    
    }
    
    def __init__(self, alpha, beta):
        self.cs = np.zeros(5)
        self.a = alpha
        self.b = beta    
    
    def rate_matrix(self):
        rmat = np.zeros((5, 5))
        rmat[0] = [-4*self.a, 4*self.a,           0,                  0,                   0]
        rmat[1] = [self.b,    -self.b - 3*self.a, 3*self.a,           0,                   0]
        rmat[2] = [0,         2*self.b,           -2*(self.a +self.b),2*self.a,            0]
        rmat[3] = [0,         0,                   3*self.b,          -3*self.b - self.a,  self.a]
        rmat[4] = [0,         0,                   0,                  4*self.b,           -4 * self.b]
        
        assert msmtools.analysis.is_rate_matrix(rmat)
        
        return rmat
    
    def transition_matrix(self, lag=1):
        tmat = expm(lag * self.rate_matrix())
        assert msmtools.analysis.is_transition_matrix(expm(lag * self.rate_matrix()))
        return tmat
    
    def rate_matrix_singlet(self):
        rmat = np.zeros((2, 2))
        rmat[0] = [-self.a, self.a]
        rmat[1] = [self.b, -self.b]
        
        assert msmtools.analysis.is_rate_matrix(rmat)
        
        return rmat
    
    def transition_matrix_singlet(self, lag=1):
        tmat = expm(lag * self.rate_matrix_singlet())
        assert msmtools.analysis.is_transition_matrix(expm(lag * self.rate_matrix()))
        return tmat
    
    def index2state(self, idx: int):
        return self.statemap[idx]

def kchannel_params(Vm):
    alpha = (0.01*(10.-Vm))/(np.exp((10.-Vm)/10.)-1)
    beta = 0.125*np.exp(-Vm/80.)                     
    steadystate = alpha/(alpha + beta)
    
    tau = 1/(alpha + beta)
    
    return alpha, beta, steadystate, tau


def dndt(n, t, Vm):
    alpha, beta, _, _ = kchannel_params(Vm)
    
    return alpha*(1-n) - beta*n  


def cg_transition_matrix(T, chi):
    """
    Map a transition matrix T to coarse states via crisp membership
    matrix chi. Implements Eq. 14 of
    Roeblitz & Weber, Adv Data Anal Classif (2013) 7:147–179
    DOI 10.1007/s11634-013-0134-6
    
    :params:
    T: np.ndarray; transition matrix in microstate space
    chi: np.ndarray membership matrix
    """
    pi = msmtools.analysis.stationary_distribution(T)
    D2 = np.diag(pi)
    D_c2_inv = np.diag(1/np.dot(chi.T, pi))

    return D_c2_inv @ chi.T @ D2 @ T @ chi
