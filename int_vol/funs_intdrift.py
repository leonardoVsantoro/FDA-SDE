# libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
from scipy.integrate import simps as simps
from random import randrange
from datetime import date
from numpy.random import normal
from datetime import datetime
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
import itertools
from scipy.integrate import quad as integrate
from scipy.integrate import simps as simps
from scipy import interpolate as interp
from scipy.stats import norm

def EulerMaruyama_approx_sol(N, mu, sigma, x0,ax):
    """
    Returns discrete numerical approximation of stochastic process solving an SDE of the form
            dX(t) =  b(t)X(t) dt + c(t)dW(t) 
    by approximating the continous solution X(t) by random variables 
    X_n = X(t_n) where t_n = n∆t, n = 0,1,2, and ∆t = 1/N

    Parameters
    ----------
    N : int
        Discretisation of [0,1] interval. Determines step size ∆t = 1/N
    mu : function
        Drift of SDE, takes (x, t) as input
    sigma : function
        Diffusion of SDE, takes (x, t) as input 
    x0 : float
        Starting point of solution

    Returns
    ----------
    X : array(float) np.array of size N
    """ 
    
    t = np.linspace(0,1,N)
    X = np.zeros(N)
    
    X[0] = x0
    dt = 1/N
    
    for i in range(N-1):
        dX = X[i]*mu(t[i])*dt + sigma(t[i])*normal()*np.sqrt(dt)
        X[i+1] = X[i] + dX
    ax.plot(t,X,lw=1,alpha=.3)
    return X

def generate_sparse_obs__forScores(n, r, dt, mu, sigma, x0, N, sigma_noise ):
    N = int(dt**(-1)) 
    X = []; T = []; Y = []; C = []
    fig,ax = plt.subplots(figsize =(20,3))
    for i in (range(n)):
        X_i = EulerMaruyama_approx_sol(N = N, mu= mu , sigma=sigma, x0 = x0, ax=ax); X.append(X_i)
        T_i = []; Y_i = []
        for j in range(r):
            random_input = randrange(N)
            T_ij = random_input/N
            Y_ij = X_i[random_input] + np.random.normal(scale = sigma_noise)
            Y_i.append(Y_ij); T_i.append(T_ij)
        Y.append(np.array([Y_i[j] for j in np.argsort(T_i)]))  
        T.append(np.array([T_i[j] for j in np.argsort(T_i)]))
    X = np.array(X); T = np.array(T); Y = np.array(Y)
    return T,Y       

def univariate_local_linear_smoothing(T,Y,K,h):
    r = Y.shape[1]; n  = Y.shape[0]; _nr = n*r
    T = T.reshape(-1); Y = Y.reshape(-1)
    K_h = lambda t: K(t/h)/h   
    cali_T = lambda t0 : np.array( [ ( (T - t0) )**p for p in [0,1] ] ).T
    bold_W = lambda t0 : np.diag(np.array([K_h(_t - t0) for _t in T]))
    def minimizers(t0):
        try:
            res =  np.linalg.inv(  cali_T(t0).T @ bold_W(t0) @ cali_T(t0) )  @ cali_T(t0).T @  bold_W(t0) @  Y 
        except:
            res =   np.linalg.pinv(  cali_T(t0).T @ bold_W(t0) @ cali_T(t0) )@ cali_T(t0).T @  bold_W(t0) @  Y 
        return  res
    return lambda t0: minimizers(t0)
          
def bivariate_local_linear_smoothing_triangular(T,Y,K,h, bias = True):
    r = Y.shape[1]; n  = Y.shape[0]
    if bias is True:
        upper_triangular_pairs = set(
            pair for pair in zip(*np.triu_indices(r))).difference(set(zip(np.arange(r),np.arange(r))))
    else:
        upper_triangular_pairs = set(pair for pair in zip(*np.triu_indices(r)))

    utp_T = [(T_i[j], T_i[k]) for (j,k) in upper_triangular_pairs for T_i in T ]
    utp_Y = [(Y_i[j], Y_i[k]) for (j,k) in upper_triangular_pairs for Y_i in Y ]

    Kh = lambda t: K(t/h)/h

    caliK = lambda s,t : np.diag( np.array([ Kh(T_ij-s)*Kh(T_ik-t) for (T_ij, T_ik) in utp_T ]))

    caliY = np.array([ Y_ij*Y_ik for (Y_ij, Y_ik) in utp_Y ]) 

    caliX = lambda s,t : np.array([ np.array([ 1, T_ij - s, T_ik - t ]) for (T_ij, T_ik) in utp_T ]) 
    
    def minimizers(s,t):
        _caliX = caliX(s,t); _caliK = caliK(s,t); _caliY = caliY; XtransposeK =  _caliX.T @ _caliK

        try:
            res = np.linalg.inv(  XtransposeK @ _caliX) @ XtransposeK @ _caliY
        except:
            res = np.linalg.pinv(  XtransposeK @ _caliX) @ XtransposeK @ _caliY
        return  res

    return lambda s,t : minimizers(min(s,t), max(s,t))


def get_mu(m,dm):
    return lambda t : dm(t)*(m(t))**(-1)

def get_sigma2_DIAGONAL_fromFull(_m, _dm, _G, _D1_G, _D2_G, _mu):
    N = _m.size
    time_grid = np.linspace(0,1,N)
    def res(t):
        t_ix = np.argmin((t-time_grid)**2)
        return (np.diagonal(_D1_G + _D2_G) - 2*_m*_dm  - 2*_mu*(np.diagonal(_G) - _m**2))[t_ix]   
    
    return lambda t : res(t)

def get_sigma2_DIAGONAL_fromDiag(_m, _dm, _diag_G, _diag_dG, _mu):
    N = _m.size
    time_grid = np.linspace(0,1,N)
    def res(t):
        t_ix = np.argmin((t-time_grid)**2)
        return (_diag_dG - 2*_m*_dm  - 2*_mu*(_diag_G - _m**2))[t_ix]   

    return lambda t : res(t)

def get_sigma2_TRIANGULAR(_m, _dm, _G, _D1_G, _b):

    N = _m.size
    time_grid = np.linspace(0,1,N)
    def res(s):
#         s_ix = max( 0, np.argmin((s-time_grid)**2)-1)
        s_ix = np.argmin((s- 0.021 -time_grid)**2)
            
        def second(s_ix,t_ix):
            return simps(_b[s_ix : t_ix+1]*_D1_G[s_ix, s_ix : t_ix+1], time_grid[s_ix: t_ix+1]  )
        
        return np.array(
                [ _D1_G[s_ix, t_ix ] - _b[s_ix]*_G[s_ix, s_ix ] - second(s_ix,t_ix)
                  for t_ix in np.arange(s_ix, time_grid.size-1)     ]).mean()

    return lambda s : res(s)




def get_int_sigma2_TRIANGULAR(_m, _dm, _G, _MU):

    N = _m.size
    time_grid = np.linspace(0,1,N)
    
    delta = (1/N)/2 +1e-2

    
    def res(s):
        _diag_G = np.array([_G[i,i] for i in np.arange(N)])
        
        s_ix = np.argmin((s- delta -time_grid)**2)
        
        first  = np.array([ _G[s_ix,t_ix] for t_ix in np.arange(s_ix, N-1)]).mean()
        
        
    
        mid = _G[0,0] + 2* simps(_diag_G[0:s_ix+1]*_MU[:s_ix+1], time_grid[:s_ix+1])
        
    
        
        last = np.array([  simps(_G[s_ix,s_ix:t_ix+1]*_MU[s_ix:t_ix+1], time_grid[s_ix: t_ix+1])
                        for t_ix in np.arange(s_ix, N-1)]).mean()
        

        
        return  first - mid - last
    

    return lambda s : res(s) 
def get_int_sigma2_DIAGONAL_tria(_m, _dm, _G, _MU):
    
    N = _m.size
    time_grid = np.linspace(0,1,N)
    
    delta = (1/N)/2 
    
    _diag_G = np.array([_G[i,i] for i in np.arange(N)])
    
    def res(s):
        s_ix = np.argmin((s- delta -time_grid)**2)
        
        first  = _diag_G[s_ix]- _diag_G[0]
        
        second = simps(_MU[:s_ix+1]*_diag_G[:s_ix+1], time_grid[:s_ix+1], even = 'avg')
        
        

        
        return  first - 2*second
    

    return lambda s : res(s)
def get_int_sigma2_DIAGONAL(_m, _dm, _diag_G, _MU):
    
   

    N = _m.size
    time_grid = np.linspace(0,1,N)
    
    delta = (1/N)/2 
    
    
    def res(s):
        s_ix = np.argmin((s- delta -time_grid)**2)
        
        first  = _diag_G[s_ix]- _diag_G[0]
        
        second = simps(_MU[:s_ix+1]*_diag_G[:s_ix+1], time_grid[:s_ix+1], even = 'avg')
        
        

        
        return  first - 2*second
    

    return lambda s : res(s)















def run_int(n, r, dt, mu, sigma, x0, N, sigma_noise, h_m, h_G, it=0):

    time_grid = np.linspace(0,1,N)
    
    T,Y = generate_sparse_obs__forScores(n, r, dt, mu, sigma, x0, N, sigma_noise )
    
    if sigma_noise == 0:
        bias = False
    else:
        bias = True 

    
    # smoothing kernel
    K = lambda x : 3/4*(1-x**2) if  3/4*(1-x**2) > 0 else 0;      

    # -------------------------------------------- compute -------------------------------------------- #
    ##### m
    _M_ = np.array([univariate_local_linear_smoothing(T, Y, K, h_m)(t) for t in (time_grid)])
    MHAT = _M_[:,0]
    dMHAT = _M_[:,1]
    



    # -------------------------------------------- evaluate -------------------------------------------- #

    MU = np.array([mu(t) for t in time_grid])
    int_MU = np.array([simps((MU)[:t_ix+1], time_grid[:t_ix+1]) for t_ix in np.arange(N)])                
    
    # mu -- drift     
    MUHAT = dMHAT/MHAT
    
    int_MUHAT = np.array([ np.log(np.abs(MHAT[i]/MHAT[0])) for i in np.arange(N) ])
    
    rmse_int_drift = ((((int_MUHAT-int_MU)**2)[2:-2]).mean())**.5


    
    
    return rmse_int_drift, int_MUHAT
                        



