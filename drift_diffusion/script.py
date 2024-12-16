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
from funs import *
now = datetime.now()






ns = [100, 200]
rs = [2,3]

iters = np.arange(2)
sigmas = [0,.05]



nr = []
for i,n in enumerate(ns):
    for j,r in enumerate(rs):
            nr.append((n,r))



mu = lambda t :  -2*(0.2+0.8*np.sin(2*np.pi*t))
sigma=lambda t: np.exp(1)**t

# x0 
x0 = 2


# discretization of unit interval for estimation evaluation 
N = 25;   


# step_size of Eurler-Maryuama approximation method
dt = 1e-3


h_G = 1.5*(n*r)**(-1/5)
h_m = 1.5*(n*r)**(-1/5)




np.seterr(divide='ignore')




def runs(it):
    
    drift = dict()

    diffusion_diag_T = dict()
    diffusion_diag_D = dict()
    diffusion_tria = dict()

    int_diffusion_diag_T = dict()
    int_diffusion_diag_D = dict()
    int_diffusion_tria = dict()


    est_drift = dict()

    est_diffusion_diag_T = dict()
    est_diffusion_diag_D = dict()
    est_diffusion_tria = dict()

    est_int_diffusion_diag_T = dict()
    est_int_diffusion_diag_D = dict()
    est_int_diffusion_tria = dict()

    for n, r in nr:
        for sigma_noise in sigmas:
            (rmse_drift, rmse_tria, rmse_diag_D, rmse_diag_T, 
                    rmse_int_sigma2_tria, rmse_int_sigma2_diag_T, rmse_int_sigma2_diag_D,
                    MUHAT, SIGMASQUAREDHAT_tria,SIGMASQUAREDHAT_diag_D, SIGMASQUAREDHAT_diag_T, 
                    int_sigma2_tria, int_sigma2_diag_T, int_sigma2_diag_D
                )      =  run(n, r, dt, mu, sigma, x0, N, sigma_noise,h_m,h_G, it)


            drift.update({('mu',n,r,sigma_noise,it) : rmse_drift})
            diffusion_tria.update({('sigma, tria',n,r,sigma_noise,it) : rmse_tria})
            diffusion_diag_D.update({('sigma, diag D',n,r,sigma_noise,it) : rmse_diag_D})
            diffusion_diag_T.update({('sigma, diag T',n,r,sigma_noise,it) : rmse_diag_T})

            int_diffusion_tria.update({('int_sigma, tria',n,r,sigma_noise,it) : rmse_int_sigma2_tria})
            int_diffusion_diag_D.update({('int_sigma, diag D',n,r,sigma_noise,it) : rmse_int_sigma2_diag_D})
            int_diffusion_diag_T.update({('int_sigma, diag T',n,r,sigma_noise,it) : rmse_int_sigma2_diag_T})



            est_drift.update({('mu',n,r,sigma_noise,it) : MUHAT})
            est_diffusion_diag_T.update({('sigma, tria',n,r,sigma_noise,it) : SIGMASQUAREDHAT_tria})
            est_diffusion_diag_D.update({('sigma, diag D',n,r,sigma_noise,it) : SIGMASQUAREDHAT_diag_D})
            est_diffusion_tria.update({('sigma, diag T',n,r,sigma_noise,it) : SIGMASQUAREDHAT_diag_T})

            est_int_diffusion_tria.update({('int_sigma, tria',n,r,sigma_noise,it) : int_sigma2_tria})
            est_int_diffusion_diag_D.update({('int_sigma, diag D',n,r,sigma_noise,it) : int_sigma2_diag_D})
            est_int_diffusion_diag_T.update({('int_sigma, diag T',n,r,sigma_noise,it) : int_sigma2_diag_T})

    res = drift.copy(); res.update(diffusion_diag_T); res.update(diffusion_diag_D);res.update(diffusion_tria);
    res.update(int_diffusion_tria); res.update(int_diffusion_diag_D);res.update(int_diffusion_diag_T);

    df = pd.DataFrame(data = np.array(list(res)), columns = ['type', 'n', 'r','nu' ,'iter'])
    df['RMSE'] = np.array(list(res.values())).reshape(-1,1)
    df.reset_index(); df.n = df.n.astype(int); df.r = df.r.astype(int)
    df_rmse = df.copy()

    res = est_drift.copy(); res.update(est_diffusion_diag_T); res.update(est_diffusion_diag_D);  res.update(est_diffusion_tria)
    res.update(est_int_diffusion_diag_T);res.update(est_int_diffusion_tria);res.update(est_int_diffusion_diag_D);

     
    df = pd.DataFrame(data = np.array(list(res)), columns = ['type', 'n', 'r','nu' ,'iter'])
    df['estimated_value'] = list(res.values())
    df.reset_index(); df.n = df.n.astype(int); df.r = df.r.astype(int)
    df_est = df.copy()

    df = pd.merge(left = df_rmse, right = df_est, on = ['type','n','r','nu','iter'])

    return df




ts = datetime.timestamp(now)


from multiprocessing import get_context
from multiprocessing import set_start_method


if __name__ == '__main__':
    set_start_method("spawn")
    with get_context("spawn").Pool(25) as p:
        data = p.map(runs, iters)
        df = pd.concat(data)
        df.to_csv('_cx_{}_____nr={}__sigmas={}__N={}__iters={}.csv'.format(ts,     nr,	 sigmas,	    N,	 len(iters)))
        p.close()










