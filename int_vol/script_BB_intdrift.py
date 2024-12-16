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
from funs_intdrift import *
now = datetime.now()





ns = [100,200,500,1000]
rs = [2,3,5,10]

iters = np.arange(100)
sigmas = [0.1]


# discretization of unit interval for estimation evaluation 
N = 25;   



nr = []
for i,n in enumerate(ns):
    for j,r in enumerate(rs):
            nr.append((n,r))



mu = lambda t : - (1)/(1-t)
sigma = lambda t : 1


# x0 
x0 = 2




# step_size of Eurler-Maryuama approximation method
dt = 1e-3



h_m = 5*(n*r)**(-1/5);
h_G = 1.5*(n*r)**(-1/5)


np.seterr(divide='ignore')


# ----------------------------------------------------------


def runs(it):
    
    int_drift = dict()
    est_int_drift = dict()


    for n, r in nr:
        for sigma_noise in sigmas:
            (rmse_int_drift, int_mu)   =  run_int(n, r, dt, mu, sigma, x0, N, sigma_noise,h_m,h_G, it)


            int_drift.update({('int_mu',n,r,sigma_noise,it) : rmse_int_drift})
          
            est_int_drift.update({('int_mu',n,r,sigma_noise,it) : int_mu})
            

    res = int_drift.copy(); 
    df = pd.DataFrame(data = np.array(list(res)), columns = ['type', 'n', 'r','nu' ,'iter'])
    df['RMSE'] = np.array(list(res.values())).reshape(-1,1)
    df.reset_index(); df.n = df.n.astype(int); df.r = df.r.astype(int)
    df_rmse = df.copy()

    res = est_int_drift.copy(); 
     
    df = pd.DataFrame(data = np.array(list(res)), columns = ['type', 'n', 'r','nu' ,'iter'])
    df['estimated_value'] = list(res.values())
    df.reset_index(); df.n = df.n.astype(int); df.r = df.r.astype(int)
    df_est = df.copy()

    df = pd.merge(left = df_rmse, right = df_est, on = ['type','n','r','nu','iter'])

    return df

# ----------------------------------------------------------



ts = datetime.timestamp(now)


from multiprocessing import get_context
from multiprocessing import set_start_method


if __name__ == '__main__':
    set_start_method("spawn")
    with get_context("spawn").Pool(25) as p:
        data = p.map(runs, iters)
        df = pd.concat(data)
        df.to_csv('./out/_BB_{}_____nr={}__sigmas={}__N={}__iters={}.csv'.format(ts,     nr,	 sigmas,	    N,	 len(iters)))
        p.close()










