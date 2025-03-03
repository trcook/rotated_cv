#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
asdt=pd.to_datetime
from scipy.signal import lfilter
from statsmodels.tsa.tsatools import add_lag
from numpy.typing import ArrayLike
from typing import Generator
import itertools
from scipy import signal
from scipy.linalg import toeplitz
from kern import *
from HP import *
from pydantic import BaseModel,RootModel
from pydantic.dataclasses import dataclass
from textwrap import dedent
from itertools import product
from typing import Optional
import json

def hv_folder(arr:ArrayLike,h:int,v:int) -> Generator:
    """
    h = gap size between test and train 
    v = size of test set * .5
    returned train set will be N - (2*v+1) - 2*h
    returns 
    validation set, test set indices
    """    
    idx=np.arange(0,len(arr))
    fold_start = v
    fold_end = len(arr)-v
    for i in range(fold_start,fold_end): 
        idx_v = idx[i-v:i+v+1] #plus 1 b/c of zero-indexing
        idx_c = np.setdiff1d(idx,idx_v)
        hvbottom = max([i-v-h,0])
        hvtop = min([i+v+h,len(arr)])
        idx_c = np.concatenate([
            idx[:hvbottom],
            idx[hvtop+1:] #plus q b/c of zero indexing
            ])
        yield [
            idx_v,
            idx_c,
            # idx[hvbottom:i-v],
            # idx[i+v+1:hvtop+1],
            # i,v,h,hvtop,hvbottom
            ]

#%%

def ar_gen(m, n, beta, correlation=0.5, sigma=1.0, random_seed=None):
    """
    Simulate data from the model y = X β + ε where ε has a non-diagonal covariance matrix.
    Parameters:
    -----------
    m : int
    Number of observations
    n : int
    Number of features/regressors
    beta : array-like
    True parameter vector of shape (n,)
    correlation : float, optional
    Correlation coefficient between adjacent error terms (default: 0.5)
    sigma : float, optional
    Standard deviation of the errors (default: 1.0)
    random_seed : int, optional
    Seed for reproducibility
    Returns:
    --------
    y : ndarray
    Dependent variable of shape (m,)
    X : ndarray
    Design matrix of shape (m,n)
    Sigma : ndarray
    True covariance matrix of the errors
    e : ndarray
    Generated errors
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # Generate design matrix X
    X = np.random.normal(0, 1, size=(m, n))
    # Create covariance matrix for errors (AR(1) structure)
    Sigma = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Sigma[i,j] = sigma**2 * correlation**abs(i-j)
    # Generate correlated errors using Cholesky decomposition
    L = np.linalg.cholesky(Sigma)
    e = np.dot(L, np.random.normal(0, 1, size=m))
    # Generate dependent variable
    y = np.dot(X, beta) + e
    return y, X, Sigma, e


def arma_gen(n,m,phi,theta,beta=[.2,.3,.1,0],random_seed=None):
    '''
    generate with arma(1,1) correlated errors
    '''
    if m>0:
        X=np.random.multivariate_normal(mean=np.zeros(m),cov=np.eye(m),size=n)
        if len(beta)>m:
            beta=np.array(beta[:m]).reshape(-1,1)
        elif len(beta)<m:
            beta= np.concat([beta,np.full((m-len(beta)),0.0)]).reshape(-1,1)
        else:
            beta = np.reshape(beta,(-1,1))
    else:
        beta = np.zeros((1,1))
        X = None
    u=np.random.normal(0,1,n)
    epsilon = np.zeros_like(u)
    epsilon[0]=0
    for i in range(1,n):
        epsilon[i] = phi * epsilon[i-1] + u[i-1] * theta +u[i]
    if X is not None:
        y = X@beta + epsilon.reshape(-1,1)
    else:
        y = epsilon.reshape(-1,1)
    return y, X, u,beta
    

#%%
def gen_arma_errors(n,phi,theta, sigma,rng = np.random.default_rng(1),burnin = 200):
    '''
    n:number of observations
    phi: ar(1) component (e.g. .3,.5,etc)
    theta: ma(1) component (e.g. .1,.4)
    sigma: standard deviation of white noise error
    rng: random number error generator
    returns: a sample from an arma (1,1) process of length n
    '''
    u = rng.normal(0,sigma,n+burnin)
    # this is a slow way to make this
    epsilon = np.zeros_like(u)
    epsilon[0]=0
    for i in range(1,n+burnin):
        epsilon[i]=phi * epsilon [i-1] + u[i-1] * theta +u[i]
    return epsilon[burnin:]

    

#%% util functions


def jsonlwriter(x:list,outpath):
    outlist=[RootModel[i.__class__](i).model_dump_json() for i in x]
    with open(outpath,'w') as f:
        for i in outlist:
            f.write(i)
            f.write("\n")

def jsonlreader(cls,fname):
    # works just as well if cls=dict
    outlist=[]
    with open(fname,'r') as f:
        for i in f:
            outlist.append(cls(**json.loads(i)))
    return outlist

#%% dataclasses



@dataclass
class HPExpConfig():
    sigma: Optional[float] = 0.0
    theta:Optional[float] = 0.0
    phi: Optional[float] = 0.0
    n: Optional[int] = 200
    lam:int = 1600


@dataclass
class HPExpResult(HPExpConfig):
    cv: float = None
    true_cv: float = None # indicates how close to trend component
    true_mse: float = None # indicates variance of disturbance term (lower limit of cv score without overfitting)

@dataclass 
class KRExpConfig():
    sigma: Optional[float] = 0.0
    theta:Optional[float] = 0.0
    phi: Optional[float] = 0.0
    n: Optional[int] = 200
    w:Optional[float] = .01

@dataclass
class KRExpResult(KRExpConfig):
    cv: float = None
    true_cv: float = None # indicates how close to trend component
    true_mse: float = None # indicates variance of disturbance term (lower limit of cv score without overfitting)
