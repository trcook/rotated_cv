# linear cv in the style of harmonic regression cv

#%%
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from util import gen_arma_errors, hv_folder,jsonlreader,slow_ft
from our_dataclasses import *
from typing import Tuple
from numpy.typing import ArrayLike
from collections import namedtuple
from pydantic.dataclasses import dataclass


#PICKUPHERE: get correct fourier transform from work laptop. generate data with 10 vars and n_vars relevant. check with work laptop -- i may have already written some of this up.


class LinearModel():
    '''
    basic linear model 
    '''
    def __init__(self,intercept=True):
        self.intercept = intercept
        self.coef_ = None
        self.intercept_ = None
    def mk_design_matrix(self,x,y)-> Tuple[ArrayLike,ArrayLike]:
        if self.intercept:
            X= np.c_[np.ones(x.shape[0]),x]
        return X,y
    def fit(self,X,y):
        self.coef_ = np.linalg.lstsq(X,y)[0]
    def predict(self, X):
        return X@self.coef_
    def L(self,yhat,y):
        return (np.abs(y-yhat)**2).mean()
    
@dataclass
class LooResult():
    cv:float=None
    true_cv: float = None
    yhats: Tuple=None

@dataclass
class InnerLoopResult(LooResult):
    best_k:int = None
    true_mse: float = None
    
    



# Experiment configuration
chunksize = 20
GlobalSeed = 1
TESTING = True
parser = argparse.ArgumentParser()
parser.add_argument('--index', help='index in experiments', type=int, default=1)
if TESTING:
    args = parser.parse_args(["--index", '0'])
else:
    args = parser.parse_args()


# Functions
def loo_hv_linear_regression(k, x, y, y_trend, h=0, v=0):
    """
    returned cv score is compared to observed data
    'true_cv_score' is score compared to true trend
    """
    mod=LinearModel(intercept =True)
    out=[]
    yhatlist = []
    true_out=[]
    X,Y= mod.mk_design_matrix(x,y)
    #  insert the fourier transform here to do the 'rotated cv'
    for testidx,trainidx in hv_folder(Y,h,v):
        mod.fit(X[trainidx],Y[trainidx])
        yhat=mod.predict(X[testidx])
        out.append(mod.L(Y[testidx],yhat))
        yhatlist.append(yhat.squeeze())
        true_out.append(mod.L(y_trend[testidx],yhat))
    true_cv_score= np.r_[true_out].mean()
    yhats = np.r_[yhatlist]
    cv_score = np.r_[out].mean()
    return LooResult(cv_score, true_cv_score,yhats)

def inner_loop(j, exp: LinearExpConfig,return_raw=False):
    """
    exp has sigma, theta, phi, n, k
    j is random number seed
    Returns cv_score, true_cv_score, true_mse
    """
    n = exp.n
    rng = np.random.default_rng(j)
    k = 10 # number of independent variables
    x = rng.multivariate_normal(mean=np.zeros(k),cov=np.eye(k),size=n)
    coefs = rng.normal(loc=1,scale = 1/4, size = exp.n_vars)
    coefs = rng.choice([-1,1],exp.n_vars)* coefs
    coefs=np.r_[coefs,np.zeros(k-exp.n_vars)]
    y_trend = x@coefs
    err = gen_arma_errors(exp.n, exp.phi, exp.theta, exp.sigma, rng, burnin=2000)
    y = y_trend + err
    chosen = LooResult(cv=np.inf,true_cv=np.inf,yhats = [])
    best_k = None
    for i in range(1,k+1):
        candidate=loo_hv_linear_regression(exp.n_vars, x[:,:i], y, y_trend, h=0, v=0)
        if candidate.cv<chosen.cv:
            chosen = candidate
            best_k = i
    # cv, true_cv,yhats = loo_hv_linear_regression(exp.n_vars, x, y, y_trend, h=0, v=0)
    true_mse = (err**2).mean()
    if return_raw:
        
        # return chosen.cv,chosen.true_cv,true_mse,chosen.yhats,best_k
        return InnerLoopResult(**chosen.__dict__,true_mse = true_mse,best_k=best_k)
    else:
        out = InnerLoopResult(**chosen.__dict__,true_mse = true_mse,best_k=best_k)
        out.yhats = None
        # return chosen.cv,chosen.true_cv,true_mse,best_k
        return out


def outer_loop(exp, GlobalSeed=GlobalSeed):
    rng = np.random.default_rng(GlobalSeed)
    seeds = rng.integers(0, 1000, 200)  # 200 repetitions
    out = [inner_loop(i, exp) for i in seeds]
    # cvs, true_cvs, errs,best_k = list(zip(*out))
    cv = np.mean([i.cv for i in out])
    true_cv = np.mean([i.true_cv for i in out])
    err = np.mean([i.true_mse for i in out])
    k = np.mean([i.best_k for i in out])
    outdict = exp.__dict__
    outdict['cv'] = cv
    outdict['true_cv'] = true_cv
    outdict['true_mse'] = err # lower bound on cv score
    outdict['n_chosen_vars'] = k

    return LinearExpResult(**outdict)

#%%
# Main execution
exps = jsonlreader(LinearExpConfig, './linearexps.jsonl')
N = len(exps)
chunk_start = max(min((args.index * chunksize), (N - chunksize)), 0)
chunk_end = min((chunk_start + chunksize), N)

exps = exps[chunk_start:chunk_end]

MP = multiprocessing.cpu_count()
out = Parallel(n_jobs=MP)(
    delayed(outer_loop)(i) for i in tqdm(exps)
)
#%%
out_df = pd.DataFrame(out)
out_df.iloc[[out_df.cv.argmin(), out_df.true_cv.argmin()], :]
out_df.to_json('./linear_cv_out.jsonl', lines=True, orient='records')
