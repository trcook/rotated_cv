# generate cv outputs using kernel regression
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from tqdm import tqdm
asdt=pd.to_datetime
# from util import *
from util import KRExpConfig,KRExpResult,jsonlreader,gen_arma_errors,hv_folder
from joblib import Parallel,delayed
import pydantic
from pydantic.dataclasses import dataclass
from typing import Optional
from time import time
st = time()
import argparse
import multiprocessing
from kern import KR, GaussianKernel,UniformKernel

#%% config
chunksize = 25
GlobalSeed = 1
TESTING =False
parser = argparse.ArgumentParser()
parser.add_argument('--index',help='index in experiments',type=int,default=1)
if TESTING:
    args=parser.parse_args(["--index",'1'])
else:
    args=parser.parse_args()

#%%
# functions 
def loo_hv_hp(w,x,y,y_trend,h=1,v=0):
    """
    returned cv score is compared to observed data
    'true_cv_score' is score compared to true trend
    """
    mod=KR(w,kernel=UniformKernel)
    out=[]
    yhatlist = []
    true_out=[]
    for testidx,trainidx in hv_folder(y,h,v):
        mod.fit(y[trainidx],x[trainidx])
        # plt.plot(x,mod.trend)
        yhat=mod.predict(x[testidx])
        out.append(mod.L(x[testidx],y[testidx]))
        yhatlist.append(yhat.squeeze())
        true_out.append(mod.L(x[testidx],y_trend[testidx]))
    true_cv_score= np.r_[true_out].mean()
    yhats = np.r_[yhatlist]
    cv_score = np.r_[out].mean()
    return cv_score,true_cv_score

def inner_loop(j,exp:KRExpConfig):
    '''
    exp has sigma,theta,phi,n,lamda
    j is random number seed
    output:
    a is cv score
    b is cv score compared to true trend
    '''
    n=exp.n
    rng=np.random.default_rng(j)
    x = np.arange(1,n+1)/n
    y_trend = np.sin(x*3*np.pi)+np.cos(x*2*np.pi)
    err=gen_arma_errors(exp.n,exp.phi,exp.theta,exp.sigma,rng,burnin=200)
    y = y_trend+err
    cv,true_cv=loo_hv_hp(exp.w,x,y,y_trend,h=0,v=0) # Using exp.lam as a proxy for window width 'w'
    true_mse = (err**2).mean()
    return cv,true_cv,true_mse
#%%
def outer_loop(exp,GlobalSeed=GlobalSeed):
    rng = np.random.default_rng(GlobalSeed)
    seeds=rng.integers(0,1000,200) # each experiment is repeated 200 times and averaged
    out=Parallel(n_jobs=8)(delayed(inner_loop)(j,exp) for j in tqdm(seeds)) # don't parallelize within already parallelized routine
    # out = [inner_loop(i,exp) for i in seeds]
    cvs,true_cvs,err = list(zip(*out)) # unzip returned list
    cv = np.mean(cvs)
    true_cv = np.mean(true_cvs)
    err=np.mean(err)
    outdict=exp.__dict__
    outdict['cv']= cv
    outdict['true_cv']=true_cv
    outdict['true_mse']=err
    return KRExpResult(**outdict)

#%%

## load in exps in chunks

exps = jsonlreader(KRExpConfig,'./kernexps.jsonl')
N = len(exps)
chunk_start = max(min((args.index * chunksize),(N-chunksize)),0)
chunk_end = min((chunk_start + chunksize),N)
#%%
exps = exps[chunk_start:chunk_end]
# with Parallel(n_jobs=16,prefer='processes'):

# MP=16

# out=Parallel(n_jobs=MP)(
        # delayed(outer_loop)(i) for i in tqdm(exps)
        # )
out = [outer_loop(i) for i in tqdm(exps)]

# out = [outer_loop(i) for i in exps[:30]]
out = pd.DataFrame(out)
out.iloc[[out.cv.argmin(),out.true_cv.argmin()],:]
out.to_json('./kern_cv_out.jsonl',lines=True,orient='records')

#%%

en = time()

print(en-st)

# pickuphere -- run on cluster
