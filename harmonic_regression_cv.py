# this script tests the Fourier.py module similar to how we test it in hp_cv.py
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from HarmonicRegression import HarmonicRegressionModel
from util import gen_arma_errors, hv_folder,jsonlreader
from our_dataclasses import *
#%%
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

#%%
# Functions
def loo_hv_harmonic_regression(k, x, y, y_trend, h=0, v=0):
    """
    returned cv score is compared to observed data
    'true_cv_score' is score compared to true trend
    """
    mod=HarmonicRegressionModel(n_harmonics=k,intercept =True)
    out=[]
    yhatlist = []
    true_out=[]
    X,Y= mod.mk_design_matrix(x,y)
    #  insert the fourier transform here to do the 'rotated cv'
    for testidx,trainidx in hv_folder(Y,h,v):
        mod.fit(X[trainidx],Y[trainidx])
        # plt.plot(x,mod.trend)
        yhat=mod.predict(X[testidx])
        out.append(mod.L(Y[testidx],yhat))
        yhatlist.append(yhat.squeeze())
        true_out.append(mod.L(y_trend[testidx],yhat))
    true_cv_score= np.r_[true_out].mean()
    yhats = np.r_[yhatlist]
    cv_score = np.r_[out].mean()
    return cv_score, true_cv_score

def inner_loop(j, exp: HRExpConfig):
    """
    exp has sigma, theta, phi, n, k
    j is random number seed
    Returns cv_score, true_cv_score, true_mse
    """
    n = exp.n
    rng = np.random.default_rng(j)
    x = np.arange(1, n+1)/n
    y_trend = np.sin(x * 3 * np.pi) + np.cos(x * 2 * np.pi)
    err = gen_arma_errors(exp.n, exp.phi, exp.theta, exp.sigma, rng, burnin=200)
    y = y_trend + err
    cv, true_cv = loo_hv_harmonic_regression(exp.n_harmonics, x, y, y_trend, h=0, v=0)
    true_mse = (err**2).mean()
    return cv, true_cv, true_mse

def outer_loop(exp, GlobalSeed=GlobalSeed):
    rng = np.random.default_rng(GlobalSeed)
    seeds = rng.integers(0, 1000, 200)  # 200 repetitions
    out = [inner_loop(i, exp) for i in seeds]
    cvs, true_cvs, errs = list(zip(*out))
    cv = np.mean(cvs)
    true_cv = np.mean(true_cvs)
    err = np.mean(errs)
    outdict = exp.__dict__
    outdict['cv'] = cv
    outdict['true_cv'] = true_cv
    outdict['true_mse'] = err # lower bound on cv score
    return HRExpResult(**outdict)

#%%
# Main execution
exps = jsonlreader(HRExpConfig, './hrexps.jsonl')
N = len(exps)
chunk_start = max(min((args.index * chunksize), (N - chunksize)), 0)
chunk_end = min((chunk_start + chunksize), N)

exps = exps[chunk_start:chunk_end]

MP = multiprocessing.cpu_count()
out = Parallel(n_jobs=MP)(
    delayed(outer_loop)(i) for i in exps
)

out_df = pd.DataFrame(out)
out_df.iloc[[out_df.cv.argmin(), out_df.true_cv.argmin()], :]
out_df.to_json('./harmonic_regression_cv_out.jsonl', lines=True, orient='records')
