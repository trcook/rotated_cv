#%%
# make and save hv_experiments for harmonic_regression

from util import *
from util import jsonlwriter,jsonlreader 
from our_dataclasses import HRExpConfig
from joblib import Parallel,delayed
import itertools

#%%
nphis = 5
nthetas = 5
nsigmas=5
phis = np.linspace(-.9,.9,nphis*2)
thetas =np.linspace(-.9,.9,nphis*2)
sigmas = np.linspace(0.1,1.0,nsigmas)
ws = np.linspace(.01,1.0,100)
ws = np.arange(2,100)/100
n_harmonics = np.arange(1,31)
n=np.r_[40,np.arange(200,1200,200)]
n = np.r_[40,100,200,300]




exps=itertools.product(
    phis,thetas,sigmas,n,n_harmonics
)

exps=[HRExpConfig(phi=i[0],theta=i[1],sigma=i[2],n=i[3],n_harmonics=i[4]) for i in exps]

jsonlwriter(exps,"hrexps.jsonl")