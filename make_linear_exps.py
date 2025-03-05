# do linear regression hv experiments
#%%
# make and save hv_experiments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
asdt=pd.to_datetime
from util import *
from util import jsonlwriter,jsonlreader 
from our_dataclasses import LinearExpConfig
from joblib import Parallel,delayed
import pydantic
from pydantic.dataclasses import dataclass
from typing import Optional
from time import time
import pickle
import itertools

#%%
nphis = 5
nthetas = 5
nsigmas=5
n_vars=np.arange(1,10) #number of correct variables. 

phis = np.linspace(-.9,.9,nphis*2)
thetas =np.linspace(-.9,.9,nphis*2)
sigmas = np.linspace(0.1,1.0,nsigmas)

n=np.r_[40,np.arange(200,1200,200)]
n = np.r_[40,100,200,300]




exps=itertools.product(
    phis,thetas,sigmas,n,n_vars
)

exps=[LinearExpConfig(phi=i[0],theta=i[1],sigma=i[2],n=i[3],n_vars=i[4]) for i in exps]

jsonlwriter(exps,"linearexps.jsonl")