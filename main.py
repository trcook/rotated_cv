#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
asdt=pd.to_datetime
from typing import Generator
from numpy.typing import ArrayLike


def slow_ft(x,intercept=True):
    n = x.shape[0]
    J = n//2
    vecs = []
    for i in range(J+1):
        if i == 0:
            vecs.append(np.cos(2 * np.pi * i/n* x).reshape(-1,1))
        elif i/n<.5:
            vecs.append(
                np.c_[
                    np.cos(2*np.pi * i/n * x),
                    np.sin(2 * np.pi * i/n * x)
                ]
            )
        elif np.isclose(i/n,0.5):
            vecs.append(np.cos(2*np.pi*1/n*x).reshape(-1,1))
    return np.hstack(vecs)


def hv_folder(arr:ArrayLike,h:int,v:int) -> Generator:
    """
    h = gap size between test and train 
    v = size of test set * .5
    returned train set will be N - (2*v+1) - 2*h
    returns 
    (test-index, train-index)
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



#%% data gen
n = 100

sigma =1.5
phi = 0
theta = 0
burnin = 200
rng = np.random.default_rng(1)
x = np.arange(1, n+1)
y_trend = np.sin(x/n * 3 * np.pi) + np.cos(np.sqrt(x/n) * 2 * np.pi)

u = rng.normal(0,sigma,n+burnin)
# this is a slow way to make this
epsilon = np.zeros_like(u)
epsilon[0]=0
for i in range(1,n+burnin):
    epsilon[i]=phi * epsilon [i-1] + u[i-1] * theta +u[i]
err = epsilon[burnin:]



y = y_trend + err

#%%
# one pass with normal harmonic regression
k = 7
X=slow_ft(x)[:,:k]

# b=np.linalg.inv(X.T@X)@(X.T@y)
b=np.linalg.lstsq(X,y)[0]

yhat = (X@b)
plt.plot(yhat)
plt.plot(y,alpha = .3,lw=3)
plt.plot(y_trend)

#%%
# one pass with 'rotated' regression -- should produce same coefficients
z = np.arange(n)
Z=slow_ft(z)*(2/np.sqrt(n))
zx=Z.T@X
zy=Z.T@y

b=np.linalg.lstsq(zx,zy)[0]

plt.plot(zy)
plt.plot(zx@b)
#%%

out = []
for i in range(50):

    k = i
    X=slow_ft(x)[:,:k]

    # just run hv cv scores
    folder = hv_folder(y,0,0)
    hv=np.array([])
    yhats = np.array([])
    hv_true = np.array([])
    for test,train in folder:


        b=np.linalg.inv(X.T@X)@(X.T@y)
        # b=np.linalg.lstsq(X,y)[0]

        yhat = (X[test]@b)
        hv=np.r_[hv,(y[test]-yhat)**2]
        yhats = np.r_[yhats,yhat]
        hv_true = np.r_[hv_true,(y_trend[test]-yhat)**2]
    out.append(dict(k=i,hv=hv.mean(),hv_true=hv_true.mean()))
out=pd.DataFrame(out)
out.sort_values("hv",inplace=True)
out


#%%
# using rotated_cv
# make rotated output and design matricies

out = []
for i in range(50):
    k = i
    X=slow_ft(x)[:,:k]

    z = np.arange(n)
    Z=slow_ft(z)*(2/np.sqrt(n))
    zx=Z.T@X
    zy=Z.T@y
    zy_trend = Z.T@y_trend


    folder = hv_folder(y,0,0)
    hv=np.array([])
    hv_true=np.array([])
    yhats = np.array([])
    for test,train in folder:


        b=np.linalg.inv(X.T@X)@(X.T@y)
        # b=np.linalg.lstsq(zx[train],zy[train])[0]

        yhat = (zx[test]@b)
        hv=np.r_[hv,(y[test]-yhat)**2]
        hv_true = np.r_[hv_true,(zy_trend[test]-yhat)**2]
        yhats = np.r_[yhats,yhat]
    out.append(dict(k=i,hv=hv.mean(),hv_true=hv_true.mean(),yhats=yhats))
out=pd.DataFrame(out)
out.sort_values("hv",inplace=True)
out
#%%

print(f"rotated_cv error: {hv.mean()}")
print(f"rotated_cv error to trend: {hv_true.mean()}")
plt.plot(out.iloc[0,:].yhats)
plt.plot(zy,alpha =.3,lw=3)
plt.plot(Z.T@y_trend)

