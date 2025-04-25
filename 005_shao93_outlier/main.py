#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
asdt=pd.to_datetime
from typing import Generator
from numpy.typing import ArrayLike
from tqdm import tqdm
from joblib import Parallel, delayed
# from statsmodels.tsa.api import acf
import itertools
from  sklearn.model_selection import KFold
from itertools import product
from pydantic.dataclasses import dataclass
from typing import Optional
import joblib

#%%
def slow_cos(x,intercept=True):
    """
    gives cosine transform matrix P
    transform vectors with left multipication by P: P@x
    """
    n=x.shape[0]
    K=np.arange(n)
    vecs=[]
    for i in range(n):
        # vecs.append(np.c_[np.cos((np.pi/n) *(i+1/2) * K)])
        vecs.append(np.c_[np.cos(2*np.pi*i*K/n)]*np.sqrt(2/n))
        # vecs.append(np.c_[np.sin(2*np.pi*i*K/n)]*np.sqrt(2/n))
    return np.hstack(vecs)
def slow_ft(x,intercept=True):
    # this gives harmonic components
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

def rand_diag(x,intercept=True):
    n=x.shape[0]
    d=np.random.normal(0,1,n)
    return np.eye(n)*d

def exp_rand_diag(x,intercept=True,lamb=0.5):
    n=x.shape[0]
    d = np.random.exponential(scale=lamb,size=n)
    return np.eye(n)*d


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

def k_folder(arr:ArrayLike,k=10,rng = np.random.default_rng(1)):
    # will truncate if arr does not fold evenly.
    n = arr.shape[0]
    n_k = n//k
    print(n_k)
    folds=rng.choice(range(n),size=(k,n_k),replace=False,shuffle=True)
    for i in range(k):
        train = folds[i,:]
        test = np.array(
            list(set(range(n))-set(train))
        )
        yield train,test

#%% data gen

betas = [
[2,0,0,4,0],
[2,0,0,4,8],
[2,9,0,4,8],
[2,9,6,4,8]
]

def data_gen(j,betas=betas):
    rng=np.random.default_rng(seed=j)
    n = 200
    ko=1


    # rng = np.random.default_rng(1)
    x=rng.standard_t(df=3,size=[n,5])
    e = rng.normal(0,1,n)
    # x[:ko,:]= x[:ko,:]*4
    # e[:ko]= e[:ko]*4

    Ys=[]
    for  b in betas:
        out = x@np.array(b).reshape(-1,1) + e.reshape(-1,1)
        Ys.append(out)
    return x,Ys

x,Ys=data_gen(None)
H = x@np.linalg.inv(x.T@x)@(x.T)
plt.plot(np.diag(H))

#%%
# non-rotated cv
@dataclass(config=dict(arbitrary_types_allowed=True))
class LOO_result:
    b:  ArrayLike
    cv_score: float
    yhat: Optional[ArrayLike]= None
    y: Optional[ArrayLike] = None
    num_vars:int=None


coef_combinations=list(product(*[[False,True] for i in range(len(betas[0]))]))
outcv=[]
for y in Ys:
    y_outcv=[]
    Y=y.copy()
    for i in tqdm(coef_combinations):
        X=x.copy()
        X[:,[j==False for j in i]]=0

        folder = hv_folder(y,0,0)
        i_outcv=[]
        for test,train in folder:
            b=np.linalg.lstsq(X[train,:],y[train])[0]
            yhat = (X[test]@b).squeeze()[()]
            hv=np.mean((y[test]-yhat)**2).squeeze()[()]
            i_outcv.append(LOO_result(b=b,yhat=yhat,cv_score=hv,y=y[test].squeeze()[()]))
        i_outcv=pd.DataFrame(i_outcv)
        hv = i_outcv.cv_score.mean()
        b = np.stack(i_outcv.b).mean(0)

        
        yhat = i_outcv.yhat.values
        y_outcv.append(pd.DataFrame([LOO_result(b=b,yhat=yhat,y=i_outcv.y.values,cv_score=hv,num_vars=np.array(i).sum())]))
    outcv.append(pd.concat(y_outcv).sort_values('cv_score'))
        
outcv[0]



#%%

# # rotated cv
# rng = np.random.default_rng()
# B=2000
# ind = np.arange(x.shape[0])
# coef_combinations=list(product(*[[False,True] for i in range(len(betas[0]))]))
# rotated_outcv=[]
# N = np.arange(x.shape[0])
# P=slow_cos(N)
# for y in Ys:
#     y_outcv=[]


#     for i in tqdm(coef_combinations):
#         i_outcv=[]
#         for b in range(B):
#             b_index = rng.choice(ind,replace=True,size=ind.shape[0])
#             X=x[b_index,:].copy() # copy original x 
#             X=P@X # rotate
#             X[:,[j==False for j in i]]=0
#             Y=y[b_index,:].copy()
#             Y=P@Y

#             folder = hv_folder(y,0,0)
#             for test,train in folder:
#                 b=np.linalg.lstsq(X[train,:],Y[train])[0]
#                 yhat = (X[test]@b).squeeze()[()]
#                 hv=np.mean((Y[test]-yhat)**2).squeeze()[()]
#                 i_outcv.append(LOO_result(b=b,yhat=yhat,cv_score=hv,y=y[test].squeeze()[()]))

            
#         i_outcv=pd.DataFrame(i_outcv)
#         hv = i_outcv.cv_score.mean()
#         b = np.stack(i_outcv.b).mean(0)

#         yhat = i_outcv.yhat.values
#         y_outcv.append(pd.DataFrame([LOO_result(b=b,yhat=yhat,y=i_outcv.y.values,cv_score=hv,num_vars=np.array(i).sum())]))
#     rotated_outcv.append(pd.concat(y_outcv).sort_values('cv_score'))
        
# rotated_outcv[0]
# print(outcv[1].iloc[0].b.round(2))
# print(rotated_outcv[1].iloc[0].b.round(2))
#%% refactor here
def loo(x,y,b):
    """
    just calculate leave one out cv score for given beta -- can't use jacknife because we are using a pre-specified beta
    """
    N=x.shape[0]
    folder= hv_folder(y,0,0)
    outcv=[]
    for test,train in folder:
        yhat = x[test]@b
        cv_score=(y[test]-yhat)**2
        outcv.append(cv_score)
    return np.mean(outcv)
def rotate_cv(x,y,rot_fn=slow_cos):
    N=x.shape[0]
    k = np.linalg.matrix_rank(x)
    P=rot_fn(x)
    X=(P@x).copy()
    Y=(P@y).copy()
    folder = hv_folder(y,0,0)
    out=[]
    for test,train in folder:
        b=np.linalg.lstsq(X[train,:],Y[train])[0]
        yhat = (X[test]@b).squeeze()[()]
        hv=np.mean((Y[test]-yhat)**2).squeeze()[()]
        out.append(LOO_result(b=b,yhat=yhat,cv_score=hv,y=Y[test].squeeze()[()]))
    out=pd.DataFrame(out)
    out=LOO_result(b=out.b.mean(),cv_score=out.cv_score.mean(),num_vars=k)
    return out

def jacknife_fast(x,y,rot_fn=slow_cos):
    """
    gives correct result up to like 5 or 6 places
    Replaces 'rotate_cv'
    """
    N=x.shape[0]
    P=rot_fn(x)
    X=(P@x).copy() # rotate X
    Y=(P@y).copy() # rotate y 
    
    b=np.linalg.lstsq(X,Y)[0]
    k = X.shape[1]
    H=X@np.linalg.inv(X.T@X)@X.T # hat matrix
    I=np.eye(X.shape[0])
    D=I*(1/(1-np.diag(H)) )
    Htilde=D@(H-I)+I
    ehat=Y-(H@Y)
    etilde=D@ehat
    # yhat = Htilde@Y # gives out of sample estimate of yhat
    cv_score=np.mean(etilde**2)
    # now generate mu=E(y|XB) for unrotated using fitted vars
    return LOO_result(b=b, cv_score = cv_score,num_vars = k)

def boot_2(B,x,y,rot_fn=slow_cos,j=None):
    """
    one iteration bootstrap.
    1. run jacknife fast
    """

    rng=np.random.default_rng()
    N=x.shape[0]
    ind= np.arange(N)
    b_index = rng.choice(ind,replace=True,size=N)
    X=x[b_index,:].copy()
    Y=y[b_index,:].copy()
    return jacknife_fast(X,Y,rot_fn)

def boot_1(B,x,y,rot_fn=slow_cos,j=None):
    rng=np.random.default_rng()
    N=x.shape[0]
    ind= np.arange(N)
    b_index = rng.choice(ind,replace=True,size=N)
    X=x[b_index,:].copy()
    Y=y[b_index,:].copy()
    return rotate_cv(X,Y,rot_fn)

def bootstrap(B,x,y,rot_fn=slow_cos,j=None):
    rng=np.random.default_rng()
    N=x.shape[0]
    ind= np.arange(N)
    
    out=[boot_2(i,x,y,slow_cos) for i in range(B)]
    out=pd.DataFrame(out)
    out=LOO_result(b=out.b.mean(),cv_score=out.cv_score.mean(),num_vars=out.num_vars.mean())
    return out
    for i in range(B):
        b_index = rng.choice(ind,replace=True,size=N)
        X=x[b_index,:].copy()
        Y=y[b_index,:].copy()
        out.append(rotate_cv(X,Y,rot_fn))
    out=pd.DataFrame(out)
    out=LOO_result(b=out.b.mean(),cv_score=out.cv_score.mean(),num_vars=out.num_vars.mean())
    return out
def subset1(B,x,y,coefs):
    X=x.copy()
    X=X[:,coefs]
    
    out=bootstrap(B,X,y)

    return_betas=np.zeros(len(coefs))
    return_betas[np.where(coefs)]=out.b.squeeze()[()]
    out.b = return_betas
    return out
def subset_select(x,y,B=2):
    coef_combinations = list(product(*[[False,True] for i in range(x.shape[1])]))
    rotated_outcv=[]
    rotated_outcv=Parallel(n_jobs=32)(delayed(subset1)(B,x,y,coefs) for coefs in coef_combinations)
    # for i in tqdm(coef_combinations):
    #     X = x.copy()
    #     X[:,[j==False for j in i]]=0
    #     rotated_outcv.append(bootstrap(B,X,y))
    out=pd.DataFrame(rotated_outcv)
    out.sort_values("cv_score",inplace=True)
    return out
#%% outer loop -- parallelize here

def outer_loop_1(niters,Ys_k=1,betas=betas,B=2):
    x,Ys=data_gen(j=i,betas=betas)
    y=Ys[Ys_k]
    ss=subset_select(x,y,B=B)
    # FIX HERE
    # Take best fit model and fit mean prediction. generate leave one out cv score and save to compare with 
    best_b = ss.iloc[0,:].b.reshape(-1,1)
    yhat=x@best_b
    errhat=np.mean((y-yhat)**2)
    ss.loc[:,"err_hat"]=np.nan
    ss.at[ss.index[0],"err_hat"]=errhat
    # loo with best b
    return ss

# def outer_loop(niters,Ys_k=1,betas=betas,B=2):
#     out=[]
#     for i in tqdm(range(niters)):
#         x,Ys=data_gen(j=i,betas=betas)
#         y=Ys[Ys_k]
#         ss=subset_select(x,y,B=B)
#         out.append(ss)
#     return out    
    
#     ind = np.arange(x.shape[0])
#     coef_combinations=list(product(*[[False,True] for i in range(len(betas[0]))]))
#     rotated_outcv=[]
#     N = np.arange(x.shape[0])
#     P=slow_cos(N)
#     for y in Ys:
#         y_outcv=[]
#         for i in tqdm(coef_combinations):
#             i_outcv=[]
#             y_outcv.append(pd.DataFrame([bootstrap(B,x,y,j=j)]))
#         rotated_outcv.append(pd.concat(y_outcv).sort_values('cv_score'))
#     return rotated_outcv
#%%
# out = Parallel(n_jobs=2)(delayed(outer_loop_1)(i,Ys_k=1,betas=betas,B=2000) for i in tqdm(range(2000)))
out =[ outer_loop_1(i,Ys_k=1,betas=betas,B=2000) for i in tqdm(range(2000))]
# out=outer_loop(2,Ys_k=1,betas=betas,B=2000)
joblib.dump(out,'./output.gz')
#%%
# out=Parallel(n_jobs=20)(delayed(outer_loop)(j) for j in tqdm(range(2)))
# import joblib
# joblib.dump(out,"./output.gz")

#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%
"""

def outer_loop(y,x,n,j):
    z=np.arange(n)

    Z=rand_diag(z)
    T = np.arange(x.shape[0])
    P = slow_ft(x)
    F = slow_ft(x)
    inner_out = []
    for i in range(n//3):
        k=i
        X=slow_ft(x)[:,:k]
        zx=F.T@X
        zy=P.T@y
        # zy_trend=P.T@y_trend
        # just run hv cv scores
        folder = hv_folder(y,h,v)
        hv=np.array([])
        yhats = np.array([])
        hv_true = np.array([])
        
        for test,train in folder:
            b=np.linalg.lstsq(zx[train],zy[train])[0]
            yhat = (zx[test]@b)
            hv=np.r_[hv,(zy[test]-yhat)**2]
            yhats = np.r_[yhats,yhat]
            # compare in-sample fit to trend
            yhat = zx@b
            # hv_true = np.r_[hv_true,((zy_trend-yhat)**2).mean()]
        inner_out.append(dict(k=k,hv=hv.mean(),hv_true=hv_true.mean(),yhats=yhats,group=j))
    return inner_out

out=Parallel(n_jobs=20)(delayed(outer_loop)(y,y_trend,x,n,j) for j in tqdm(range(2000)))

out=list(itertools.chain.from_iterable(out))

out=pd.DataFrame(out)
#%%
out.groupby('k').mean().sort_values("hv_true")
# out.sort_values("hv")
#%%

print(f"rotated_cv error: {hv.mean()}")
print(f"rotated_cv error to trend: {hv_true.mean()}")
plt.plot(out.iloc[0,:].yhats)
plt.plot(zy,alpha =.3,lw=3)
plt.plot(Z.T@y_trend)


"""