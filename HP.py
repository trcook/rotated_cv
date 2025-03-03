# functions for hp filter
#%%
import scipy
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

#%%




class HP_filter(object):
    """hp filter regression"""
    def __init__(self,lam,x,y) -> None:
        self.lam=lam
        self.D_,self.y_aug_ = self.make_design(x,y)
        self.n = x.shape[0]
    def hp_filter_matrix(self,n, λ):
        """
        Creates the matrix representation of HP filter
        
        Parameters:
        n (int): Length of the time series
        λ (float): Smoothing parameter
        
        Returns:
        X (sparse matrix): Design matrix for the HP filter
        """
        # Identity matrix
        I = sparse.eye(n)
        
        # Second difference matrix
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
        
        # HP filter matrix
        X = sparse.vstack([I, np.sqrt(λ) * D])
        
        return X
    def make_design(self,x,y):
        n=y.shape[0]
        x=self.hp_filter_matrix(n,self.lam).toarray() # may need to revisit here to use sparse arrays for solving
        y_aug = np.concatenate([y,np.zeros(n-2)])
        return x,y_aug
    def get_train_slice(self,idx):
        """
        given a set of indices on the original input (y), get transformed set of indicies in the desgin matrix (TODO: remove trend component in design matrix too)
        """
        out = list(set(idx).union(range(self.n,(self.n+self.n-2)))) # maybe union with self.n+set(idx) instead...
        return out
    def fit_(self,x,y):
        self.b_=np.linalg.solve((x.T@x),(x.T@y))
        return self.b_
    def fit(self,train_idx):
        """
        x should be training index, a list or integer array
        """
        idx=self.get_train_slice(train_idx)
        x=self.D_[idx,:]
        y= self.y_aug_[idx]
        self.fit_(x,y)
    @property
    def trend(self):
        return self.b_
    def predict(self,test_index):
        return self.D_[test_index]@self.b_
    def L(self,y,yhat):
        "give empirical loss -- difference between estimated trend and observed data"
        return np.sum((y-yhat)**2)

        
#%%
#example
#from util import *
# n=100
# lam = 1600
# x = np.arange(1,n+1)/n
# y = np.sin(x*3*np.pi)+np.cos(x*2*np.pi)+np.random.normal(0,.4,n)

# mod=HP_filter(lam,x,y)
# out=[]
# yhatlist = []
# for testidx,trainidx in hv_folder(y,1,0):
#     mod.fit(trainidx)
#     # plt.plot(x,mod.trend)
#     yhat=mod.predict(testidx)
#     out.append(mod.L(y[testidx],yhat))
#     yhatlist.append(yhat.squeeze())
# yhats = np.r_[yhatlist]
# plt.plot(x,mod.trend,color='red',lw=1)
# plt.plot(x,y)
# plt.scatter(x,yhats)
# cv_score = np.r_[out].mean()
# print(cv_score)


# %%
# '''
# (1) y  = t + c
# let y be vector length T
# let x be identity matrix of length T
# rewrite (1) as 
# (2) y = xb + c = b+c (because x is just identity and be is T-length vector)

# Estimate b as 
# (3) minimize_b ||y-b||^2  + ||lambda * D_2(b)||^2
# where D_2 is (centered) second difference operator  b: D_2(b_t) = b_(t+1)-2 b_(t) - b_(t-1) and D_2(b) is length T-2 (the first and last observations must be dropped to accommodate the lag structure)

# (3) can be satisfied by ols written as:
# bhat = (Z' Z)^1 (Z' y_aug)
# where yhat is [y',0_(T-2)']', where 0_(T-2) is a vector of zeros of length T-2.
# and where Z is [I_T, D_2] -- stacking x and the second difference matrix

# To produce new pred
# '''