#%%
import pandas as pd
import numpy as np
# from util import *
from scipy.stats import norm
import scipy.optimize as optim

class KERN(object):
    def __init__(self):
        pass

class GaussianKernel (KERN):
    def __init__(self,w):
        super().__init__()
        self.w=1/np.sqrt(2 * np.pi ) * w

    def __call__(self,a):
        return norm.pdf(a/self.w,0,self.w)

class UniformKernel (KERN):
    def __init__(self,w):
        super().__init__()
        self.w = w
    def __call__(self,a):
        condarray = np.abs(a)<=self.w
        out = np.where(condarray,1/self.w,0.0).T
        return out
        if np.abs(a) < self.w:
            return 1/self.w
        else:
            return 0.0


class KR(object):
    def __init__(self,w,kernel: KERN | None =None):
        self.w=w
        if kernel:
            self.k = kernel(w)
        else:
            self.k = GaussianKernel(w)

    def mk_K(self,x,p):
        # add code to handle more than (1,) shape arrays
        outshape = (x.shape[0],p.shape[0])
        a = np.abs(
            (x.reshape(-1,1)*np.ones(outshape)) - (p*np.ones(outshape))
            )
        out = self.k(a)

        # out = self.k(np.c_[x]*np.ones_like(x)-p)
        return out
        m=[]
        for i in p:
            m.append(
                np.apply_along_axis(lambda a: self.k(a),1,(x-i).reshape(-1,1)).squeeze()
                )
        m=np.r_[m]
        return m
    def fit(self,y,x):
        self.x_=x
        self.m_ = self.mk_K(x,x)
        self.y_=y
    def predict(self,newx):
        # add code to handle more than 1 var
        m = self.mk_K(self.x_,newx)
        return m@self.y_/m.sum(1)
    def L(self,x,y):
        '''
        requires fit first
        '''
        yhat = self.predict(x)
        return ((y-yhat)**2).sum()




def opt_w(mod,newx,newy,x,y,starting_w = np.r_[.5]):
    '''
    find optimal window for a given model
    '''
    def refit_and_loss(w,mod=mod):
        mod=mod.__class__(w,mod.k.__class__)
        mod.fit(y,x)
        return mod.L(newx,newy)
    res =optim.minimize(refit_and_loss,method='Nelder-Mead',x0=starting_w)
    return res

def opt_w_line(mod,newx,newy,x,y,line=np.linspace(0.01,1,100)):
    '''
    find optimal window for a given model
    '''
    def refit_and_loss(w,mod=mod):
        mod=mod.__class__(w,mod.k.__class__)
        mod.fit(y,x)
        L=mod.L(newx,newy)
        if np.isnan(L):
            return np.inf
        return L
    out = []
    for i in line:
        res=refit_and_loss(i,mod=mod)
        out.append([i,res])
    return np.r_[out]


# %%
