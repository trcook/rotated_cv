#%%
import typing
import scipy
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from typing import Tuple
#%%

class HarmonicRegressionModel():
    def __init__(self,n_harmonics=3,intercept=True):
        # setup initial configuration
        self.n_harmonics = n_harmonics
        self.intercept = intercept
    def extract_harmonics(self, X):
        """
        Transform input data into Fourier features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        X_fourier : array-like of shape (n_samples, n_features * (2 * n_harmonics ))
            Transformed input data with Fourier features.
        """
        if len(X.shape)==1:
            n_features = 1
            n_samples = X.shape[0]
            X = X.reshape(-1,1)
        else:
            n_samples, n_features = X.shape
        k = 2 * self.n_harmonics + int(self.intercept)
        X_fourier = np.zeros((n_samples, n_features * k))
        if self.intercept:
            X_fourier[:,0] = 1
        
        for i in range(n_features):
            # For each feature, create its Fourier terms
            col_idx = (i * k) + int(self.intercept)
            

            # Add sine and cosine terms for each harmonic
            for h in range(1, self.n_harmonics + 1):
                X_fourier[:, col_idx] = np.sin(h *2*np.pi* X[:, i])
                X_fourier[:, col_idx + 1] = np.cos(h *2*np.pi* X[:, i])
                col_idx += 2
                
        return X_fourier
    
    def mk_design_matrix(self,x,y)-> Tuple[ArrayLike,ArrayLike]:
        '''
        returns x,y design matrix
        '''
        return self.extract_harmonics(x),y
    def fit(self,x,y):
        '''
        expects transformed design matrix
        '''
        self.b_= np.linalg.lstsq(x,y)[0]
        self.x_ = x
        self.y_ = y
    def fit_transform(self,x,y):
        '''
        use if input is not already transformed into harmonics
        '''
        X,Y = self.mk_design_matrix(x,y)
        return self.fit(X,Y)
    def predict(self,x):
        return x@self.b_.conj().T
    def L(self,yhat,y):
        return (np.abs(y-yhat)**2).mean()
    
# use this pattern fo the other classes -- we will do model init, mk design matrix, model fit, then have separate function to rotate& predict
# so we will do model.init(), x,y= model.mk_design_matrix(), then zx=dft(x), zy=dft(y) then cv(zx,zy)

# make harmonic transform its own transform class, set least squared as basic harmonic model class and then use pipeline to do combinations for algos:
#[transform -> cv[ model fit-> model predict ]]
#[transform -> dft -> cv[model fit -> model predict]] 
