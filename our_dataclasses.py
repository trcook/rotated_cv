from pydantic import BaseModel,RootModel
from pydantic.dataclasses import dataclass
from textwrap import dedent
from itertools import product
from typing import Optional
import json



#%% dataclasses

@dataclass
class EXP():
    sigma: Optional[float] = 0.0
    theta:Optional[float] = 0.0
    phi: Optional[float] = 0.0
    n: Optional[int] = 200

@dataclass
class HPExpConfig(EXP):
    lam:int = 1600


@dataclass
class HPExpResult(HPExpConfig):
    cv: float = None
    true_cv: float = None # indicates how close to trend component
    true_mse: float = None # indicates variance of disturbance term (lower limit of cv score without overfitting)

@dataclass 
class KRExpConfig(EXP):
    w:Optional[float] = .01

@dataclass
class KRExpResult(KRExpConfig):
    cv: float = None
    true_cv: float = None # indicates how close to trend component
    true_mse: float = None # indicates variance of disturbance term (lower limit of cv score without overfitting)

@dataclass
class HRExpConfig(EXP):
    n_harmonics:int = 3

@dataclass
class HRExpResult(HRExpConfig):
    """
    Note, where experiment is fft, y and yhats are fft permuted (in frequency domain)
    """
    cv: float = None
    true_cv: float = None
    true_mse: float = None
