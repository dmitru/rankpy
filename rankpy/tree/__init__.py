''' 
Module containing implementation of Online Random Forest based
on 'Consistency of Online Random Forests' by Misha Denil et al.
(2013).
'''

from .tree import OnlineRandomForestRegressor

__all__ = ['OnlineRandomForestRegressor']
