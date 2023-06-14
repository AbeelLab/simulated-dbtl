import numpy as np

def add_homoschedastic_noise(values,percentage):
    """add noise to the training set, to see the effect of noise models on performance"""
    mean_percentage=percentage
    newvalue=np.random.normal(values,mean_percentage)
    newvalue[newvalue<0]=0
    return newvalue
    
def add_heteroschedastic_noise(values,percentage):
    """add noise to the training set, to see the effect of noise models on performance"""
    mean_percentage=percentage*values
    value=np.random.normal(values,mean_percentage)
    value[value<0]=0
    return value
    